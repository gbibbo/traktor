"""
PURPOSE: Tests de src/v4/common/drive_api.py (Google Drive por su API, sin rclone, para Colab) contra un Drive
         falso en memoria que responde los mismos endpoints: protocolo de gpuq.storage.Remote (subida a .part,
         renombrado, verificación por tamaño y MD5), listado con páginas, carpetas creadas al subir, reemplazo
         de un archivo existente, borrado, reintentos ante 429, renovación del token ante 401, y que ningún
         error lleve una credencial. Necesita el paquete gpuq (clon gpuq-personal o GPUQ_HOME); sin él se salta.
CHANGELOG:
  - 2026-11-13: Creación.
"""
import hashlib
import io
import json
import re
import sys
import urllib.parse
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common import drive_api, remote_storage as rs  # noqa: E402

try:
    storage = rs.import_gpuq_storage()
except ImportError:  # pragma: no cover
    pytest.skip("gpuq no está disponible", allow_module_level=True)

SECRET = "s3cr3t-value"


class FakeDrive:
    """In-memory Drive API v3: the calls drive_api makes, and nothing else."""

    def __init__(self, page=2):
        self.items = {"root": {"id": "root", "name": "", "mimeType": drive_api.FOLDER, "parents": []}}
        self.sessions, self.n, self.page, self.calls = {}, 0, page, []
        self.fail = []  # statuses to answer, once each, before the next API call

    def _id(self):
        self.n += 1
        return f"id{self.n}"

    def _meta(self, it):
        out = {k: it[k] for k in ("id", "name", "mimeType")}
        if it["mimeType"] != drive_api.FOLDER:
            out.update(size=str(len(it["data"])), md5Checksum=hashlib.md5(it["data"]).hexdigest())
        return out

    def __call__(self, method, url, headers, body):
        data = body.read() if hasattr(body, "read") else body
        self.calls.append((method, url))
        if url == drive_api.TOKEN_URL:
            form = urllib.parse.parse_qs(data.decode())
            assert form["client_secret"] == [SECRET] and form["refresh_token"] == ["r"]
            return 200, {}, io.BytesIO(json.dumps({"access_token": "fresh", "expires_in": 3600}).encode())
        if self.fail:
            return self.fail.pop(0), {}, io.BytesIO(b"{}")
        if headers.get("Authorization") != "Bearer fresh":
            return 401, {}, io.BytesIO(b"{}")
        u = urllib.parse.urlparse(url)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        path = u.path
        if u.netloc == "upload.fake":
            meta = self.sessions.pop(path)
            it = self.items.setdefault(meta.get("id") or self._id(), {"mimeType": "application/octet-stream"})
            it.update({k: v for k, v in meta.items() if k != "id"}, data=data)
            it["id"] = next(k for k, v in self.items.items() if v is it)
            return 200, {}, io.BytesIO(json.dumps(self._meta(it)).encode())
        if path.startswith("/upload/drive/v3/files"):
            meta = json.loads(data or b"{}")
            fid = path.rsplit("/", 1)[-1] if method == "PATCH" else None
            sid = f"/s{len(self.sessions) + self.n}"
            self.sessions[sid] = {**meta, **({"id": fid} if fid else {})}
            return 200, {"Location": f"https://upload.fake{sid}"}, io.BytesIO(b"")
        if path == "/drive/v3/files" and method == "GET":
            q = qs["q"]
            parent = re.match(r"'([^']+)' in parents", q).group(1)
            name = re.search(r"name = '((?:[^'\\]|\\.)*)'", q)
            hits = [it for it in self.items.values() if parent in it.get("parents", []) and
                    (name is None or it["name"] == name.group(1).replace("\\'", "'").replace("\\\\", "\\"))]
            start = int(qs.get("pageToken", 0))
            size = self.page if "pageToken" in qs or not name else 10
            chunk = hits[start:start + size]
            out = {"files": [self._meta(it) for it in chunk]}
            if start + size < len(hits) and not name:
                out["nextPageToken"] = str(start + size)
            return 200, {}, io.BytesIO(json.dumps(out).encode())
        if path == "/drive/v3/files" and method == "POST":
            meta = json.loads(data)
            fid = self._id()
            self.items[fid] = {"id": fid, **meta}
            return 200, {}, io.BytesIO(json.dumps(self._meta(self.items[fid])).encode())
        fid = path.rsplit("/", 1)[-1]
        if fid not in self.items:
            return 404, {}, io.BytesIO(b"{}")
        it = self.items[fid]
        if method == "GET" and qs.get("alt") == "media":
            return 200, {}, io.BytesIO(it["data"])
        if method == "DELETE":
            del self.items[fid]
            return 204, {}, io.BytesIO(b"")
        if method == "PATCH":
            it.update(json.loads(data))
            if "addParents" in qs:
                it["parents"] = [qs["addParents"]]
            return 200, {}, io.BytesIO(json.dumps(self._meta(it)).encode())
        raise AssertionError(f"unexpected call {method} {url}")


def conf(tmp_path, **over) -> Path:
    sec = {"type": "drive", "scope": "drive.file", "client_id": "cid", "client_secret": SECRET,
           "token": json.dumps({"access_token": "stale", "refresh_token": "r"}), **over}
    p = tmp_path / "rclone.conf"
    p.write_text("[erclm]\n" + "".join(f"{k} = {v}\n" for k, v in sec.items() if v is not None), encoding="utf-8")
    return p


@pytest.fixture
def world(tmp_path):
    fake = FakeDrive()
    remote = drive_api.make_drive_remote(storage, "erclm:traktor_ml", conf(tmp_path), http=fake, sleep=lambda s: None)
    remote.backoff_s = 0
    return fake, remote, tmp_path


def test_put_get_ls_stat_delete_like_any_gpuq_remote(world):
    fake, remote, tmp = world
    src = tmp / "a.npz"
    src.write_bytes(b"x" * 1000)
    info = remote.put(src, "v/shards/a.npz")
    assert info == {"size": 1000, "md5": hashlib.md5(b"x" * 1000).hexdigest()}
    assert remote.stat("v/shards/a.npz") == info and remote.exists("v/shards/a.npz")
    assert not remote.exists("v/shards/a.npz.part")                      # renamed, not copied
    assert remote.get("v/shards/a.npz", tmp / "back.npz").read_bytes() == b"x" * 1000
    remote.put_json({"k": 1}, "v/meta.json")
    assert remote.get_json("v/meta.json") == {"k": 1} and remote.get_json("v/none.json") is None
    for i in range(5):                                                   # the fake answers 2 files per page
        (tmp / f"p{i}.txt").write_text(str(i))
        remote.put(tmp / f"p{i}.txt", f"v/parc/p{i}.txt")
    assert list(remote.ls("v/parc")) == [f"p{i}.txt" for i in range(5)] and remote.ls("v/nada") == {}
    assert list(remote.ls("v")) == ["meta.json"]                          # folders are not listed
    src.write_bytes(b"y" * 10)                                           # same name: replaced, not duplicated
    remote.put(src, "v/shards/a.npz")
    assert remote.stat("v/shards/a.npz")["size"] == 10 and len(remote.ls("v/shards")) == 1
    remote.delete("v/shards/a.npz")
    assert not remote.exists("v/shards/a.npz") and remote.ls("v/shards") == {}
    with pytest.raises(FileNotFoundError):
        remote.get("v/shards/a.npz", tmp / "x")


def test_retries_rate_limits_and_refreshes_an_expired_token(world):
    fake, remote, tmp = world
    (tmp / "f").write_text("hola")
    remote.put(tmp / "f", "f.txt")
    fake.fail = [429, 503]
    assert remote.stat("f.txt")["size"] == 4
    remote.api.access_token = "expired"                                  # the API answers 401: refresh, again
    assert remote.stat("f.txt")["size"] == 4 and remote.api.access_token == "fresh"
    fake.fail = [400]
    with pytest.raises(storage.StorageError) as e:
        remote.ls("")
    assert "HTTP 400" in str(e.value) and SECRET not in str(e.value) and "fresh" not in str(e.value)


def test_config_must_be_a_drive_remote_with_its_own_client(tmp_path):
    with pytest.raises(drive_api.DriveError):
        drive_api.read_remote(conf(tmp_path, client_id=None), "erclm")   # rclone's shared client: refused
    with pytest.raises(drive_api.DriveError):
        drive_api.read_remote(conf(tmp_path, type="s3"), "erclm")
    with pytest.raises(drive_api.DriveError):
        drive_api.read_remote(conf(tmp_path), "otro")
    assert drive_api.rclone_config_path({"RCLONE_CONFIG": "/x/r.conf"}) == Path("/x/r.conf")
    assert drive_api.rclone_config_path({"HOME": "/h"}).as_posix().endswith(".config/rclone/rclone.conf")


def test_open_remote_uses_the_drive_api_only_when_asked(tmp_path):
    env = {"TRAKTOR_REMOTE": "erclm:traktor_ml", "RCLONE_CONFIG": str(conf(tmp_path)), "RCLONE": "rclone"}
    assert type(rs.open_remote({**env, "TRAKTOR_DRIVE_API": "1"})).__name__ == "DriveRemote"
    assert type(rs.open_remote(env)).__name__ == "RcloneRemote"
    assert type(rs.open_remote({**env, "TRAKTOR_DRIVE_API": "0"})).__name__ == "RcloneRemote"
    assert type(rs.open_remote({**env, "TRAKTOR_REMOTE": str(tmp_path / "local")})).__name__ == "LocalRemote"
