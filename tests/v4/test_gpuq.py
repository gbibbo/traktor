"""
PURPOSE: Tests de la integración con gpuq: raíz del almacenamiento remoto (src/v4/common/remote_storage.py),
         el envoltorio para encolar (tools/gpuq/cola.py) y la prueba de punta a punta del Drive
         (tools/gpuq/smoke_drive.py) contra una carpeta local en lugar del Drive. La ida y vuelta necesita
         el paquete gpuq (clon gpuq-personal o GPUQ_HOME); sin él se salta.
CHANGELOG:
  - 2026-11-09: Creación (puesta en marcha de gpuq en este repo).
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common import remote_storage as rs  # noqa: E402
from tools.gpuq import cola, smoke_drive  # noqa: E402


def test_remote_root_resolution(tmp_path):
    assert rs.remote_root({}) == "erclm:traktor_ml"
    assert rs.remote_root({"GPUQ_REMOTE": "erclm:gpuq-test"}) == "erclm:traktor_ml"
    assert rs.remote_root({"GPUQ_REMOTE": "otro:a/b"}) == "otro:traktor_ml"
    assert rs.remote_root({"TRAKTOR_REMOTE": "gd:carpeta/", "GPUQ_REMOTE": "erclm:x"}) == "gd:carpeta"
    assert rs.remote_root({"GPUQ_REMOTE": str(tmp_path)}) == str(tmp_path / "traktor_ml")
    assert rs.is_local(str(tmp_path)) and not rs.is_local("erclm:traktor_ml")


def test_find_rclone_prefers_env():
    assert rs.find_rclone({"RCLONE": "C:/x/rclone.exe"}) == "C:/x/rclone.exe"


def test_repo_slug_and_add_arguments():
    assert cola.repo_slug("https://github.com/gbibbo/traktor.git") == "gbibbo/traktor"
    assert cola.repo_slug("https://github.com/gbibbo/traktor") == "gbibbo/traktor"
    assert cola.repo_slug("git@github.com:gbibbo/traktor.git") == "gbibbo/traktor"
    with pytest.raises(ValueError):
        cola.repo_slug("https://gitlab.com/x/y.git")
    full = cola.build_add(["j1", "--cmd", "python a.py"], "o/r", "abc")
    assert full == ["add", "j1", "--cmd", "python a.py", "--repo", "o/r", "--ref", "abc"]
    kept = cola.build_add(["j1", "--cmd", "x", "--ref=main"], "o/r", "abc")
    assert kept.count("--ref") == 0 and "--ref=main" in kept and "--repo" in kept


def _need_gpuq():
    try:
        rs.import_gpuq_storage()
    except ImportError as exc:
        pytest.skip(str(exc))


def test_smoke_round_trip_on_a_local_folder(tmp_path, monkeypatch):
    _need_gpuq()
    root = tmp_path / "remote"
    monkeypatch.setenv("TRAKTOR_REMOTE", str(root))
    monkeypatch.setattr(smoke_drive, "gpu_matmul", lambda n: {"torch": "x", "cuda": True, "n": n, "matmul_ms": 1.0})
    assert smoke_drive.main(["prepare", "--run", "t1"]) == 0
    assert smoke_drive.main(["run", "--run", "t1", "--n", "8"]) == 0
    assert smoke_drive.main(["check", "--run", "t1"]) == 0
    inp = json.loads((root / "smoke" / "t1" / "input.json").read_text(encoding="utf-8"))
    res = json.loads((root / "smoke" / "t1" / "result.json").read_text(encoding="utf-8"))
    assert res["nonce_echo"] == inp["nonce"] and len(inp["nonce"]) == 32
    assert smoke_drive.main(["prepare", "--run", "t1"]) == 2  # never overwrites a finished run


def test_smoke_run_fails_without_input_or_without_cuda(tmp_path, monkeypatch):
    _need_gpuq()
    monkeypatch.setenv("TRAKTOR_REMOTE", str(tmp_path / "remote"))
    assert smoke_drive.main(["run", "--run", "nada"]) == 2
    monkeypatch.setattr(smoke_drive, "gpu_matmul", lambda n: {"torch": "x", "cuda": False})
    assert smoke_drive.main(["prepare", "--run", "cpu"]) == 0
    assert smoke_drive.main(["run", "--run", "cpu"]) == 3
    assert smoke_drive.main(["run", "--run", "cpu", "--allow-cpu"]) == 0
    assert smoke_drive.main(["check", "--run", "cpu"]) == 1
    assert smoke_drive.main(["check", "--run", "cpu", "--allow-cpu"]) == 0


def test_gpu_matmul_runs_on_this_machine():
    pytest.importorskip("torch")
    out = smoke_drive.gpu_matmul(n=32)
    assert out["device"] in ("cpu", "cuda") and out["n"] == 32 and out["checksum"] > 0
