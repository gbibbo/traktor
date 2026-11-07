"""
PURPOSE: Tests de src/v4/pipeline/tag_downloads.py: la metadata de los temas descargados (Beatport y Vocal)
         reutiliza beatport_lookup / write_beatport_tags / tag_vocals; si Beatport pide la verificación
         anti-robots, los temas quedan en espera y se completan en la próxima corrida.
CHANGELOG:
  - 2026-11-07: Creación.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.pipeline import tag_downloads  # noqa: E402


def make_files(folder: Path, names):
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for name in names:
        path = folder / name
        path.write_bytes(b"x")
        files.append(path)
    return files


def fake_catalog(files):
    rows = [{"track_uid": f"uid-{p.stem}", "source_path": str(p), "rel_path": p.name} for p in files]
    return pd.DataFrame(rows), {r["track_uid"]: {} for r in rows}


def run(argv, *, available=(True, ""), beatport=None):
    looked_up = []

    def beatport_tags(catalog, rels, client, backup):
        looked_up.append(sorted(Path(p).name for p in catalog["source_path"]))
        if beatport:
            raise beatport
        return pd.DataFrame([{"track_uid": u, "source_path": p, "level": "A", "action": "beatport",
                              "new_genre": "Techno (Peak Time / Driving)", "status": "written"}
                             for u, p in zip(catalog["track_uid"], catalog["source_path"])])

    with mock.patch.object(tag_downloads, "catalog_rows", side_effect=fake_catalog), \
            mock.patch.object(tag_downloads, "beatport_available", return_value=available), \
            mock.patch.object(tag_downloads, "beatport_tags", side_effect=beatport_tags), \
            mock.patch.object(tag_downloads, "vocal_tags", return_value=pd.DataFrame()) as vocal, \
            mock.patch.object(sys, "argv", ["tag_downloads.py", *argv]):
        code = tag_downloads.main()
    return code, looked_up, vocal


def test_a_folder_gets_beatport_and_vocal(tmp_path):
    folder = tmp_path / "Workout Delft"
    make_files(folder, ["A - One.mp3", "B - Two.flac"])
    (folder / "Workout Delft.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
    code, looked_up, vocal = run([str(folder), "--backup-dir", str(tmp_path / "bk")])
    assert code == 0
    assert looked_up == [["A - One.mp3", "B - Two.flac"]]  # el .m3u8 no
    vocal.assert_called_once()


def test_when_beatport_asks_for_the_anti_bot_check_the_tracks_wait_for_the_next_run(tmp_path):
    first = make_files(tmp_path / "Lista 1", ["A - One.mp3"])
    second = make_files(tmp_path / "Lista 2", ["C - Three.mp3"])
    backup = tmp_path / "bk"
    code, looked_up, vocal = run([str(first[0]), "--backup-dir", str(backup)],
                                 available=(False, "pide la verificación anti-robots de Cloudflare"))
    assert code == 1 and looked_up == []
    vocal.assert_called_once()  # las voces no esperan a Beatport
    pending = backup / tag_downloads.PENDING_NAME
    assert "A - One.mp3" in pending.read_text(encoding="utf-8")
    # La próxima descarga (otra playlist) completa también lo que esperaba, y la lista de espera se vacía.
    code, looked_up, _ = run([str(second[0]), "--backup-dir", str(backup)])
    assert code == 0
    assert looked_up == [["A - One.mp3", "C - Three.mp3"]]
    assert not pending.exists()


def test_beatport_blocking_in_the_middle_keeps_the_tracks_waiting(tmp_path):
    files = make_files(tmp_path / "Lista", ["A - One.mp3"])
    backup = tmp_path / "bk"
    code, _, _ = run([str(files[0]), "--backup-dir", str(backup)],
                     beatport=tag_downloads.BeatportBlocked("5 intentos"))
    assert code == 1
    assert "A - One.mp3" in (backup / tag_downloads.PENDING_NAME).read_text(encoding="utf-8")
