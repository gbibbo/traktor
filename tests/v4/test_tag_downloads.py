"""
PURPOSE: Tests de src/v4/pipeline/tag_downloads.py: la metadata y el embedding de los temas descargados (Beatport,
         Vocal, clap_full) reutilizan beatport_lookup / write_beatport_tags / tag_vocals / file_embeddings; si
         Beatport pide la verificación anti-robots, los temas quedan en espera; --watch procesa tema por tema.
CHANGELOG:
  - 2026-11-07: Creación.
  - 2026-11-07: Embedding y modo --watch (tema por tema mientras la descarga sigue).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
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
    """main() con Beatport, CLAP y la escritura simulados. Devuelve (código, búsquedas, voces, embeddings)."""
    looked_up, embedded = [], []

    def beatport_tags(catalog, rels, client, backup, predict=None):
        looked_up.append(sorted(Path(p).name for p in catalog["source_path"]))
        if beatport:
            raise beatport
        return pd.DataFrame([{"track_uid": u, "source_path": p, "level": "A", "action": "beatport",
                              "new_genre": "Techno (Peak Time / Driving)", "status": "written"}
                             for u, p in zip(catalog["track_uid"], catalog["source_path"])])

    def vocal_tags(catalog, cache, backup):
        return pd.DataFrame({"track_uid": list(catalog["track_uid"]), "is_vocal": [True] * len(catalog)})

    with mock.patch.object(tag_downloads, "catalog_rows", side_effect=fake_catalog), \
            mock.patch.object(tag_downloads, "beatport_available", return_value=available), \
            mock.patch.object(tag_downloads, "beatport_tags", side_effect=beatport_tags), \
            mock.patch.object(tag_downloads, "vocal_tags", side_effect=vocal_tags) as vocal, \
            mock.patch.object(tag_downloads.Models, "vocal_windows"), \
            mock.patch.object(tag_downloads.Models, "clap_full", return_value=np.zeros(512, dtype=np.float32)), \
            mock.patch.object(tag_downloads.Models, "maest") as maest, \
            mock.patch.object(tag_downloads, "embed_in_file",
                              side_effect=lambda path, uid, vec: embedded.append(Path(path).name) or True), \
            mock.patch.object(sys, "argv", ["tag_downloads.py", *argv]):
        code = tag_downloads.main()
    run.maest_calls = maest.call_count
    return code, looked_up, vocal, embedded


def test_a_folder_gets_beatport_vocal_and_the_embedding(tmp_path):
    folder = tmp_path / "Workout Delft"
    make_files(folder, ["A - One.mp3", "B - Two.flac"])
    (folder / "Workout Delft.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
    code, looked_up, vocal, embedded = run([str(folder), "--backup-dir", str(tmp_path / "bk")])
    assert code == 0
    assert looked_up == [["A - One.mp3", "B - Two.flac"]]  # el .m3u8 no
    vocal.assert_called_once()
    assert sorted(embedded) == ["A - One.mp3", "B - Two.flac"]
    assert run.maest_calls == 2  # MAEST en caché para «Agregar música nueva»


def _library_for_genres(art: Path, labels):
    """Biblioteca falsa: dos géneros separables en l7 ‖ clap, estilos electrónicos (el gate no salta)."""
    rng = np.random.default_rng(0)
    uids = [f"u{i:03d}" for i in range(80)]
    genre = ["Techno (Peak Time / Driving)" if i % 2 else "Deep House" for i in range(80)]
    l7 = rng.standard_normal((80, 8)).astype(np.float32) + np.array([[4 if i % 2 else -4] for i in range(80)])
    clap = rng.standard_normal((80, 4)).astype(np.float32)
    styles = np.full((80, len(labels)), 0.01, dtype=np.float32)
    styles[:, 0] = 0.9
    for rep, mat in (("maesthf_full_l7", l7), ("clap_full", clap), ("maesthf_full_styles", styles)):
        (art / "representations" / rep).mkdir(parents=True)
        np.save(art / "representations" / rep / "embeddings.npy", mat)
        (art / "representations" / rep / "track_uids.json").write_text(json.dumps(uids))
    (art / "features").mkdir()
    pd.DataFrame({"track_uid": uids, "level": "A", "confirmed": True, "bp_genre": genre,
                  "bp_artists": [f"Artista {i}" for i in range(80)], "cur_artist": ""}).to_parquet(
        art / "features" / "beatport.parquet", index=False)


def test_unconfirmed_tracks_get_the_library_genre_model_with_the_library_rules(tmp_path):
    """Gabriel, 2026-11-07: los temas nuevos tienen que llevar la metadata de la colección. Lo que Beatport no
    confirma lleva el género del clasificador de la biblioteca si la confianza es >= 0.6 y el gate no salta."""
    from src.v4.pipeline import write_beatport_tags
    labels = ["Electronic---Techno", "Electronic---House", "Pop---Ballad", "Rock---Punk"]
    _library_for_genres(tmp_path, labels)
    model = tag_downloads.GenreModel(tmp_path, labels)
    techno = model.predict(np.full(8, 4.0, np.float32), np.zeros(4, np.float32), np.array([0.9, 0.1, 0.01, 0.01]))
    assert techno["model_source"] == "classifier" and techno["model_genre"] == "Techno (Peak Time / Driving)"
    assert techno["model_conf"] >= 0.6
    unsure = model.predict(np.zeros(8, np.float32), np.zeros(4, np.float32), np.array([0.9, 0.1, 0.01, 0.01]))
    pop = model.predict(np.full(8, 4.0, np.float32), np.zeros(4, np.float32), np.array([0.2, 0.1, 0.8, 0.01]))
    assert pop == {"model_source": "gate_discogs", "model_conf": None, "model_genre": "Pop"}
    # build_plan (la regla de la biblioteca): solo el clasificador seguro escribe Genre
    bp = pd.DataFrame({"track_uid": ["t1", "t2", "t3"], "level": ["D", "C", "D"], "rel_path": "x",
                       "confirmed": False})
    pred = pd.DataFrame([{"track_uid": "t1", **techno}, {"track_uid": "t2", **unsure}, {"track_uid": "t3", **pop}])
    catalog = pd.DataFrame({"track_uid": ["t1", "t2", "t3"], "source_path": ["a.mp3", "b.mp3", "c.mp3"]})
    plan = write_beatport_tags.build_plan(bp, pred, catalog).set_index("track_uid")
    assert plan.at["t1", "action"] == "model" and plan.at["t1", "new_genre"] == "Techno (Peak Time / Driving)"
    assert plan.at["t3", "action"] == "keep"
    assert unsure["model_conf"] < 0.6 and plan.at["t2", "action"] == "keep"


def test_when_beatport_asks_for_the_anti_bot_check_the_tracks_wait_for_the_next_run(tmp_path):
    first = make_files(tmp_path / "Lista 1", ["A - One.mp3"])
    second = make_files(tmp_path / "Lista 2", ["C - Three.mp3"])
    backup = tmp_path / "bk"
    code, looked_up, vocal, embedded = run([str(first[0]), "--backup-dir", str(backup)],
                                           available=(False, "pide la verificación anti-robots de Cloudflare"))
    assert code == 1 and looked_up == []
    vocal.assert_called_once()  # las voces y el embedding no esperan a Beatport
    assert embedded == ["A - One.mp3"]
    pending = backup / tag_downloads.PENDING_NAME
    assert "A - One.mp3" in pending.read_text(encoding="utf-8")
    # La próxima descarga (otra playlist) completa también lo que esperaba, y la lista de espera se vacía.
    code, looked_up, _, _ = run([str(second[0]), "--backup-dir", str(backup)])
    assert code == 0
    assert looked_up == [["A - One.mp3", "C - Three.mp3"]]
    assert not pending.exists()


def test_beatport_blocking_in_the_middle_keeps_the_tracks_waiting(tmp_path):
    files = make_files(tmp_path / "Lista", ["A - One.mp3"])
    backup = tmp_path / "bk"
    code, _, _, _ = run([str(files[0]), "--backup-dir", str(backup)],
                        beatport=tag_downloads.BeatportBlocked("5 intentos"))
    assert code == 1
    assert "A - One.mp3" in (backup / tag_downloads.PENDING_NAME).read_text(encoding="utf-8")


def test_watch_processes_each_track_as_it_arrives_and_ends_with_the_queue(tmp_path):
    # Gabriel, 2026-11-07: la metadata se hacía recién al final de toda la descarga (casi 5 minutos para 10
    # temas). Ahora cada tema se completa apenas está bajado o grabado, mientras sigue el resto.
    files = make_files(tmp_path / "run", ["A - One.mp3", "B - Two.mp3"])
    queue, done = tmp_path / "cola.jsonl", tmp_path / "hechos.jsonl"
    queue.write_text("".join(json.dumps({"path": str(p)}) + "\n" for p in files)
                     + json.dumps({"path": str(tmp_path / "ya-no-esta.mp3")}) + "\n"
                     + json.dumps({"end": True}) + "\n" + '{"path": "a me', encoding="utf-8")  # última a medias
    calls = []

    def process(paths, **kwargs):
        calls.append([p.name for p in paths])
        return [{"path": str(paths[0]), "name": paths[0].name, "genre": "Techno", "vocal": False,
                 "embedding": "escrito"}]

    with mock.patch.object(tag_downloads, "process", side_effect=process):
        assert tag_downloads.watch(queue, done) == 0
    assert calls == [["A - One.mp3"], ["B - Two.mp3"]]  # uno por vez
    results = [json.loads(line) for line in done.read_text(encoding="utf-8").splitlines()]
    assert [r["name"] for r in results] == ["A - One.mp3", "B - Two.mp3", "ya-no-esta.mp3"]
    assert results[2]["error"] == "ya no está"
    # Retomada: lo que ya está en HECHOS no se procesa de nuevo.
    with mock.patch.object(tag_downloads, "process", side_effect=process):
        assert tag_downloads.watch(queue, done) == 0
    assert len(calls) == 2
