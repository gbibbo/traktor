"""
PURPOSE: Tests de las funciones puras de tools/dj_sets (piloto de trayectorias de DJs): normalización y
         emparejamiento de títulos de MusicBrainz con YouTube, límites por duración acumulada, novedad y
         picos de una secuencia de embeddings, precisión/recuperación/F de cortes, el resumen de
         MusicBrainz y la lectura en bloques de clap_blocks.
CHANGELOG:
  - 2026-11-07: Creación.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.dj_sets import piloto as pl  # noqa: E402
from tools.dj_sets.musicbrainz_mixes import release_summary  # noqa: E402


def test_norm_and_match_titles():
    assert pl.norm("Oliver Lieb - Violet Phophorous (Original Mix) [Official Audio]") == "oliver lieb violet phophorous"
    tracks = [{"artista": "Oliver Lieb", "titulo": "Violet Phophorous"},
              {"artista": "Drag & Drop", "titulo": "Discolabirinto"},
              {"artista": "Nadie", "titulo": "Nada Que Ver"}]
    entries = [{"titulo": "Drag & Drop - Discolabirinto (Original Mix) [Official Audio]"},
               {"titulo": "Oliver Lieb - Violet Phophorous (Original Mix) [Official Audio]"},
               {"titulo": "John Digweed - Live (Continuous Mix 1) [Official Audio]"}]
    assert pl.match_tracks(tracks, entries) == [1, 0, None]
    # el artista en otro alfabeto o con otros créditos: decide el título
    greek = [{"artista": "Στέλιος Βασιλούδης", "titulo": "Blood Orange"},
             {"artista": "Digweed & Muir", "titulo": "Fanfare (Darren Emerson Osaka rework)"},
             {"artista": "Verche", "titulo": "Raving At The Acropolis"}]
    yt = [{"titulo": "Emerson, Digweed & Muir - Fanfare (Live) [Official Audio]"},
          {"titulo": "Stelios Vassiloudis - Blood Orange (Live) [Official Audio]"},
          {"titulo": "Verche - Rapid Eye Movement (Original Mix) [Official Audio]"}]
    assert pl.match_tracks(greek, yt) == [1, 0, None]


def test_bounds_accumulate_durations():
    assert pl.bounds([60000, 90000, None]) == [(0.0, 60.0), (60.0, 150.0), (150.0, 150.0)]


def test_novelty_peaks_find_the_changes_and_prf_scores_them():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16))
    B = np.concatenate([np.tile(c, (10, 1)) + 0.01 * rng.normal(size=(10, 16)) for c in centers])
    nov = pl.novelty(B, k=2)
    assert pl.peaks(nov, min_gap=4, n_peaks=2) == [10, 20]
    assert pl.peaks(nov, min_gap=4, thr=0.5) == [10, 20]
    assert pl.prf([300.0, 610.0, 900.0], [300.0, 600.0], tol=30) == {"precision": 0.667, "recuperacion": 1.0, "f": 0.8}
    assert pl.prf([], [300.0], tol=30)["f"] == 0.0


def test_release_summary_keeps_discs_and_lengths():
    r = {"id": "x", "title": "Live", "date": "2018", "artist-credit": [{"name": "John Digweed"}],
         "media": [{"position": 1, "format": "CD", "tracks": [
             {"position": "1", "title": "A", "length": 61000, "recording": {"id": "r1"},
              "artist-credit": [{"name": "X", "joinphrase": " & "}, {"name": "Y"}]}]}]}
    s = release_summary(r)
    assert s["artista"] == "John Digweed" and s["media"][0]["pistas"][0] == {
        "posicion": 1, "artista": "X & Y", "titulo": "A", "ms": 61000, "recording": "r1"}


def test_stream_blocks_reads_whole_blocks_and_keeps_a_long_tail(tmp_path):
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return
    from tools.dj_sets import clap_blocks as cb
    wav = tmp_path / "t.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=75:sample_rate=8000",
                    str(wav)], check=True)
    sr, dur = cb.probe(wav)
    blocks = list(cb.stream_blocks(wav, sr))
    assert sr == 8000 and round(dur) == 75
    assert [t for t, _ in blocks] == [0.0, 30.0, 60.0] and len(blocks[-1][1]) == 15 * 8000


def test_youtube_playlists_are_validated_by_durations_against_musicbrainz():
    from tools.dj_sets.youtube_buscar import match_tracks_by_duration, validate
    rel = {"media": [{"pistas": [{"ms": 300000}, {"ms": 420000}, {"ms": 360000}]},
                     {"pistas": [{"ms": 500000}, {"ms": 280000}]}]}
    # cortada por tema (con un video de más al principio y una duración a 2 s)
    per_track = [{"duration": 60}, {"duration": 301}, {"duration": 418}, {"duration": 360},
                 {"duration": 500}, {"duration": 282}]
    v = validate(rel, per_track)
    assert v["tipo"] == "por_tema" and v["cobertura_temas"] == 1.0 and v["archivo_por_pista"] == [1, 2, 3, 4, 5]
    # mezcla continua por disco: 1080 s y 780 s
    v = validate(rel, [{"duration": 1085}, {"duration": 779}, {"duration": 410}])
    assert v["tipo"] == "continua" and v["archivo_por_disco"] == [0, 1] and v["diferencia_por_disco_s"] == [5.0, -1.0]
    # un disco de otro DJ que dura 15 s más no cuenta (los CD duran casi todos lo mismo)
    assert validate(rel, [{"duration": 1095}, {"duration": 795}])["tipo"] is None
    # otra cosa: nada coincide
    assert validate(rel, [{"duration": 100}, {"duration": 200}])["tipo"] is None
    # dos pistas de la misma duración: se respeta el orden de la playlist
    assert match_tracks_by_duration([300, 300], [300, 301]) == [0, 1]


def test_traktor_history_dates_and_sets_split_after_20_minutes_off_air():
    import datetime as dt
    from tools.dj_sets.sets_traktor import split_sets, start_of
    assert start_of("132778520", "8989") == dt.datetime(2026, 10, 24, 2, 29, 49)
    assert start_of("x", "1") is None
    t = dt.datetime(2026, 2, 4, 20, 0)
    rows = [{"inicio": t, "segundos_al_aire": 300},
            {"inicio": t + dt.timedelta(minutes=4), "segundos_al_aire": 400},
            {"inicio": t + dt.timedelta(minutes=30), "segundos_al_aire": 300},    # 19,3 min sin nada: sigue
            {"inicio": t + dt.timedelta(minutes=56), "segundos_al_aire": 300}]    # 21 min sin nada: set nuevo
    assert [len(s) for s in split_sets(rows)] == [3, 1]


def test_bench_breaks_ties_at_random_and_resamples_whole_sets():
    from types import SimpleNamespace as NS
    from src.v4.evaluation import banco_sets as bs

    class FakeRec:                                   # todos empatados: el orden es el de los candidatos
        def __init__(self, uids):
            self.candidates = [NS(uid=u) for u in uids]
            self.weights = None

        def compute(self, principal, played):
            return [], {"uids": [c.uid for c in self.candidates]}

    uids = [f"t{i}" for i in range(100)]
    rec = FakeRec(uids)
    items = {u: NS(path=None, original=u, rel=u, bpm=120.0) for u in uids}
    sets = {"s1": [{"track_uid": "t0", "posicion": "0"}, {"track_uid": "t99", "posicion": "1"}]}
    ranks = {bs.evaluate(rec, items, sets, {"clap": 1}, seed=s)["rr_medio"] for s in range(10)}
    assert len(ranks) > 1                            # sin desempate al azar, t99 quedaría siempre 100.º
    assert [c.uid for c in rec.candidates] == uids    # el orden original se restituye
    lo, hi = bs.bootstrap_sets({"a": [1, 1, 0], "b": [0, 0, 0, 0], "c": []}, n=200)
    assert 0.0 <= lo <= hi <= 1.0


def test_d7_download_keeps_only_files_that_still_match_by_duration():
    from tools.dj_sets.bajar_d7 import wanted
    release = {"media": [{"disco": 1, "pistas": [{"ms": 300000, "posicion": 1}, {"ms": 400000, "posicion": 2}]},
                         {"disco": 2, "pistas": [{"ms": 500000, "posicion": 1}]}]}
    entries = [{"id": "a", "duration": 701}, {"id": "b", "duration": 530}, {"id": "c", "duration": 300},
               {"id": "d", "duration": 401}, {"id": "e", "duration": 499}]
    cont = {"youtube": {"tipo": "continua", "archivo_por_disco": [0, 1]}}
    assert [(w["id"], w["disco"]) for w in wanted(cont, release, entries)] == [("a", 1)]   # b ya no dura 500
    per_track = {"youtube": {"tipo": "por_tema", "archivo_por_pista": [2, 3, 4]}}
    assert [(w["id"], w["disco"], w["pista"]) for w in wanted(per_track, release, entries)] == [
        ("c", 1, 1), ("d", 1, 2), ("e", 2, 1)]
