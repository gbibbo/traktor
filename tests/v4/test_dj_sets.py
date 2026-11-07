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
