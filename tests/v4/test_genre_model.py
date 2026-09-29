"""
PURPOSE: Tests de src/v4/pipeline/genre_model.py con datos sintéticos: gate MAEST (cociente no
         electrónico / electrónico y mapeo a Beatport), selección de etiquetas confirmadas alineadas
         por track_uids, validación cruzada agrupada y métricas.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.pipeline.genre_model import (  # noqa: E402
    cross_validate, gate_genre, gate_scores, labeled_frame, metrics, primary_artist,
)

LABELS = ["Electronic---House", "Electronic---Techno", "Hip Hop---Boom Bap", "Rock---Indie Rock"]


def test_gate_scores_and_mapping():
    styles = np.array([[0.8, 0.3, 0.05, 0.01],    # house
                       [0.1, 0.05, 0.6, 0.02]])   # rap
    ratio, parent = gate_scores(styles, LABELS)
    assert ratio[0] < 1 < ratio[1]
    assert parent == ["Hip Hop", "Hip Hop"]
    assert gate_genre("Hip Hop") == "Hip-Hop"
    assert gate_genre("Jazz") == "Jazz (Discogs)"


def test_primary_artist():
    assert primary_artist("Groove Armada; Myd") == "groove armada"
    assert primary_artist("Peer Kusiv, Lenny") == "peer kusiv"
    assert primary_artist("") == ""


def test_labeled_frame_confirmed_and_alignment():
    bp = pd.DataFrame({
        "track_uid": ["u1", "u2", "u3", "u4", "u5"],
        "level": ["A", "B", "B", "C", "B"],
        "confirmed": [True, True, False, False, True],
        "bp_genre": ["House", "Techno", "House", None, "House"],
        "bp_artists": ["X", "Y", "Z", None, "W"],
        "cur_artist": ["x", "y", "z", "q", "w"],
    })
    uids = ["u5", "u1", "u2", "u3"]   # u4 sin representaciones
    lab = labeled_frame(bp, uids, confirmed_only=True)
    assert list(lab.index) == ["u1", "u2", "u5"]
    assert dict(zip(lab.index, lab["row"])) == {"u1": 1, "u2": 2, "u5": 0}
    assert len(labeled_frame(bp, uids, confirmed_only=False)) == 4


def test_cross_validate_separable_classes():
    rng = np.random.default_rng(0)
    y = np.array(["House"] * 40 + ["Techno"] * 40)
    X = rng.normal(size=(80, 8)) + np.where(y == "House", 3.0, -3.0)[:, None]
    groups = np.array([f"a{i}" for i in range(80)])
    classes, P = cross_validate(X, y, groups)
    m = metrics(y, classes, P)
    assert m["accuracy"] > 0.95 and m["n_classes"] == 2 and m["majority_baseline"] == 0.5
    assert m["by_confidence"][0]["coverage"] == 1.0
    assert np.allclose(P.sum(1), 1.0)


def test_load_features_aligns_by_track_uid(tmp_path):
    import json as _json
    from src.v4.pipeline.genre_model import CLF_REPS, STYLES_REP, load_features
    rows = {"a": 1.0, "b": 2.0, "c": 3.0}
    for rep, order in zip(CLF_REPS + (STYLES_REP,), (["a", "b", "c"], ["c", "a", "b"], ["b", "c", "a"])):
        d = tmp_path / "representations" / rep
        d.mkdir(parents=True)
        (d / "track_uids.json").write_text(_json.dumps(order), encoding="utf-8")
        np.save(d / "embeddings.npy", np.array([[rows[u]] for u in order], dtype=np.float32))
    uids, X, styles = load_features(tmp_path)
    assert uids == ["a", "b", "c"]
    assert X.tolist() == [[1, 1], [2, 2], [3, 3]] and styles.ravel().tolist() == [1, 2, 3]
