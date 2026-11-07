"""
PURPOSE: Tests de src/v4/harmonic_tiv (rama harmonic-tiv): TIV igual a TIV.lib, transposición,
         front-end con señales sintéticas y métricas proxy.
CHANGELOG:
  - 2026-11-06: Creación.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.v4.harmonic_tiv import evaluate as ev  # noqa: E402
from src.v4.harmonic_tiv import frontend as fe  # noqa: E402
from src.v4.harmonic_tiv import tiv as tv  # noqa: E402

# Valores de referencia calculados con tivlib/tiv.py de github.com/gbibbo/harmonic_mix (TIV.from_pcp,
# small_scale_compatibility, get_max_compatibility) el 2026-11-06.
A_MINOR = np.array([1.0, 0, 0.2, 0, 0.8, 0.3, 0, 0.1, 0, 1.0, 0, 0.2])     # índice 0 = C
E_MINOR = np.roll(A_MINOR, 7)


def _tivlib_reference(a, b):
    """Fórmulas de TIV.lib escritas de nuevo, tal cual (sin vectorizar), para comparar."""
    w = np.array([3, 8, 11.5, 15, 14.5, 7.5])

    def from_pcp(p):
        f = np.fft.rfft(p, n=12)
        return f[1:7] / f[0] * w

    ta, tb = from_pcp(a), from_pcp(b)
    rel = np.linalg.norm(ta - tb) / (np.linalg.norm(w) * 2)
    dis = 1 - np.linalg.norm((ta + tb) / 2) / np.linalg.norm(w)
    return ta, rel * dis


def test_tiv_matches_tivlib_formulas():
    rng = np.random.default_rng(3)
    for _ in range(50):
        a, b = rng.random(12), rng.random(12)
        ta, ref = _tivlib_reference(a, b)
        assert np.allclose(tv.tiv_from_chroma(a), ta)
        assert tv.hc_tivlib(tv.tiv_from_chroma(a), tv.tiv_from_chroma(b)) == pytest.approx(ref, abs=1e-12)


def test_transpose_equals_rolling_the_chroma():
    t = tv.tiv_from_chroma(A_MINOR)
    for p in range(-6, 7):
        assert np.allclose(tv.transpose(t, p), tv.tiv_from_chroma(np.roll(A_MINOR, p)))
    many = tv.transpose(t, np.arange(12))
    assert many.shape == (12, 6)


def test_k0_and_k6_are_real_not_a_bug():
    # La tesis (6.2) lo reportó como bug de TIV.lib; es la DFT de 12 valores reales.
    f = np.fft.rfft(np.random.default_rng(0).random(12))
    assert f[0].imag == 0 and abs(f[6].imag) < 1e-12
    assert tv.tiv_from_chroma(A_MINOR)[5].imag == pytest.approx(0.0, abs=1e-12)


def test_best_shift_recovers_transposition_and_prefers_small_shift():
    a, e = tv.tiv_from_chroma(A_MINOR), tv.tiv_from_chroma(E_MINOR)
    s, hc = tv.best_shift(a, e)
    assert (s - (-7)) % 12 == 0 and s == 5 and hc == pytest.approx(0.0, abs=1e-12)
    s2, _ = tv.best_shift(a, e, max_shift=2)
    assert abs(s2) <= 2
    assert tv.tivlib_best_shift(a, e)[0] == 5


def test_hc_matrix_shape_and_identity():
    T = np.stack([tv.tiv_from_chroma(A_MINOR), tv.tiv_from_chroma(E_MINOR)])
    m = tv.hc_matrix(T, T, shifts=[0, 5])
    assert m.shape == (2, 2, 2)
    assert m[0, 0, 0] == pytest.approx(0.0) and m[0, 1, 1] == pytest.approx(0.0, abs=1e-12)


def test_flat_chroma_has_zero_tiv_and_strength():
    assert np.allclose(tv.tiv_from_chroma(np.ones(12)), 0)
    assert tv.tonal_strength(tv.tiv_from_chroma(np.ones(12))) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(tv.tiv_from_chroma(np.zeros(12)), 0)


def _tone(midi, detune, sr, seconds=8.0):
    t = np.arange(int(sr * seconds)) / sr
    f0 = 440.0 * 2 ** ((midi - 69 + detune) / 12)
    return sum(0.6 ** (h - 1) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 6))


@pytest.mark.parametrize("detune", [0.0, 0.25, -0.3])
def test_frontend_tuning_and_chroma_on_synthetic_chord(detune):
    cfg = fe.FrontendConfig()
    x = (_tone(57, detune, cfg.sr) + _tone(60, detune, cfg.sr) + _tone(64, detune, cfg.sr)).astype(np.float32)
    out = fe.analyze_signal(x / np.abs(x).max(), cfg, variants=("white_lin",))
    assert out["tuning"] == pytest.approx(detune, abs=0.05)
    ch = fe.semitone_to_chroma(out["white_lin"].astype(float), fe.chroma_profile("treble")).sum(0)
    assert set(np.argsort(ch)[::-1][:3]) == {9, 0, 4}          # A, C, E
    assert out["white_lin"].shape[0] == fe.N_SEMI and np.all(np.isfinite(out["white_lin"].astype(float)))


def test_segmented_hpss_equals_whole_signal():
    # Por tramos (memoria acotada) tiene que dar lo mismo que de una vez: tramos alineados al salto.
    cfg = fe.FrontendConfig()
    rng = np.random.default_rng(5)
    x = (_tone(57, 0.0, cfg.sr, 14.0) + 0.3 * rng.standard_normal(int(cfg.sr * 14.0))).astype(np.float32)
    whole = fe._hpss_block(x, cfg)
    seg = fe.hpss_harmonic(x, cfg, segment_s=5.0, pad_s=1.0)
    assert np.allclose(whole, seg, atol=1e-5)


def _kick_track(bpm, first_beat_s, seconds, sr, hat=True):
    """Bombos (seno de 55 Hz con caída) en cada beat y hi-hats (ruido) a contratiempo."""
    rng = np.random.default_rng(0)
    x = np.zeros(int(seconds * sr), dtype=np.float32)
    beat = 60.0 / bpm
    kick = (np.sin(2 * np.pi * 55 * np.arange(int(0.15 * sr)) / sr) * np.exp(-np.arange(int(0.15 * sr)) / (0.04 * sr)))
    hh = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.005 * sr)) * 0.8
    t = first_beat_s
    while t < seconds - 0.2:
        i = int(t * sr)
        x[i:i + len(kick)] += kick
        if hat:
            j = int((t + beat / 2) * sr)
            x[j:j + len(hh)] += hh
        t += beat
    return x


def test_beat_phase_follows_the_kick_not_the_offbeat_hat():
    from src.v4.harmonic_tiv import render as rd
    x = _kick_track(126.0, 0.137, 20.0, rd.SR)
    phase, conf = rd.beat_phase_conf(x, 126.0)
    assert abs(phase - 0.137) < 0.012 and conf > 2.0


def _groove(bpm, start_beat, n_beats, sr, clap_beats=(1, 3), seed=0, delay_s=0.0):
    """Bombo en cada beat, bajo fuerte a contratiempo (60 Hz), hi-hat a contratiempo y palma en los beats
    `clap_beats` del compás, a partir del beat `start_beat` del patrón; todo corrido `delay_s`."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    x = np.zeros(int((n_beats + 1) * beat * sr), dtype=np.float32)
    n_k = int(0.15 * sr)
    kick = np.sin(2 * np.pi * 55 * np.arange(n_k) / sr) * np.exp(-np.arange(n_k) / (0.03 * sr))
    n_b = int(0.2 * sr)
    bass = 1.6 * np.sin(2 * np.pi * 62 * np.arange(n_b) / sr) * np.exp(-np.arange(n_b) / (0.08 * sr))
    for b in range(n_beats):
        t0 = b * beat + delay_s
        i, j = int(t0 * sr), int((t0 + beat / 2) * sr)
        x[i:i + n_k] += kick[: len(x[i:i + n_k])]
        x[j:j + n_b] += bass[: len(x[j:j + n_b])]
        hh = rng.standard_normal(int(0.02 * sr)) * np.exp(-np.arange(int(0.02 * sr)) / (0.004 * sr)) * 0.3
        x[j:j + len(hh)] += hh[: len(x[j:j + len(hh)])]
        if (b + start_beat) % 4 in clap_beats:
            cl = rng.standard_normal(int(0.08 * sr)) * np.exp(-np.arange(int(0.08 * sr)) / (0.02 * sr)) * 0.6
            x[i:i + len(cl)] += cl[: len(x[i:i + len(cl)])]
    return x


def test_rhythm_lag_recovers_a_quarter_beat_with_offbeat_bass():
    from src.v4.harmonic_tiv import render as rd
    bpm, beat = 125.0, 60.0 / 125.0
    ref = _groove(bpm, 0, 32, rd.SR, seed=1)
    x = _groove(bpm, 0, 32, rd.SR, seed=2, delay_s=beat / 4)
    lag, conf, _ = rd.rhythm_lag(ref, x, beat)
    assert abs(lag - beat / 4) < 0.01 and conf > rd.ALIGN_CONF_MIN
    lag0, _, _ = rd.rhythm_lag(ref, _groove(bpm, 0, 32, rd.SR, seed=3), beat)
    assert abs(lag0) < 0.01


def test_vote_lag_prefers_the_layers_that_agree_over_two_strong_hat_bands():
    from src.v4.harmonic_tiv import render as rd
    n = rd.BINS_PER_BEAT
    xc = np.zeros((5, n))
    for b in range(3):                       # bombo, bajo y medios: pico débil en 0
        xc[b, 0] = 0.3
    for b in (3, 4):                         # dos bandas de hi-hat: pico fuerte a medio beat
        xc[b, n // 2] = 0.9
    assert int(np.argmax(xc.sum(axis=0))) == n // 2        # la suma sola elegiría medio beat
    assert rd.vote_lag(xc, tol_bins=4) == 0


def test_bar_offset_puts_the_claps_together():
    from src.v4.harmonic_tiv import render as rd
    bpm, beat = 125.0, 60.0 / 125.0
    ref = _groove(bpm, 0, 32, rd.SR, seed=1)                       # palmas en 2 y 4 (índices 1, 3)
    same, _ = rd.bar_offset(ref, _groove(bpm, 0, 32, rd.SR, seed=2), beat)
    shifted, sc = rd.bar_offset(ref, _groove(bpm, 1, 32, rd.SR, seed=2), beat)   # arranca en el beat 2
    assert same == 0 and shifted in (1, 3) and sc[shifted] > sc[0]


def test_render_items_balances_the_order_of_the_two_mixes(tmp_path):
    lt = _lt()
    items = [{"target": f"t{i}", "pick_a": f"a{i}", "pick_b": f"b{i}", "shift_a": 0, "shift_b": 0,
              "auto_tie": i % 7 == 0, "mode": "central"} for i in range(36)]
    key_items, page = {}, []
    lt.render_items(None, items, tmp_path, np.random.default_rng(5), "x", "y", "c1", key_items, page, render=False)
    firsts = [key_items[p["id"]]["clip_of_a"] for p in page]
    assert len(firsts) == 30 and firsts.count(1) == 15


def test_mix_transition_length_and_finite():
    from src.v4.harmonic_tiv import render as rd
    t = rd.Template()
    beat = 60.0 / 125.0
    tgt = _kick_track(125.0, 0.0, (t.lead_beats + t.overlap_beats) * beat, rd.SR)
    cand = _kick_track(125.0, 0.0, (t.overlap_beats + t.tail_beats) * beat, rd.SR)
    out = rd.mix_transition(tgt, cand, beat, t)
    expected = sum(int(round(n * beat * rd.SR)) for n in (t.lead_beats, t.overlap_beats, t.tail_beats))
    assert len(out) == expected and np.all(np.isfinite(out))


def _lt():
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "harmonic_tiv"))
    import listening_test
    return listening_test


def test_camelot_control_replicates_the_live_recommender_filter():
    lt = _lt()
    keys = {"t": "8A", "same": "8A", "rel": "8B", "far": "2A", "nokey": "?", "shift1": "3A"}
    passing, shifts = lt.camelot_pass(keys, "t", ["same", "rel", "far", "nokey", "shift1"])
    assert set(passing) == {"same", "rel", "shift1"}          # 3A a +1 semitono = 10A: no; a -1 = 8A: sí
    assert shifts["same"] == 0 and abs(shifts["shift1"]) == 1
    keys["t"] = "?"                                            # principal sin tonalidad: no filtra
    passing, _ = lt.camelot_pass(keys, "t", ["same", "far", "nokey"])
    assert set(passing) == {"same", "far", "nokey"}


def test_build_items_never_repeats_a_track_and_marks_auto_ties():
    lt = _lt()
    rng = np.random.default_rng(0)
    ids = [f"x{i}" for i in range(200)]

    def pool_fn(t, used):
        return [x for x in ids if x not in used]

    def arm_first(t, pool):
        return pool[0], 0

    def arm_second(t, pool):
        return pool[1], 0
    used = set()
    items, _ = lt.build_items(ids[:20], 5, pool_fn, arm_first, arm_first, used, rng, only_disagreements=False)
    assert all(it["auto_tie"] for it in items)
    items2, _ = lt.build_items(ids[20:60], 5, pool_fn, arm_first, arm_second, used, rng, only_disagreements=True)
    seen = [u for it in items + items2 for u in (it["target"], it["pick_a"], it["pick_b"])]
    assert len(seen) - len(items) == len(set(seen))            # en un empate pick_a == pick_b: se cuenta una vez


def test_score_u_with_ties_sequential_verdict_and_repeats():
    lt = _lt()
    items = {}
    for k in range(60):
        items[f"c1{k:03d}"] = {"auto_tie": k < 10, "clip_of_a": 1, "comparison": "c1", "repeat_of": None}
        items[f"c2{k:03d}"] = {"auto_tie": False, "clip_of_a": 2, "comparison": "c2", "repeat_of": None}
    for r in range(20):
        items[f"r{r:03d}"] = {"auto_tie": False, "clip_of_a": 2, "comparison": "c1", "repeat_of": f"c1{10 + r:03d}"}
    ans = {}
    for k in range(10, 60):                                    # c1: 40 victorias del TIV, 10 derrotas
        ans[f"c1{k:03d}"] = "1" if k < 50 else "2"
    for k in range(60):                                        # c2: 30 victorias, 30 empates del oyente
        ans[f"c2{k:03d}"] = "2" if k < 30 else "0"
    for r in range(20):                                        # repetidos con el orden invertido: coinciden 18
        ans[f"r{r:03d}"] = "2" if r < 18 else "1"
    res = lt.score({"phase": "confirm", "items": items}, ans)
    assert res["c1"]["U"] == pytest.approx((40 + 0.5 * 10) / 60, abs=1e-3) and res["c1"]["aprueba"]
    assert res["c2"]["U"] == pytest.approx((30 + 15) / 60, abs=1e-3) and res["c2"]["aprueba"]
    assert res["repetidos"]["acuerdos"] == 18 and res["veredicto"].startswith("SE INTEGRA")
    for it in items.values():
        it.update(arm_a="tiv", arm_b="control")
    few = {f"c1{k:03d}": ["1"] for k in range(10, 20)}         # 10 de 110 renderizados: los estímulos valen
    res = lt.score({"phase": "confirm", "items": items}, ans, few)
    assert res["desfasadas"]["items"] == 10 and res["desfasadas"]["mezclas_por_brazo"] == {"tiv": 10}
    assert res["veredicto"].startswith("SE INTEGRA") and res["c1_sin_desfasadas"]["N"] == 50
    many = {f"c2{k:03d}": ["2"] for k in range(30)}            # 30 de 110 (> 20 %): no concluyente
    assert "desfasados" in lt.score({"phase": "confirm", "items": items}, ans, many)["veredicto"]
    for r in range(20):                                        # consistencia insuficiente -> no concluyente
        ans[f"r{r:03d}"] = "1"
    assert lt.score({"phase": "confirm", "items": items}, ans)["veredicto"].startswith("NO CONCLUYENTE")


def test_fifths_distance_and_shift_target():
    assert ev.fifths_distance("8A", "8B") == 0
    assert ev.fifths_distance("8A", "9A") == 1
    assert ev.fifths_distance("1A", "7A") == 6
    assert ev.key_shift_target("8A", "8A") == 0
    assert ev.key_shift_target("8A", "9A") in (5, -7 + 12)       # bajar una quinta = subir 5
    assert ev.key_shift_target("8A", "1A") % 12 == (7 * 7) % 12


def test_selfsup_continuation_finds_the_next_fragment_and_shift_zero():
    rng = np.random.default_rng(2)
    frags = {}
    for t in range(30):
        base = rng.random(12) ** 3                       # perfil de alturas propio de cada tema
        frags[f"t{t}"] = np.array([tv.tiv_from_chroma(base + 0.02 * rng.random(12)) for _ in range(6)])
    out = ev.selfsup_continuation(frags, tv.hc_tivlib, n_neg=20, pairs_per_track=2)
    assert out["tiv"]["mrr"] > 0.9 and out["tiv"]["shift0_full"] > 0.9
    assert out["tiv"]["mrr"] > 3 * out["tiv"]["chance_mrr"]


def test_selfsup_skips_silent_fragments_and_intro_outro_runs():
    rng = np.random.default_rng(4)
    frags = {f"t{t}": np.array([tv.tiv_from_chroma(rng.random(12)) for _ in range(4)]) for t in range(10)}
    frags["t0"][1] = np.nan                                # fragmento en silencio
    out = ev.selfsup_continuation(frags, tv.hc_tivlib, n_neg=5, pairs_per_track=3)
    assert out["tiv"]["n_queries"] > 0
    first = {u: f[0] for u, f in frags.items()}
    last = {u: f[-1] for u, f in frags.items()}
    io = ev.selfsup_intro_outro(first, last, tv.hc_tivlib, n_neg=5, key_of={u: "8A" for u in frags})
    assert io["camelot"]["auc"] == pytest.approx(0.5)      # misma tonalidad para todos: Camelot empata


def test_timed_sections():
    from src.v4.harmonic_tiv import features as ft
    m = ft.section_mask(1000, "last120s")
    assert m.sum() == 240 and m[-1] and not m[0]
    m = ft.section_mask(100, "first120s")
    assert m.all()


def test_auc_and_hubness():
    assert ev.auc(np.array([3, 4]), np.array([1, 2])) == 1.0
    assert ev.auc(np.array([1, 2]), np.array([1, 2])) == 0.5
    s = np.zeros((5, 5))
    s[:, 0] = 1.0                                               # el tema 0 aparece arriba para todos
    h = ev.hubness(s, k=1)
    assert h["max_occ"] == 4 and h["never"] > 0.5


def test_key_pair_metrics_perfect_score():
    keys = ["8A", "8A", "9A", "9A", "2A", "2A"]
    n = len(keys)
    score = np.array([[-ev.fifths_distance(keys[i], keys[j]) for j in range(n)] for i in range(n)], float)
    m = ev.key_pair_metrics(score, keys)
    assert m["auc_near_far"] == 1.0 and m["mrr_same"] == 1.0
