"""
PURPOSE: Tests de tools/dj_sets/vistas_djsets.py (A0 de la línea A): la ventana de YouTube, los segmentos válidos, la
         población por shard, la recodificación igual a la de E17, un shard de punta a punta con un remoto local y un
         modelo falso (también cortado y retomado desde el parcial), el plan de trabajos, juntar y verificar.
CHANGELOG:
  - 2026-11-13: Creación.
"""
import csv
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.remote_storage import import_gpuq_storage  # noqa: E402
from tools.dj_sets import vistas_djsets as vd  # noqa: E402

HAY_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture(autouse=True)
def _sin_drive(tmp_path, monkeypatch):
    """Ningún test de este archivo llega al Drive: open_remote() da una carpeta temporal."""
    monkeypatch.setenv("TRAKTOR_REMOTE", str(tmp_path / "_guardia"))


def test_ventana_es_la_del_preview():
    assert vd.ventana(377.0) == pytest.approx((150.8, 120.0))
    assert vd.ventana(150.0) == pytest.approx((60.0, 90.0))
    assert vd.ventana(89.0) == pytest.approx((35.6, 53.4))


def test_segmentos_validos():
    casos = {120: 4, 119.9: 4, 104: 3, 90: 3, 50: 2, 44.9: 1, 30: 1, 10: 1, 0: 1}
    assert {s: vd.n_seg_validos(s) for s in casos} == casos


def test_control_es_un_decimo_y_estable():
    ids = [f"yt_{i:05d}" for i in range(5000)]
    frac = np.mean([vd.es_control(r) for r in ids])
    assert 0.08 < frac < 0.12
    assert [vd.es_control(r) for r in ids[:50]] == [vd.es_control(r) for r in ids[:50]]


def _t(track_id, rec, shard, member, fuente="beatport", audio="1"):
    return {"track_id": track_id, "recording_id": rec, "audio_shard": shard, "audio_member": member,
            "audio_source": fuente, "has_audio": audio}


def test_poblacion_una_entrada_por_grabacion():
    tracks = [_t("a", "bp_1", "audio/beatport/beatport_000.tar", "1.mp3"),
              _t("b", "bp_1", "audio/beatport/beatport_000.tar", "1.mp3"),      # mismo audio, otro tema
              _t("c", "", "", "", audio="0"),
              _t("d", "yt_x", "audio/youtube/youtube_000.tar", "x.m4a", "youtube")]
    pob = vd.poblacion(tracks)
    assert [m["recording_id"] for m in pob["audio/beatport/beatport_000.tar"]] == ["bp_1"]
    assert pob["audio/youtube/youtube_000.tar"][0]["fuente"] == "youtube"


@pytest.mark.skipif(not HAY_FFMPEG, reason="sin ffmpeg")
def test_recodificacion_igual_a_la_de_e17():
    from src.v4.evaluation import codec_clap as cc
    sr = 44100
    t = np.arange(8 * sr) / sr
    x = np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.2 * np.sin(2 * np.pi * 997 * t)], axis=1).astype(np.float32)
    assert np.array_equal(vd.mp3_ida_y_vuelta(x, sr, 96), cc.mp3_ida_y_vuelta(x, sr, 96))
    ruido = np.random.default_rng(0).normal(size=8 * sr).astype(np.float32)
    atrasado = np.r_[np.zeros(300, np.float32), ruido[:-300]]
    assert vd.retardo(ruido, atrasado, sr) == cc.retardo(ruido, atrasado, sr) == 300


def _wav(segundos: float, amp: float, sr: int = 8000, canales: int = 1) -> bytes:
    import soundfile as sf
    b = io.BytesIO()
    s = (amp * np.sin(np.arange(int(segundos * sr)) / 5.0)).astype(np.float32)
    sf.write(b, np.stack([s] * canales, axis=1) if canales > 1 else s, sr, format="WAV")
    return b.getvalue()


@pytest.mark.skipif(not HAY_FFMPEG, reason="sin ffmpeg")
def test_audio_youtube_toma_la_ventana(tmp_path):
    p = tmp_path / "tema.m4a"                      # contenido WAV: ffmpeg lo reconoce por el contenido
    p.write_bytes(_wav(200, 0.3, sr=44100, canales=2))
    a = vd.audio_youtube(p, control=True)
    assert a["duracion_s"] == pytest.approx(200.0, abs=0.01)
    assert a["comienzo_s"] == pytest.approx(80.0, abs=0.01)
    assert a["segundos"] == pytest.approx(120.0, abs=0.01)
    assert len(a["x"]) == len(a["orig"]) == 120 * 44100 and np.isfinite(a["x"]).all()


class _Falso:
    info = {"modelo": "falso"}

    def embed(self, items):
        n = len(items)
        rms = np.asarray([float(np.sqrt(np.mean(x ** 2))) for x, _ in items])
        return {k: np.ones((n, vd.N_SEG, d), np.float32) * rms[:, None, None] for k, d in vd.DIMS}


class _SeCorta(_Falso):
    def __init__(self, n):
        self.n = n

    def embed(self, items):
        if self.n <= 0:
            raise RuntimeError("máquina perdida")
        self.n -= len(items)
        return super().embed(items)


def _tar(path: Path, miembros):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for nombre, datos in miembros:
            info = tarfile.TarInfo(nombre)
            info.size = len(datos)
            tar.addfile(info, io.BytesIO(datos))


def _leer_yt_falso(path, control):
    """Como audio_youtube, sin ffmpeg: el WAV entero es la ventana."""
    a = vd.audio_beatport(path)
    return dict(a, comienzo_s=1.0, duracion_s=10.0, orig=(a["x"] * 0.5 if control else None))


def test_un_shard_de_youtube_de_punta_a_punta(tmp_path):
    remote = import_gpuq_storage().LocalRemote(tmp_path / "remoto")
    ids = [f"yt_{i}" for i in range(40)]
    ctrl = [r for r in ids if vd.es_control(r)]
    assert ctrl, "la muestra tiene que traer algún control"
    _tar(tmp_path / "remoto" / vd.DATASET / "audio/youtube/youtube_000.tar",
         [(f"{r}.m4a", _wav(2, 0.01 * (i + 1))) for i, r in enumerate(ids[:-1])])     # el último no está
    miembros = [{"member": f"{r}.m4a", "recording_id": r, "fuente": "youtube"} for r in ids]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    meta = vd.procesar_shard(remote, "audio/youtube/youtube_000.tar", miembros, _Falso(), scratch, lote=4,
                             destino="d", leer_yt=_leer_yt_falso)
    assert meta["hechos"] == 39 and [f["recording_id"] for f in meta["fallidos"]] == ["yt_39"]
    z = np.load(tmp_path / "remoto" / "d" / "youtube" / "youtube_000.npz")
    assert list(z["recording_id"]) == ids[:-1]
    assert z["clap"].shape == (39, vd.N_SEG, 512) and np.all(np.diff(z["clap"][:, 0, 0]) > 0)
    assert sorted(z["control_id"]) == sorted(r for r in ctrl if r != "yt_39")
    i = list(z["recording_id"]).index(z["control_id"][0])
    assert z["clap_orig"][0, 0, 0] == pytest.approx(0.5 * z["clap"][i, 0, 0])      # el control es su tema
    assert np.allclose(z["comienzo_s"], 1.0) and json.loads(str(z["meta"]))["fuente"] == "youtube"
    assert not list(scratch.glob("*.tar")) and not list(scratch.glob("*.npz"))


def test_un_shard_cortado_sigue_desde_el_parcial(tmp_path):
    remote = import_gpuq_storage().LocalRemote(tmp_path / "remoto")
    ids = [f"bp_{20 + i}" for i in range(6)]
    _tar(tmp_path / "remoto" / vd.DATASET / "audio/beatport/beatport_001.tar",
         [(f"{r}.wav", _wav(2, 0.1 * (i + 1))) for i, r in enumerate(ids)])
    miembros = [{"member": f"{r}.wav", "recording_id": r, "fuente": "beatport"} for r in ids]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(RuntimeError):
        vd.procesar_shard(remote, "audio/beatport/beatport_001.tar", miembros, _SeCorta(2), scratch, lote=1, cada=1,
                          destino="d")
    parcial = tmp_path / "remoto" / "d" / "beatport" / "beatport_001.parcial.npz"
    assert parcial.exists() and list(np.load(parcial)["recording_id"]) == ids[:2]
    meta = vd.procesar_shard(remote, "audio/beatport/beatport_001.tar", miembros, _Falso(), scratch, lote=1, cada=1,
                             destino="d")
    z = np.load(tmp_path / "remoto" / "d" / "beatport" / "beatport_001.npz")
    assert list(z["recording_id"]) == ids and meta["hechos"] == 6 and not meta["fallidos"]
    assert np.all(np.diff(z["clap"][:, 0, 0]) > 0) and np.allclose(z["segundos"], 2.0)
    assert not parcial.exists()


def test_pendientes_y_trabajos():
    shards = [{"shard": "audio/beatport/beatport_000.tar", "source": "beatport", "origin": "beatport_previews/lote_01/tanda_00000.tar"},
              {"shard": "audio/beatport/beatport_001.tar", "source": "beatport", "origin": "beatport_previews/lote_05/tanda_00000.tar"},
              {"shard": "audio/beatport/beatport_002.tar", "source": "beatport", "origin": "beatport_previews/lote_05/tanda_00001.tar"},
              {"shard": "audio/youtube/youtube_000.tar", "source": "youtube", "origin": "youtube_temas/yt_01/tanda_00000.tar"}]
    pend = vd.pendientes(shards, {"lote_01/tanda_00000.npz"}, {"beatport": {"beatport_002.npz"}})
    assert pend == ["audio/beatport/beatport_001.tar", "audio/youtube/youtube_000.tar"]
    pob = {f"audio/beatport/beatport_00{i}.tar": [{}] * 1000 for i in range(4)}
    trabajos = vd.agrupar([f"audio/beatport/beatport_00{i}.tar" for i in range(4)], pob, {"beatport": 1.5}, minutos=55)
    assert [len(t) for t in trabajos] == [2, 2]             # 25 min por shard: dos por trabajo


def _escribir_csv(path: Path, filas):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def _salida_a0(path: Path, ids, valor, fuente, ctrl=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    meta = {"shard": f"audio/{fuente}/{path.stem}.tar", "fuente": fuente, "fallidos": [], "gpuq_job": "prueba"}
    np.savez(path, recording_id=np.asarray(ids, dtype=str), segundos=np.full(n, 120.0, np.float32),
             comienzo_s=np.full(n, 0.4 * 300, np.float32), duracion_s=np.full(n, 300.0, np.float32),
             control_id=np.asarray(list(ctrl), dtype=str), meta=np.asarray(json.dumps(meta)),
             **{k: np.full((n, vd.N_SEG, d), valor, np.float32) for k, d in vd.DIMS},
             **{f"{k}_orig": np.full((len(ctrl), vd.N_SEG, d), valor + 1, np.float32) for k, d in vd.DIMS})


def test_juntar_alinea_por_track_id(tmp_path):
    raiz = tmp_path / "remoto"
    remote = import_gpuq_storage().LocalRemote(raiz)
    tracks = [{"track_id": "t1", "recording_id": "bp_1", "has_audio": "1", "audio_source": "beatport"},
              {"track_id": "t2", "recording_id": "", "has_audio": "0", "audio_source": ""},
              {"track_id": "t3", "recording_id": "yt_a", "has_audio": "1", "audio_source": "youtube"},
              {"track_id": "t4", "recording_id": "bp_1", "has_audio": "1", "audio_source": "beatport"},
              {"track_id": "t5", "recording_id": "bp_2", "has_audio": "1", "audio_source": "beatport"}]
    _escribir_csv(raiz / vd.TRACKS, tracks)
    _escribir_csv(raiz / vd.SHARDS, [{"shard": "audio/beatport/beatport_000.tar", "source": "beatport",
                                      "origin": "beatport_previews/lote_01/tanda_00000.tar"}])
    v7 = raiz / vd.V7 / "lote_01" / "tanda_00000.npz"
    v7.parent.mkdir(parents=True)
    np.savez(v7, bp_id=np.asarray(["1"]), segundos=np.asarray([90.0], np.float32),
             meta=np.asarray(json.dumps({"gpuq_job": "traktor-v7-2"})),
             **{k: np.full((1, vd.N_SEG, d), 3.0, np.float32) for k, d in vd.DIMS})
    _salida_a0(raiz / "d" / "youtube" / "youtube_000.npz", ["yt_a"], 5.0, "youtube", ctrl=["yt_a"])
    rep = vd.juntar(remote, "d", "sal", tmp_path / "cache")
    assert rep["con_vistas"] == 3 and rep["sin_vistas"] == 1           # bp_2 no está en ninguna salida
    assert rep["umbral_1_cobertura"] is False and rep["umbral_2_finitos"] and rep["umbral_3_alineacion"]
    assert rep["umbral_6_ventana_youtube"] is True
    ind = list(csv.DictReader(open(raiz / "sal" / "indice.csv", encoding="utf-8")))
    assert [r["track_id"] for r in ind] == ["t1", "t3", "t4", "t5"]
    assert [r["n_seg_validos"] for r in ind] == ["3", "4", "3", "0"]
    clap = np.load(raiz / "sal" / "clap.npy")
    assert clap.shape == (4, vd.N_SEG, 512) and clap[0, 0, 0] == clap[2, 0, 0] == 3.0 and clap[1, 0, 0] == 5.0
    ctrl = np.load(raiz / "sal" / "youtube_control.npz")
    assert list(ctrl["recording_id"]) == ["yt_a"] and ctrl["clap_orig"][0, 0, 0] == 6.0
    leeme = json.loads((raiz / "sal" / "LEEME.json").read_text(encoding="utf-8"))
    assert set(leeme["md5"]) == {"indice.csv", "clap.npy", "l7.npy", "styles.npy", "youtube_control.npz"}


def test_verificar_v7(tmp_path):
    rng = np.random.default_rng(0)
    V = {k: rng.normal(size=(3, vd.N_SEG, d)).astype(np.float32) for k, d in vd.DIMS}
    v7 = tmp_path / "v7.npz"
    np.savez(v7, bp_id=np.asarray(["7", "8", "9"]), segundos=np.full(3, 120.0, np.float32), **V)
    a0 = tmp_path / "a0.npz"
    np.savez(a0, recording_id=np.asarray(["bp_9", "bp_7"]), segundos=np.full(2, 120.0, np.float32),
             **{k: V[k][[2, 0]] for k in V})
    r = vd.verificar_v7(a0, v7)
    assert r["comunes"] == 2 and r["pasa"] and r["clap"]["min"] == pytest.approx(1.0)
    np.savez(a0, recording_id=np.asarray(["bp_9", "bp_7"]), segundos=np.full(2, 120.0, np.float32),
             **{k: V[k][[0, 2]] for k in V})                            # filas cambiadas
    assert not vd.verificar_v7(a0, v7)["pasa"]
