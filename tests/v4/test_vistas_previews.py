"""
PURPOSE: Tests de tools/dj_sets/vistas_previews.py (V7, plan sección 12.9): la ventana V4 igual a la de E15, los temas
         de un manifiesto, y una tanda de punta a punta con un remoto local y un modelo falso (sin red ni modelos).
CHANGELOG:
  - 2026-11-10: Creación.
"""
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.remote_storage import import_gpuq_storage  # noqa: E402
from src.v4.evaluation import dominio_previews as dp  # noqa: E402
from tools.dj_sets import vistas_previews as vp  # noqa: E402


def test_cortar_es_la_ventana_v4_de_e15():
    x = np.random.default_rng(0).normal(size=100).astype(np.float32)
    for a, b in zip(vp.cortar(x, sr=1), dp.cortar(x, sr=1)):
        assert np.array_equal(a, b)


def test_temas_ok_sin_repetidos_ni_fallidos():
    man = {"temas": [{"estado": "ok", "beatport_id": 1, "archivo": "1.mp3"}, {"estado": "ok", "beatport_id": 1, "archivo": "1.mp3"},
                     {"estado": "sin_id", "beatport_id": "", "archivo": ""}, {"estado": "ok", "beatport_id": 2, "archivo": "2.mp3"}]}
    assert [t["beatport_id"] for t in vp.temas_ok(man)] == [1, 2]


class _Falso:
    info = {"modelo": "falso"}

    def embed(self, previews):
        n = len(previews)
        rms = np.asarray([float(np.sqrt(np.mean(x ** 2))) for x, _ in previews])
        return {k: np.ones((n, vp.N_SEG, d), np.float32) * rms[:, None, None] for k, d in (("clap", 512), ("l7", 768), ("styles", 400))}


def _wav(segundos: float, amp: float, sr: int = 8000) -> bytes:
    import soundfile as sf
    b = io.BytesIO()
    sf.write(b, (amp * np.sin(np.arange(int(segundos * sr)) / 5.0)).astype(np.float32), sr, format="WAV")
    return b.getvalue()


def test_una_tanda_de_punta_a_punta(tmp_path):
    storage = import_gpuq_storage()
    remote = storage.LocalRemote(tmp_path / "remoto")
    origen = tmp_path / "remoto" / vp.ORIGEN / "lote_x"
    origen.mkdir(parents=True)
    temas = [{"estado": "ok", "beatport_id": 11, "archivo": "11.wav"}, {"estado": "ok", "beatport_id": 12, "archivo": "12.wav"},
             {"estado": "ok", "beatport_id": 13, "archivo": "13.wav"}]          # el 13 no está en el .tar
    (origen / "tanda_00000.json").write_text(json.dumps({"temas": temas}), encoding="utf-8")
    with tarfile.open(origen / "tanda_00000.tar", "w") as tar:
        for nombre, datos in (("11.wav", _wav(5, 0.5)), ("12.wav", _wav(3, 0.1)), ("otro.wav", _wav(1, 0.2))):
            info = tarfile.TarInfo(nombre)
            info.size = len(datos)
            tar.addfile(info, io.BytesIO(datos))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    meta = vp.procesar_tanda(remote, "lote_x", "tanda_00000", _Falso(), scratch, lote=1)
    assert meta["hechos"] == 2 and [f["bp_id"] for f in meta["fallidos"]] == ["13"]
    z = np.load(tmp_path / "remoto" / vp.DESTINO / "lote_x" / "tanda_00000.npz")
    assert list(z["bp_id"]) == ["11", "12"]
    assert z["clap"].shape == (2, vp.N_SEG, 512) and z["styles"].shape == (2, vp.N_SEG, 400)
    assert z["clap"][0, 0, 0] > z["clap"][1, 0, 0]                      # cada fila es su preview
    assert np.allclose(z["segundos"], [5.0, 3.0])
    assert json.loads(str(z["meta"]))["tanda"] == "tanda_00000"
    assert not list(scratch.glob("*.tar")) and not list(scratch.glob("*.npz"))
