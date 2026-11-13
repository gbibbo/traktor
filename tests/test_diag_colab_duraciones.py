"""
PURPOSE: tools/gpuq/diag_colab/duraciones.py empareja cada intento de Colab con su resultado y resume por período.
CHANGELOG:
  - 2026-11-13: Creación (informe docs/reports/colab_corte_13min_2026-11-13.md).
"""
import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gpuq" / "diag_colab" / "duraciones.py"
spec = importlib.util.spec_from_file_location("duraciones", TOOL)
duraciones = importlib.util.module_from_spec(spec)
spec.loader.exec_module(duraciones)

LOG = """[2026-10-30 07:41:12] colab: starting largo-1 attempt 1
[2026-10-30 07:50:00] kaggle: starting otro-1 attempt 1
[2026-10-30 09:11:29] colab: largo-1 -> done
[2026-11-02 17:22:24] colab: starting corto-1 attempt 2
[2026-11-02 17:30:00] stable moved, restarting on the new code
[2026-11-02 17:34:45] colab: corto-1 -> interrupted
[2026-11-02 17:34:49] colab: starting corto-1 attempt 3
"""


def test_pairs_attempts_and_ignores_other_lines():
    rows = duraciones.attempts(LOG)
    assert [(r["job"], r["n"], r["outcome"]) for r in rows] == [("largo-1", 1, "done"), ("corto-1", 2, "interrupted")]
    assert round(rows[0]["min"], 1) == 90.3 and round(rows[1]["min"], 1) == 12.3


def test_main_splits_by_period(tmp_path, capsys):
    log = tmp_path / "gpuq.txt"
    log.write_text(LOG, encoding="utf-8")
    duraciones.main([str(log), "--corte", "2026-11-02 17:00"])
    out = capsys.readouterr().out
    assert "GPU before 2026-11-02 17:00 UTC: done 1 (median 90.3 min" in out
    assert "GPU from 2026-11-02 17:00 UTC: interrupted 1 (median 12.3 min" in out
