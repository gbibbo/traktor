"""
PURPOSE: Mantiene docs/STATUS.md chico y con sus tres secciones. El hook SessionStart lo inyecta entero
         y Claude Code solo pasa completa una salida de hook de hasta 10.000 caracteres (más larga, llega
         un adelanto de 2.000). El tope de 6 KB deja margen y evita que el bloque crezca como crónica.
CHANGELOG:
  - 2026-09-29: Creación inicial.
  - 2026-11-13: El tope se mide sobre el contenido con LF. En Windows, con core.autocrlf, un checkout deja el
                archivo con CRLF (un byte más por línea) y el test fallaba con el mismo contenido que en Git.
"""
from pathlib import Path

STATUS = Path(__file__).resolve().parents[1] / "docs" / "STATUS.md"
MAX_BYTES = 6 * 1024


def test_status_doc_fits_and_has_sections():
    data = STATUS.read_bytes().replace(b"\r\n", b"\n")
    assert len(data) <= MAX_BYTES, f"docs/STATUS.md pesa {len(data)} bytes (tope {MAX_BYTES})"
    text = data.decode("utf-8")
    for section in ("## CURRENT STATE", "## OPEN ITEMS", "## RUN RECIPES"):
        assert section in text, f"falta la sección {section!r}"
