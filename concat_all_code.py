#!/usr/bin/env python3
"""
PURPOSE: Concatenar todo el código del proyecto traktor en un único .txt.
CHANGELOG:
  - 2026-02-05: Creación inicial, adaptado de opro3_final.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "TODO_EL_CODIGO.txt"

# Carpetas a recorrer recursivamente
FOLDERS = ["scripts", "slurm", "tests", "docs", "plans", "legacy"]

# Extensiones válidas
VALID_EXTENSIONS = {".py", ".sh", ".job", ".md", ".txt", ".yaml", ".yml", ".json"}

# Directorios a excluir
EXCLUDE_DIRS = {"__pycache__", ".git", "data", "results", "logs", "models", "playlists"}


def should_include(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
    return path.suffix.lower() in VALID_EXTENSIONS


def main():
    all_files = []

    # Archivos dentro de las carpetas especificadas
    for folder in FOLDERS:
        folder_path = ROOT / folder
        if folder_path.exists():
            for file_path in folder_path.rglob("*"):
                if file_path.is_file() and should_include(file_path):
                    all_files.append(file_path)

    # Archivos sueltos en la raíz (.py, .md, .sh, etc.)
    for file_path in ROOT.iterdir():
        if (
            file_path.is_file()
            and file_path.suffix.lower() in VALID_EXTENSIONS
            and file_path != OUTPUT
        ):
            all_files.append(file_path)

    all_files.sort()

    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("CONTENIDO COMPLETO DEL PROYECTO TRAKTOR\n")
        out.write(f"Total de archivos: {len(all_files)}\n")
        out.write("=" * 80 + "\n\n")

        for i, file_path in enumerate(all_files, 1):
            rel_path = file_path.relative_to(ROOT)
            out.write("\n" + "=" * 80 + "\n")
            out.write(f"[{i}/{len(all_files)}] ARCHIVO: {rel_path}\n")
            out.write(f"RUTA COMPLETA: {file_path}\n")
            out.write("=" * 80 + "\n\n")

            try:
                content = file_path.read_text(encoding="utf-8")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            except Exception as e:
                out.write(f"[ERROR al leer: {e}]\n")

            out.write("\n")

    print(f"\nArchivo generado: {OUTPUT}")
    print(f"Archivos incluidos: {len(all_files)}\n")
    for f in all_files:
        print(f"  - {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
