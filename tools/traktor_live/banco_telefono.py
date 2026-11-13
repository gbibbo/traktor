"""
PURPOSE: El banco de regresión del teléfono (línea B, idea L11): mide, casilla por casilla, cuánto de lo que entiende el
         sistema coincide con lo esperado en config/telefono/banco_pedidos.jsonl (la semilla de la auditoría y sus
         variantes; los pedidos reales del registro del teléfono se agregan después). Lo que un pedido no nombra en
         «espera» tiene que quedar en su valor por defecto: así se mide también que no se inventen géneros.
         Dos medidas, con los umbrales de config/telefono/banco_umbral.json, fijados antes de medir:
         - reglas solas: de lo que llenan, cuánto está bien (precisión por casilla; tienen que leer con certeza);
         - todo el intérprete (reglas y LLM): cuánto está bien (exactitud por casilla), y géneros inventados.
         Un intérprete es una función (pedido, hay_set) -> {"formulario": {...}, "salida": ..., "no_se_oye": [...]}.
         Línea de comandos: python tools/traktor_live/banco_telefono.py evaluar --interprete hoy
CHANGELOG:
  - 2026-11-13: Creación inicial (línea B, B3: el banco antes que las reglas).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.traktor_live import ejecutor as ej  # noqa: E402
from tools.traktor_live import formulario as fm  # noqa: E402

BANCO_PATH = REPO_ROOT / "config" / "telefono" / "banco_pedidos.jsonl"
UMBRAL_PATH = REPO_ROOT / "config" / "telefono" / "banco_umbral.json"
CASILLAS = ("generos", "bpm", "tonalidad", "voz", "referencia", "momento", "texto", "otros", "pesos", "orden",
            "largo", "salida", "no_se_oye")
SALIDAS = ("lista", "pregunta", "no_hay", "pide_detalle")
Interprete = Callable[[str, bool], dict]


def cargar(path: Path = BANCO_PATH) -> List[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _palabras(texto: Optional[str]) -> set:
    return {w for w in ej.normalizar(texto or "").split() if w not in ej.VACIAS}


def _nombres(lista: Sequence[str]) -> set:
    return {ej.normalizar(x) for x in lista}


def esperado(entrada: dict) -> dict:
    """Lo esperado, completo: lo que dice «espera» sobre los valores por defecto."""
    e = entrada["espera"]
    form = fm.vacio()
    f = form["filtros"]
    for k in ("bpm", "generos", "tonalidad", "voz"):
        if k in e:
            f[k] = e[k]
    if "otros" in e:
        f["otros"].update(e["otros"])
    for k in ("referencia", "pesos", "momento", "orden", "largo"):
        if k in e:
            form[k] = e[k]
    return {"formulario": form, "texto": bool(e.get("texto")), "salida": e.get("salida", "lista"),
            "no_se_oye": bool(e.get("no_se_oye"))}


def _generos(g) -> tuple:
    if not g:
        return (frozenset(), frozenset(), None)
    incl = frozenset(g.get("incluir") or [])
    return (incl, frozenset(g.get("excluir") or []), g.get("modo", "blando") if incl else None)


def comparar(pred: dict, entrada: dict) -> Dict[str, bool]:
    """Para cada casilla, si lo que entendió el intérprete coincide con lo esperado."""
    esp = esperado(entrada)
    form, _ = fm.validar(pred.get("formulario") or {})
    ef, pf = esp["formulario"], form
    out: Dict[str, bool] = {}
    out["generos"] = _generos(pf["filtros"]["generos"]) == _generos(ef["filtros"]["generos"])
    out["bpm"] = fm.validar({"filtros": {"bpm": ef["filtros"]["bpm"]}})[0]["filtros"]["bpm"] == pf["filtros"]["bpm"]
    out["tonalidad"] = pf["filtros"]["tonalidad"] == ef["filtros"]["tonalidad"]
    out["voz"] = pf["filtros"]["voz"] == ef["filtros"]["voz"]
    er, pr = ef["referencia"], pf["referencia"]
    out["referencia"] = er["tipo"] == pr["tipo"] and (
        er["tipo"] in ("ninguna", "sonando") or bool(pr.get("uid")) or _palabras(er.get("busca")) <= _palabras(pr.get("busca")))
    out["momento"] = pf["momento"] == ef["momento"]
    out["texto"] = bool(pf["texto"]) == esp["texto"]
    eo, po = ef["filtros"]["otros"], pf["filtros"]["otros"]
    out["otros"] = (eo["anio"] == po["anio"] and eo["excluir_tocados"] == po["excluir_tocados"]
                    and all(_nombres(eo[k]) == _nombres(po[k]) for k in ("sellos", "artistas", "artistas_excluir")))
    out["pesos"] = (pf["pesos"]["perfil"] == ef["pesos"]["perfil"]
                    and {(a["factor"], a["nivel"]) for a in pf["pesos"]["ajustes"]} == {(a["factor"], a["nivel"]) for a in ef["pesos"]["ajustes"]})
    out["orden"] = pf["orden"] == ef["orden"]
    out["largo"] = pf["largo"] == ef["largo"]
    out["salida"] = pred.get("salida", "lista") == esp["salida"]
    out["no_se_oye"] = bool(pred.get("no_se_oye")) == esp["no_se_oye"]
    return out


def lleno(pred: dict) -> Dict[str, bool]:
    """Qué casillas llenó el intérprete (distintas de su valor por defecto): para la precisión de las reglas."""
    form, _ = fm.validar(pred.get("formulario") or {})
    v = fm.vacio()
    return {"generos": form["filtros"]["generos"] is not None, "bpm": form["filtros"]["bpm"] is not None,
            "tonalidad": form["filtros"]["tonalidad"] is not None, "voz": form["filtros"]["voz"] != "da_igual",
            "referencia": form["referencia"]["tipo"] != "ninguna", "momento": form["momento"] is not None,
            "texto": bool(form["texto"]), "otros": form["filtros"]["otros"] != v["filtros"]["otros"],
            "pesos": form["pesos"] != v["pesos"], "orden": form["orden"] != v["orden"], "largo": form["largo"] != v["largo"],
            "salida": pred.get("salida", "lista") != "lista", "no_se_oye": bool(pred.get("no_se_oye"))}


def evaluar(interprete: Interprete, banco: Sequence[dict]) -> dict:
    """Exactitud y precisión por casilla, géneros inventados y los pedidos que fallan."""
    bien, llenas, llenas_bien = Counter(), Counter(), Counter()
    fallas: Dict[str, List[str]] = defaultdict(list)
    inventados: List[str] = []
    for entrada in banco:
        pred = interprete(entrada["pedido"], entrada["set"])
        cmp_, fill = comparar(pred, entrada), lleno(pred)
        for c in CASILLAS:
            bien[c] += cmp_[c]
            if fill[c]:
                llenas[c] += 1
                llenas_bien[c] += cmp_[c]
            if not cmp_[c]:
                fallas[c].append(entrada["id"])
        if "generos" not in entrada["espera"] and fill["generos"]:
            inventados.append(entrada["id"])
    n = len(banco)
    return {"n": n,
            "exactitud": {c: round(bien[c] / n, 3) for c in CASILLAS},
            "precision": {c: (round(llenas_bien[c] / llenas[c], 3) if llenas[c] else None) for c in CASILLAS},
            "llenas": {c: llenas[c] for c in CASILLAS},
            "generos_inventados": inventados,
            "fallas": {c: v for c, v in fallas.items()}}


def cumple(resultado: dict, modo: str, umbral: Optional[dict] = None) -> List[str]:
    """Lo que no cumple el umbral fijado antes de medir. modo: «reglas» o «completo»."""
    u = umbral or json.loads(UMBRAL_PATH.read_text(encoding="utf-8"))
    problemas = []
    if modo == "reglas":
        minimo = u["reglas"]["precision_minima"]
        for c, p in resultado["precision"].items():
            if p is not None and p < minimo:
                problemas.append(f"reglas: precisión de {c} {p} < {minimo}")
    else:
        minimo = u["completo"]["exactitud_minima"]
        for c, a in resultado["exactitud"].items():
            if a < minimo:
                problemas.append(f"completo: exactitud de {c} {a} < {minimo}")
    if len(resultado["generos_inventados"]) > u["generos_inventados_maximo"]:
        problemas.append(f"géneros inventados en {resultado['generos_inventados']}")
    return problemas


def interprete_hoy(pedido: str, hay_set: bool) -> dict:
    """El circuito de la v5, como intérprete: todo va a la casilla texto y nada se filtra."""
    return {"formulario": {"texto": ["whatever the llm wrote"]}, "salida": "lista", "no_se_oye": []}


class InterpreteReglas:
    """Las reglas del Worker (cloudflare/reglas.js con config/telefono/reglamento.json), corridas con Node sobre todo el
    banco de una vez. Mismo código que en producción: no hay una copia en Python."""

    def __init__(self, banco_path: Path = BANCO_PATH, anio: int = 2026):
        import shutil
        import subprocess
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Hace falta Node.js para correr las reglas del Worker")
        salida = subprocess.run([node, str(REPO_ROOT / "cloudflare" / "banco_reglas.mjs"), str(banco_path), str(anio)],
                                capture_output=True, text=True, encoding="utf-8", check=True).stdout
        por_id = {r["id"]: r["pred"] for r in (json.loads(l) for l in salida.splitlines() if l.strip())}
        self.pred = {(x["pedido"], x["set"]): por_id[x["id"]] for x in cargar(banco_path)}

    def __call__(self, pedido: str, hay_set: bool) -> dict:
        return self.pred[(pedido, hay_set)]


INTERPRETES: Dict[str, Callable[[], Interprete]] = {"hoy": lambda: interprete_hoy, "reglas": InterpreteReglas}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Banco de regresión del teléfono")
    sub = ap.add_subparsers(dest="accion", required=True)
    ev = sub.add_parser("evaluar")
    ev.add_argument("--interprete", choices=sorted(INTERPRETES), default="hoy")
    ev.add_argument("--modo", choices=("reglas", "completo"), default="completo")
    ev.add_argument("--fallas", action="store_true", help="listar los pedidos que fallan en cada casilla")
    args = ap.parse_args(argv)
    banco = cargar()
    r = evaluar(INTERPRETES[args.interprete](), banco)
    print(f"[INFO] {r['n']} pedidos; intérprete {args.interprete}; modo {args.modo}")
    for c in CASILLAS:
        p = r["precision"][c]
        print(f"   {c:<11} exactitud {r['exactitud'][c]:.3f}   llenas {r['llenas'][c]:>3}   "
              f"precisión {'-' if p is None else f'{p:.3f}'}")
        if args.fallas and r["fallas"].get(c):
            print("      falla en: " + ", ".join(r["fallas"][c]))
    print(f"[INFO] géneros inventados: {len(r['generos_inventados'])} {r['generos_inventados'][:12]}")
    problemas = cumple(r, args.modo)
    for p in problemas:
        print(f"[NO CUMPLE] {p}")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
