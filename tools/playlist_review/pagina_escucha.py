"""
PURPOSE: Página autónoma para una escucha a ciegas de listas de a dos (idea E12 del plan de recomendación). Lee una
         carpeta con pares `NN_A.m3u8` y `NN_B.m3u8` (el tema principal primero y después sus candidatos) y escribe un
         HTML que se abre desde el disco: reproduce los archivos locales, marca los temas que están en las dos listas,
         guarda en el navegador qué lista prefiere Gabriel para cada tema principal y exporta las respuestas a JSON.
         No lee ningún otro archivo de la carpeta: la clave de cuál lista es cuál no entra en la página.
         Uso: python tools/playlist_review/pagina_escucha.py --listas "<carpeta>" [--out escucha.html] [--id <nombre>]
CHANGELOG:
  - 2026-11-13: Creación inicial (pedido de Gabriel: un HTML con las indicaciones para hacer la escucha E12).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

PAR = re.compile(r"^(\d+)_([AB])\.m3u8$", re.I)


def leer_m3u8(path: Path) -> List[Tuple[str, str]]:
    """[(título, ruta)] en el orden de la lista. El título sale de #EXTINF; si falta, del nombre del archivo."""
    out: List[Tuple[str, str]] = []
    titulo: Optional[str] = None
    for linea in path.read_text(encoding="utf-8-sig").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        if linea.upper().startswith("#EXTINF"):
            titulo = linea.split(",", 1)[1].strip() if "," in linea else None
        elif not linea.startswith("#"):
            out.append((titulo or PureWindowsPath(linea).stem, linea))
            titulo = None
    return out


def uri(ruta: str) -> str:
    """file:// de una ruta de Windows o POSIX, sin tocar el disco."""
    win = PureWindowsPath(ruta)
    return win.as_uri() if win.drive else Path(ruta).as_uri()


def armar(carpeta: Path) -> dict:
    """Los pares de la carpeta, con la marca de los temas que están en las dos listas."""
    pares: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for f in sorted(carpeta.iterdir()):
        m = PAR.match(f.name)
        if m:
            pares.setdefault(m.group(1), {})[m.group(2).upper()] = leer_m3u8(f)
    temas, faltan = [], 0
    for n in sorted(pares, key=int):
        par = pares[n]
        if set(par) != {"A", "B"} or not par["A"] or not par["B"]:
            raise SystemExit(f"[ERROR] El tema {n} no tiene sus dos listas (A y B) con contenido")
        if par["A"][0][1] != par["B"][0][1]:
            raise SystemExit(f"[ERROR] Las listas A y B del tema {n} no empiezan con el mismo tema principal")
        en_a = {r for _, r in par["A"][1:]}
        en_b = {r for _, r in par["B"][1:]}

        def filas(lista: List[Tuple[str, str]]) -> List[dict]:
            return [{"t": t, "u": uri(r), "c": int(r in en_a and r in en_b)} for t, r in lista[1:]]

        for _, r in par["A"] + par["B"][1:]:
            faltan += int(not Path(r).exists())
        temas.append({"n": n, "p": {"t": par["A"][0][0], "u": uri(par["A"][0][1])},
                      "A": filas(par["A"]), "B": filas(par["B"]), "comunes": len(en_a & en_b)})
    if not temas:
        raise SystemExit(f"[ERROR] No hay pares NN_A.m3u8 y NN_B.m3u8 en {carpeta}")
    return {"temas": temas, "faltan": faltan}


def pagina(datos: dict, ident: str, titulo: str) -> str:
    cuerpo = json.dumps({"id": ident, "temas": datos["temas"]}, ensure_ascii=False).replace("</", "<\\/")
    return PLANTILLA.replace("__TITULO__", titulo).replace("__DATOS__", cuerpo)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Página de escucha a ciegas de listas de a dos")
    ap.add_argument("--listas", required=True, type=Path, help="carpeta con NN_A.m3u8 y NN_B.m3u8")
    ap.add_argument("--out", type=Path, default=None, help="por defecto, escucha.html al lado de la carpeta")
    ap.add_argument("--id", default=None, help="nombre de la escucha; identifica las respuestas guardadas")
    ap.add_argument("--titulo", default="Escucha a ciegas")
    args = ap.parse_args(argv)
    carpeta = args.listas.resolve()
    datos = armar(carpeta)
    out = args.out or carpeta.parent / "escucha.html"
    ident = args.id or carpeta.parent.name
    out.write_text(pagina(datos, ident, args.titulo), encoding="utf-8", newline="\n")
    comunes = [t["comunes"] for t in datos["temas"]]
    print(f"[OK] {out}: {len(datos['temas'])} temas principales, {sum(len(t['A']) + len(t['B']) for t in datos['temas'])} "
          f"candidatos; en las dos listas, de {min(comunes)} a {max(comunes)} por tema; archivos que no existen: {datos['faltan']}")
    return 0


PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITULO__</title>
<style>
:root { --fondo:#f6f5f2; --papel:#ffffff; --tinta:#1d1c1a; --suave:#6b6862; --linea:#dcd9d2; --acento:#0b6e6e; --sonando:#e3f1f0; --comun:#9a968e; }
@media (prefers-color-scheme: dark) {
  :root { --fondo:#161615; --papel:#1f1f1d; --tinta:#eceae4; --suave:#a09c93; --linea:#35342f; --acento:#4fc3c0; --sonando:#173534; --comun:#77736b; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--fondo); color:var(--tinta); font:16px/1.5 system-ui, "Segoe UI", sans-serif; padding:0 16px 150px; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:1.5rem; margin:24px 0 4px; }
h2 { font-size:1.15rem; margin:0 0 8px; }
p { margin:0 0 10px; }
.suave { color:var(--suave); }
.tarjeta { background:var(--papel); border:1px solid var(--linea); border-radius:10px; padding:16px; margin:14px 0; }
details summary { cursor:pointer; font-weight:600; }
details ol, details ul { margin:8px 0 0; padding-left:22px; }
details li { margin:4px 0; }
nav { display:flex; flex-wrap:wrap; gap:6px; margin:14px 0; }
button { font:inherit; color:var(--tinta); background:var(--papel); border:1px solid var(--linea); border-radius:8px; padding:6px 12px; cursor:pointer; }
button:hover { border-color:var(--acento); }
nav button.actual { border-color:var(--acento); box-shadow:0 0 0 2px var(--acento) inset; }
nav button.listo::after { content:" ✓"; color:var(--acento); }
.principal { display:flex; flex-wrap:wrap; align-items:center; gap:12px; }
.principal strong { font-size:1.1rem; }
.listas { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
@media (max-width:760px) { .listas { grid-template-columns:1fr; } }
ol.lista { list-style:none; margin:0; padding:0; counter-reset:n; }
ol.lista li { counter-increment:n; display:flex; gap:8px; align-items:baseline; padding:5px 8px; border-radius:6px; cursor:pointer; }
ol.lista li::before { content:counter(n); color:var(--suave); min-width:1.6em; text-align:right; font-variant-numeric:tabular-nums; }
ol.lista li:hover { background:var(--sonando); }
ol.lista li.sonando { background:var(--sonando); font-weight:600; }
ol.lista li.comun { color:var(--comun); }
ol.lista li.comun::after { content:"en las dos"; font-size:.75rem; border:1px solid var(--linea); border-radius:10px; padding:0 7px; margin-left:auto; white-space:nowrap; }
.opciones { display:flex; flex-wrap:wrap; gap:14px; align-items:center; }
.opciones label { cursor:pointer; }
textarea { width:100%; font:inherit; color:var(--tinta); background:var(--fondo); border:1px solid var(--linea); border-radius:8px; padding:8px; }
select, input[type=number] { font:inherit; color:var(--tinta); background:var(--papel); border:1px solid var(--linea); border-radius:6px; padding:3px 6px; }
#barra { position:fixed; left:0; right:0; bottom:0; background:var(--papel); border-top:1px solid var(--linea); padding:10px 16px; }
#barra div { max-width:1100px; margin:0 auto; }
#barra audio { width:100%; margin-top:6px; }
#ahora { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.fila { display:flex; flex-wrap:wrap; gap:14px; align-items:center; }
</style>
</head>
<body>
<main>
<h1>__TITULO__</h1>
<p class="suave" id="avance"></p>

<details class="tarjeta" open>
<summary>Cómo hacer la escucha</summary>
<ol>
<li><strong>Qué decides.</strong> Para cada tema principal hay dos listas de candidatos, A y B. Una es la que el recomendador arma hoy y la otra es la nueva. Cuál es cuál se sorteó para cada tema y esta página no lo sabe.</li>
<li><strong>La pregunta.</strong> «Si estuviera sonando el tema principal, ¿con cuál de las dos listas me quedaría para elegir lo que sigue?» Vale más una lista donde casi todo se puede mezclar con seguridad que una con dos o tres aciertos brillantes y mucho relleno.</li>
<li><strong>Cómo escuchar.</strong> Escucha primero el tema principal. Después recorre una lista y luego la otra, en el orden que te indica cada tema, que va alternando. Un clic en un tema lo reproduce desde el medio. Los primeros de cada lista son los que el sistema considera más parecidos.</li>
<li><strong>Cuánto escuchar.</strong> No hace falta oír los temas enteros. Con «recorrer sola» la página pasa al siguiente cada 20 segundos. Escucha al menos los 15 primeros de cada lista.</li>
<li><strong>Los temas «en las dos».</strong> Están en A y en B, así que no ayudan a decidir. Puedes saltarlos.</li>
<li><strong>Contesta</strong> A, B o «no las distingo», y pasa al tema siguiente. Si quieres, deja una nota.</li>
<li><strong>Al terminar,</strong> usa «Descargar respuestas» y pásale el archivo al agente, o copia el texto en el chat.</li>
</ol>
<ul>
<li>Puedes hacerla en varias sentadas: las respuestas quedan guardadas en este navegador.</li>
<li>No abras <code>clave.json</code> hasta haber contestado todo.</li>
<li>Si dudas con un tema, las mismas listas están como archivos <code>.m3u8</code> en la carpeta <code>listas</code>: se pueden arrastrar a Traktor para probar la mezcla de verdad.</li>
<li>Teclas: espacio pausa; flecha abajo y flecha arriba cambian de tema dentro de la lista.</li>
</ul>
</details>

<nav id="nav"></nav>

<section class="tarjeta">
  <div class="principal">
    <div><div class="suave" id="rotulo"></div><strong id="ptitulo"></strong></div>
    <button id="pplay">▶ Escuchar el principal</button>
  </div>
  <p class="suave" id="orden" style="margin-top:10px"></p>
  <div class="fila">
    <label>Empezar cada tema en
      <select id="desde"><option value="0.35">el medio (35 %)</option><option value="0">el comienzo</option><option value="0.6">más adelante (60 %)</option></select>
    </label>
    <label><input type="checkbox" id="sola"> Recorrer sola, cada <input type="number" id="seg" min="5" max="120" value="20" style="width:4.5em"> s</label>
  </div>
</section>

<div class="listas">
  <section class="tarjeta"><h2>Lista A</h2><ol class="lista" id="LA"></ol></section>
  <section class="tarjeta"><h2>Lista B</h2><ol class="lista" id="LB"></ol></section>
</div>

<section class="tarjeta">
  <h2>Tu respuesta para este tema</h2>
  <div class="opciones">
    <label><input type="radio" name="pref" value="A"> Prefiero la A</label>
    <label><input type="radio" name="pref" value="B"> Prefiero la B</label>
    <label><input type="radio" name="pref" value="igual"> No las distingo</label>
    <button id="sig">Tema siguiente →</button>
  </div>
  <p style="margin-top:10px"><textarea id="nota" rows="2" placeholder="Nota opcional: por qué, o qué te llamó la atención"></textarea></p>
</section>

<section class="tarjeta">
  <h2>Al terminar</h2>
  <div class="fila"><button id="bajar">Descargar respuestas</button><button id="copiar">Copiar respuestas</button><span class="suave" id="aviso"></span></div>
  <p style="margin-top:10px"><textarea id="salida" rows="5" readonly placeholder="Acá aparece el texto de las respuestas"></textarea></p>
</section>
</main>

<div id="barra"><div>
  <div id="ahora" class="suave">Nada sonando</div>
  <audio id="au" controls preload="none"></audio>
</div></div>

<script>
const D = __DATOS__;
const CLAVE = 'escucha:' + D.id;
let R = {};
try { R = JSON.parse(localStorage.getItem(CLAVE) || '{}') || {}; } catch (e) { R = {}; }
let actual = 0, sonando = null, reloj = null;
const $ = id => document.getElementById(id);
const au = $('au');

function guardar() { try { localStorage.setItem(CLAVE, JSON.stringify(R)); } catch (e) {} pintarNav(); }
function tema() { return D.temas[actual]; }

function pintarNav() {
  const nav = $('nav'); nav.textContent = '';
  D.temas.forEach((t, i) => {
    const b = document.createElement('button');
    b.textContent = 'Tema ' + Number(t.n);
    if (i === actual) b.classList.add('actual');
    if (R[t.n] && R[t.n].prefiere) b.classList.add('listo');
    b.onclick = () => ir(i);
    nav.appendChild(b);
  });
  const hechos = D.temas.filter(t => R[t.n] && R[t.n].prefiere).length;
  $('avance').textContent = hechos + ' de ' + D.temas.length + ' temas contestados';
}

function pintarLista(id, letra) {
  const ol = $(id); ol.textContent = '';
  tema()[letra].forEach((c, i) => {
    const li = document.createElement('li');
    const s = document.createElement('span'); s.textContent = c.t; li.appendChild(s);
    if (c.c) li.classList.add('comun');
    li.onclick = () => tocar(letra, i);
    ol.appendChild(li);
  });
}

function ir(i, sinMover) {
  parar();
  actual = Math.max(0, Math.min(D.temas.length - 1, i));
  const t = tema();
  $('rotulo').textContent = 'Tema principal ' + Number(t.n) + ' de ' + D.temas.length;
  $('ptitulo').textContent = t.p.t;
  const primero = Number(t.n) % 2 ? 'A' : 'B';
  $('orden').textContent = 'En este tema empieza por la lista ' + primero + '. Comparten ' + t.comunes + ' temas.';
  pintarLista('LA', 'A'); pintarLista('LB', 'B');
  const r = R[t.n] || {};
  document.querySelectorAll('input[name=pref]').forEach(x => { x.checked = x.value === r.prefiere; });
  $('nota').value = r.nota || '';
  pintarNav();
  if (!sinMover) window.scrollTo({ top: $('nav').offsetTop - 10 });
}

function marcar() {
  document.querySelectorAll('ol.lista li').forEach(li => li.classList.remove('sonando'));
  if (sonando && sonando.l !== 'P') {
    const li = $('L' + sonando.l).children[sonando.i];
    if (li) { li.classList.add('sonando'); li.scrollIntoView({ block: 'nearest' }); }
  }
}

function parar() { clearTimeout(reloj); au.pause(); sonando = null; marcar(); $('ahora').textContent = 'Nada sonando'; }

function tocar(l, i) {
  clearTimeout(reloj);
  const t = tema(), c = l === 'P' ? t.p : t[l][i];
  if (!c) { parar(); return; }
  sonando = { l: l, i: i };
  const desde = parseFloat($('desde').value);
  au.src = c.u;
  au.onloadedmetadata = () => { if (desde > 0 && isFinite(au.duration)) au.currentTime = au.duration * desde; };
  au.onerror = () => { $('ahora').textContent = 'No se pudo abrir: ' + c.t; };
  au.play().catch(() => {});
  $('ahora').textContent = (l === 'P' ? 'Principal' : 'Lista ' + l + ', puesto ' + (i + 1)) + ': ' + c.t;
  marcar();
  if ($('sola').checked && l !== 'P') {
    const seg = Math.max(5, Math.min(120, parseInt($('seg').value, 10) || 20));
    reloj = setTimeout(() => mover(1), seg * 1000);
  }
}

function mover(paso) {
  if (!sonando || sonando.l === 'P') return;
  const i = sonando.i + paso, n = tema()[sonando.l].length;
  if (i < 0 || i >= n) { parar(); return; }
  tocar(sonando.l, i);
}

function exportar() {
  const texto = JSON.stringify({ escucha: D.id, exportado: new Date().toISOString(), respuestas: R }, null, 1);
  $('salida').value = texto;
  return texto;
}

$('pplay').onclick = () => tocar('P', 0);
$('sig').onclick = () => ir(actual + 1);
au.onended = () => mover(1);
document.querySelectorAll('input[name=pref]').forEach(x => {
  x.onchange = () => { const n = tema().n; R[n] = Object.assign(R[n] || {}, { prefiere: x.value }); guardar(); };
});
$('nota').oninput = () => { const n = tema().n; R[n] = Object.assign(R[n] || {}, { nota: $('nota').value }); guardar(); };
$('bajar').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([exportar()], { type: 'application/json' }));
  a.download = 'respuestas_' + D.id + '.json';
  document.body.appendChild(a); a.click(); a.remove();
  $('aviso').textContent = 'Archivo descargado.';
};
$('copiar').onclick = () => {
  const texto = exportar();
  const ok = () => { $('aviso').textContent = 'Copiado.'; };
  const mano = () => { $('salida').select(); $('aviso').textContent = 'Selecciona el texto de abajo y cópialo.'; };
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(texto).then(ok, mano); else mano();
};
document.addEventListener('keydown', e => {
  if (['TEXTAREA', 'INPUT', 'SELECT', 'BUTTON'].includes(e.target.tagName)) return;
  if (e.code === 'Space') { e.preventDefault(); if (au.paused) au.play().catch(() => {}); else au.pause(); }
  if (e.code === 'ArrowDown') { e.preventDefault(); mover(1); }
  if (e.code === 'ArrowUp') { e.preventDefault(); mover(-1); }
});
const pendiente = D.temas.findIndex(t => !(R[t.n] && R[t.n].prefiere));
ir(pendiente < 0 ? 0 : pendiente, true);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
