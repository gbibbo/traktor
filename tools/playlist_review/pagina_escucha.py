"""
PURPOSE: Página autónoma para una escucha a ciegas de listas de a dos (idea E12 del plan de recomendación), en su
         versión corta: unos 12 minutos para 10 temas principales. Lee una carpeta con pares `NN_A.m3u8` y `NN_B.m3u8`
         (el tema principal primero y después sus candidatos) y escribe un HTML que se abre desde el disco. Para cada
         tema principal la página hace sonar un fragmento del principal y después fragmentos de los primeros temas
         que cada lista tiene y la otra no; Gabriel elige A, B o «no las distingo» con un clic y pasa al siguiente.
         Guarda las respuestas en el navegador y las exporta a JSON.
         No lee ningún otro archivo de la carpeta: la clave de cuál lista es cuál no entra en la página.
         Uso: python tools/playlist_review/pagina_escucha.py --listas "<carpeta>" [--por-lista 3] [--segundos 10]
CHANGELOG:
  - 2026-11-13: Creación inicial (pedido de Gabriel: un HTML con las indicaciones para hacer la escucha E12).
  - 2026-11-13: Versión corta. La primera mostraba las dos listas enteras (10 × 60 temas, de una hora y media a tres
                horas) y Gabriel la rechazó por inviable. Ahora cada lista aporta sus primeros temas propios, la
                página los hace sonar sola y hay una decisión por pantalla.
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
POR_LISTA = 3          # temas propios de cada lista que se escuchan
SEGUNDOS = 10          # largo de cada fragmento


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


def armar(carpeta: Path, por_lista: int = POR_LISTA) -> dict:
    """Por cada tema principal, los primeros `por_lista` temas que cada lista tiene y la otra no, en su orden."""
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
        rutas = {letra: {r for _, r in par[letra][1:]} for letra in "AB"}
        tema = {"n": n, "p": {"t": par["A"][0][0], "u": uri(par["A"][0][1])}, "comunes": len(rutas["A"] & rutas["B"])}
        faltan += int(not Path(par["A"][0][1]).exists())
        for letra, otra in (("A", "B"), ("B", "A")):
            propios = [(t, r, i) for i, (t, r) in enumerate(par[letra][1:], start=1) if r not in rutas[otra]][:por_lista]
            if not propios:
                raise SystemExit(f"[ERROR] La lista {letra} del tema {n} no tiene temas propios: es igual a la otra")
            tema[letra] = [{"t": t, "u": uri(r), "puesto": i} for t, r, i in propios]
            faltan += sum(int(not Path(r).exists()) for _, r, _ in propios)
        temas.append(tema)
    if not temas:
        raise SystemExit(f"[ERROR] No hay pares NN_A.m3u8 y NN_B.m3u8 en {carpeta}")
    return {"temas": temas, "faltan": faltan}


def minutos(datos: dict, segundos: int) -> int:
    """Lo que dura la escucha si se oye todo una vez: los fragmentos más unos segundos para decidir."""
    total = sum((1 + len(t["A"]) + len(t["B"])) * segundos + 5 for t in datos["temas"])
    return max(1, round(total / 60))


def pagina(datos: dict, ident: str, titulo: str, segundos: int = SEGUNDOS) -> str:
    cuerpo = json.dumps({"id": ident, "seg": segundos, "min": minutos(datos, segundos), "temas": datos["temas"]},
                        ensure_ascii=False).replace("</", "<\\/")
    return PLANTILLA.replace("__TITULO__", titulo).replace("__DATOS__", cuerpo)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Página de escucha a ciegas de listas de a dos, versión corta")
    ap.add_argument("--listas", required=True, type=Path, help="carpeta con NN_A.m3u8 y NN_B.m3u8")
    ap.add_argument("--out", type=Path, default=None, help="por defecto, escucha.html al lado de la carpeta")
    ap.add_argument("--id", default=None, help="nombre de la escucha; identifica las respuestas guardadas")
    ap.add_argument("--titulo", default="Escucha a ciegas")
    ap.add_argument("--por-lista", type=int, default=POR_LISTA, help="temas propios de cada lista que se escuchan")
    ap.add_argument("--segundos", type=int, default=SEGUNDOS, help="largo de cada fragmento")
    args = ap.parse_args(argv)
    carpeta = args.listas.resolve()
    datos = armar(carpeta, args.por_lista)
    out = args.out or carpeta.parent / "escucha.html"
    ident = args.id or carpeta.parent.name
    out.write_text(pagina(datos, ident, args.titulo, args.segundos), encoding="utf-8", newline="\n")
    print(f"[OK] {out}: {len(datos['temas'])} temas principales, {args.por_lista} temas por lista, fragmentos de "
          f"{args.segundos} s; dura unos {minutos(datos, args.segundos)} min; archivos que no existen: {datos['faltan']}")
    return 0


PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITULO__</title>
<style>
:root { --fondo:#f6f5f2; --papel:#ffffff; --tinta:#1d1c1a; --suave:#6b6862; --linea:#dcd9d2; --acento:#0b6e6e; --sonando:#d9eeed; }
@media (prefers-color-scheme: dark) {
  :root { --fondo:#161615; --papel:#1f1f1d; --tinta:#eceae4; --suave:#a09c93; --linea:#35342f; --acento:#4fc3c0; --sonando:#17403e; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--fondo); color:var(--tinta); font:17px/1.5 system-ui, "Segoe UI", sans-serif; padding:0 16px 40px; }
main { max-width:820px; margin:0 auto; }
h1 { font-size:1.4rem; margin:24px 0 2px; }
h2 { font-size:1.05rem; margin:0 0 8px; }
p { margin:0 0 10px; }
.suave { color:var(--suave); }
.tarjeta { background:var(--papel); border:1px solid var(--linea); border-radius:12px; padding:20px; margin:16px 0; }
button { font:inherit; color:var(--tinta); background:var(--papel); border:1px solid var(--linea); border-radius:10px; padding:10px 16px; cursor:pointer; }
button:hover { border-color:var(--acento); }
button.fuerte { background:var(--acento); border-color:var(--acento); color:var(--papel); font-weight:600; }
button.chico { padding:5px 10px; font-size:.9rem; }
#ptitulo { font-size:1.25rem; font-weight:700; margin:2px 0 14px; }
.grupos { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:16px 0; }
@media (max-width:640px) { .grupos { grid-template-columns:1fr; } }
.clip { position:relative; overflow:hidden; border:1px solid var(--linea); border-radius:8px; padding:8px 10px; margin:6px 0; cursor:pointer; }
.clip span { position:relative; }
.clip.sonando { border-color:var(--acento); font-weight:600; }
.clip.sonando::before { content:""; position:absolute; inset:0; background:var(--sonando); transform-origin:left; animation:avance var(--seg) linear forwards; }
@keyframes avance { from { transform:scaleX(0); } to { transform:scaleX(1); } }
.eleccion { display:flex; flex-wrap:wrap; gap:10px; margin-top:6px; }
.eleccion button { flex:1 1 180px; padding:14px; font-weight:600; }
.fila { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
textarea { width:100%; font:inherit; color:var(--tinta); background:var(--fondo); border:1px solid var(--linea); border-radius:8px; padding:8px; }
[hidden] { display:none !important; }
</style>
</head>
<body>
<main>
<h1>__TITULO__</h1>
<p class="suave" id="avance"></p>

<section class="tarjeta" id="intro">
  <p><strong>Son unos <span id="mins"></span> minutos.</strong> Puedes cortar cuando quieras y seguir después: se guarda en este navegador.</p>
  <p>Para cada tema principal, la página hace sonar sola un fragmento del principal y después tres temas de la lista A y tres de la lista B. Una lista es la del recomendador de hoy y la otra es la nueva; cuál es cuál está sorteado.</p>
  <p>Tú solo eliges: <strong>¿con cuál de las dos te quedarías para seguir al principal?</strong> Un clic, y pasa al tema siguiente.</p>
  <button class="fuerte" id="empezar">Empezar</button>
</section>

<section class="tarjeta" id="tema" hidden>
  <div class="suave" id="rotulo"></div>
  <div id="ptitulo"></div>
  <div class="fila">
    <button id="oir">▶ Escuchar de nuevo</button>
    <button id="pausa">Pausa</button>
    <span class="suave" id="estado"></span>
  </div>
  <div class="clip" id="cP" style="margin-top:14px"><span></span></div>
  <div class="grupos">
    <div><h2>Lista A</h2><div id="gA"></div></div>
    <div><h2>Lista B</h2><div id="gB"></div></div>
  </div>
  <div class="eleccion">
    <button data-v="A">Me quedo con la A</button>
    <button data-v="B">Me quedo con la B</button>
    <button data-v="igual">No las distingo</button>
  </div>
  <p style="margin-top:14px"><button class="chico" id="atras">← Tema anterior</button></p>
</section>

<section class="tarjeta" id="fin" hidden>
  <p><strong>Listo, gracias.</strong> Pásale las respuestas al agente: descarga el archivo o copia el texto y pégalo en el chat.</p>
  <div class="fila"><button class="fuerte" id="bajar">Descargar respuestas</button><button id="copiar">Copiar respuestas</button><button class="chico" id="revisar">Revisar una respuesta</button><span class="suave" id="aviso"></span></div>
  <p style="margin-top:10px"><textarea id="salida" rows="6" readonly></textarea></p>
</section>
</main>

<audio id="au" preload="none"></audio>

<script>
const D = __DATOS__;
const CLAVE = 'escucha_corta:' + D.id;
let R = {};
try { R = JSON.parse(localStorage.getItem(CLAVE) || '{}') || {}; } catch (e) { R = {}; }
let actual = 0, cola = [], pos = -1, reloj = null, ficha = 0, pausado = false;
const $ = id => document.getElementById(id);
const au = $('au');
document.documentElement.style.setProperty('--seg', D.seg + 's');
$('mins').textContent = D.min;

function guardar() { try { localStorage.setItem(CLAVE, JSON.stringify(R)); } catch (e) {} }
function tema() { return D.temas[actual]; }
function hechos() { return D.temas.filter(t => R[t.n] && R[t.n].prefiere).length; }
function mostrar(id) { ['intro', 'tema', 'fin'].forEach(x => { $(x).hidden = x !== id; }); }

function pintarAvance() {
  const h = hechos(), n = D.temas.length;
  const quedan = Math.max(0, Math.round((n - h) * D.min / n));
  $('avance').textContent = h + ' de ' + n + ' contestados' + (h < n ? ' · quedan unos ' + quedan + ' minutos' : '');
}

function clips() {
  const t = tema(), primero = Number(t.n) % 2 ? 'A' : 'B', segundo = primero === 'A' ? 'B' : 'A';
  const lista = l => t[l].map((c, i) => ({ el: $('g' + l).children[i], c: c, nombre: 'Lista ' + l }));
  return [{ el: $('cP'), c: t.p, nombre: 'Principal' }].concat(lista(primero), lista(segundo));
}

function pintarTema() {
  const t = tema();
  $('rotulo').textContent = 'Tema principal ' + (actual + 1) + ' de ' + D.temas.length;
  $('ptitulo').textContent = t.p.t;
  $('cP').firstChild.textContent = 'Principal: ' + t.p.t;
  ['A', 'B'].forEach(l => {
    const g = $('g' + l); g.textContent = '';
    t[l].forEach(c => {
      const d = document.createElement('div'); d.className = 'clip';
      const s = document.createElement('span'); s.textContent = c.t; d.appendChild(s); g.appendChild(d);
    });
  });
  clips().forEach((k, i) => { k.el.onclick = () => { cola = [clips()[i]]; tocar(0); }; });
  const r = R[t.n] || {};
  document.querySelectorAll('.eleccion button').forEach(b => b.classList.toggle('fuerte', b.dataset.v === r.prefiere));
  $('atras').hidden = actual === 0;
  pintarAvance();
}

function apagar() {
  clearTimeout(reloj); ficha++; au.pause(); pos = -1; pausado = false;
  document.querySelectorAll('.clip').forEach(x => x.classList.remove('sonando'));
  $('pausa').textContent = 'Pausa';
}

function tocar(k) {
  clearTimeout(reloj);
  document.querySelectorAll('.clip').forEach(x => x.classList.remove('sonando'));
  if (k >= cola.length) { pos = -1; $('estado').textContent = 'Elige A, B o «no las distingo».'; return; }
  pos = k; pausado = false; $('pausa').textContent = 'Pausa';
  const mia = ++ficha, item = cola[k];
  let arrancado = false;
  $('estado').textContent = item.nombre;
  au.onloadedmetadata = () => { if (mia === ficha && isFinite(au.duration)) au.currentTime = au.duration * 0.35; };
  au.onplaying = () => {
    if (mia !== ficha || arrancado) return;
    arrancado = true;
    void item.el.offsetWidth; item.el.classList.add('sonando');
    reloj = setTimeout(() => { if (mia === ficha) tocar(k + 1); }, D.seg * 1000);
  };
  au.onerror = () => {
    if (mia !== ficha) return;
    $('estado').textContent = 'No se pudo abrir: ' + item.c.t;
    reloj = setTimeout(() => { if (mia === ficha) tocar(k + 1); }, 1200);
  };
  au.src = item.c.u;
  au.play().catch(() => {});
}

function secuencia() { apagar(); cola = clips(); tocar(0); }

function ir(i, sonar) {
  apagar();
  if (i >= D.temas.length) { terminar(); return; }
  actual = Math.max(0, i);
  mostrar('tema'); pintarTema();
  if (sonar) secuencia(); else $('estado').textContent = '';
}

function exportar() {
  const texto = JSON.stringify({ escucha: D.id, protocolo: 'corta', segundos: D.seg, exportado: new Date().toISOString(), respuestas: R }, null, 1);
  $('salida').value = texto;
  return texto;
}

function terminar() { apagar(); mostrar('fin'); pintarAvance(); exportar(); }

$('empezar').onclick = () => { const p = D.temas.findIndex(t => !(R[t.n] && R[t.n].prefiere)); ir(p < 0 ? 0 : p, true); };
$('oir').onclick = secuencia;
$('pausa').onclick = () => {
  if (pos < 0) return;
  if (!pausado) { pausado = true; clearTimeout(reloj); au.pause(); $('pausa').textContent = 'Seguir'; }
  else { tocar(pos); }
};
$('atras').onclick = () => ir(actual - 1, false);
$('revisar').onclick = () => ir(0, false);
document.querySelectorAll('.eleccion button').forEach(b => {
  b.onclick = () => {
    R[tema().n] = { prefiere: b.dataset.v };
    guardar();
    const falta = D.temas.findIndex((t, i) => i > actual && !(R[t.n] && R[t.n].prefiere));
    if (hechos() === D.temas.length) terminar(); else ir(falta < 0 ? D.temas.findIndex(t => !(R[t.n] && R[t.n].prefiere)) : falta, true);
  };
});
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
pintarAvance();
if (hechos() === D.temas.length) terminar();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
