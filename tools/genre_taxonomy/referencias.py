"""
PURPOSE: Baja las taxonomías de referencia para construir la taxonomía propia de géneros de Beatport
         (DECISIONS 2026-11-06, punto 10), a artifacts/v4/genre_taxonomy/inputs/:
           - musicbrainz_generos_todos.json: los géneros de MusicBrainz (nombre, MBID);
           - musicbrainz_electronica.json: el árbol de la música electrónica de MusicBrainz, bajando por
             los subgéneros desde "electronic", con todas las relaciones de cada género tal como las
             muestra su página (subgenre of, subgenres, fusion of, has fusion genres, influenced by,
             influenced genres...). La API (ws/2) no devuelve las relaciones entre géneros: se leen de
             musicbrainz.org/genre/<mbid>;
           - wikidata_electronica.json: los géneros que en Wikidata son subclase (P279) de la música
             electrónica (Q9778), con sus padres dentro de ese árbol y su MBID (P8052) si lo tienen.
         MusicBrainz pide como mucho una consulta por segundo e identificarse: pausa de 1.2 s y
         reintentos con espera creciente ante 503. Solo biblioteca estándar.
CHANGELOG:
  - 2026-11-06: Creación.
  - 2026-11-06: Las relaciones de MusicBrainz salen de la página de cada género (la API no las da).
"""
import argparse
import html as htmllib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "artifacts" / "v4" / "genre_taxonomy" / "inputs"
UA = "TRAKTOR-ML/0.1 (https://github.com/gbibbo/traktor)"
MB = "https://musicbrainz.org/ws/2"
WIKIDATA = "https://query.wikidata.org/sparql"
SPARQL = """
SELECT ?g ?gLabel ?p ?pLabel ?mb WHERE {
  ?g wdt:P279* wd:Q9778 .
  OPTIONAL { ?g wdt:P279 ?p . ?p wdt:P279* wd:Q9778 . }
  OPTIONAL { ?g wdt:P8052 ?mb . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}"""


_ROW = re.compile(r"<th[^>]*>\s*([^<:]+?):?\s*</th>\s*<td[^>]*>(.*?)</td>", re.S)
_LINK = re.compile(r'href="/genre/([0-9a-f-]{36})"[^>]*>(?:<bdi>)?([^<]+)')


def genre_relations(page: str) -> dict:
    """{"subgenre of": [(mbid, nombre)], "subgenres": [...], ...} de la sección Relationships."""
    i = page.find("Relationships")
    j = page.find("External links", i)
    seg = page[i:j if j > 0 else None] if i >= 0 else ""
    out = {}
    for label, cell in _ROW.findall(seg):
        links = [(m, htmllib.unescape(n).strip()) for m, n in _LINK.findall(cell)]
        if links:
            out[htmllib.unescape(label).strip().lower()] = links
    return out


class Fetcher:
    def __init__(self, pause: float = 1.2):
        self.pause, self.last, self.n = pause, 0.0, 0

    def text(self, url: str) -> str:
        return self.json(url, accept="text/html", raw=True)

    def json(self, url: str, accept: str = "application/json", raw: bool = False):
        wait = 5.0
        for _ in range(8):
            delay = self.last + self.pause - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()
            self.n += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
                body = urllib.request.urlopen(req, timeout=60).read()
                return body.decode("utf-8", "replace") if raw else json.loads(body)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(wait)
            wait *= 2
        raise RuntimeError(f"Sin respuesta tras varios intentos: {url}")


def musicbrainz(f: Fetcher, roots, max_nodes: int) -> None:
    allg, off = [], 0
    while True:
        d = f.json(f"{MB}/genre/all?limit=100&offset={off}&fmt=json")
        allg += [{"id": g["id"], "name": g["name"], "disambiguation": g.get("disambiguation", "")}
                 for g in d["genres"]]
        off += 100
        if off >= d["genre-count"]:
            break
    (OUT / "musicbrainz_generos_todos.json").write_text(json.dumps(allg, ensure_ascii=False), encoding="utf-8")
    by_name = {g["name"]: g["id"] for g in allg}
    print(f"[musicbrainz] {len(allg)} géneros", flush=True)
    queue = deque(by_name[r] for r in roots if r in by_name)
    seen, nodes = set(queue), {}
    name_of = {g["id"]: g["name"] for g in allg}
    while queue and len(nodes) < max_nodes:
        gid = queue.popleft()
        rels = genre_relations(f.text(f"https://musicbrainz.org/genre/{gid}"))
        nodes[gid] = {"id": gid, "name": name_of.get(gid, gid),
                      "relaciones": {k: [{"id": m, "genero": n} for m, n in v] for k, v in rels.items()}}
        for m, _n in rels.get("subgenres", []):     # se baja por los subgéneros; se anota todo
            if m not in seen:
                seen.add(m)
                queue.append(m)
        if len(nodes) % 50 == 0:
            print(f"  {len(nodes)} géneros recorridos, {len(queue)} en cola", flush=True)
    (OUT / "musicbrainz_electronica.json").write_text(
        json.dumps({"raices": list(roots), "fuente": "musicbrainz.org/genre/<mbid> (sección Relationships)",
                    "generos": list(nodes.values())}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[musicbrainz] árbol electrónico: {len(nodes)} géneros ({f.n} consultas)", flush=True)


def wikidata(f: Fetcher) -> None:
    d = f.json(WIKIDATA + "?" + urllib.parse.urlencode({"query": SPARQL, "format": "json"}),
               accept="application/sparql-results+json")
    items = {}
    for b in d["results"]["bindings"]:
        q = b["g"]["value"].rsplit("/", 1)[-1]
        it = items.setdefault(q, {"qid": q, "nombre": b.get("gLabel", {}).get("value"), "padres": [], "mbid": None})
        if "p" in b:
            p = {"qid": b["p"]["value"].rsplit("/", 1)[-1], "nombre": b.get("pLabel", {}).get("value")}
            if p not in it["padres"]:
                it["padres"].append(p)
        if "mb" in b:
            it["mbid"] = b["mb"]["value"]
    (OUT / "wikidata_electronica.json").write_text(
        json.dumps({"raiz": "Q9778 (electronic music)", "generos": list(items.values())}, ensure_ascii=False,
                   indent=1), encoding="utf-8")
    print(f"[wikidata] {len(items)} géneros bajo la música electrónica", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Taxonomías de referencia (MusicBrainz, Wikidata)")
    ap.add_argument("--solo", choices=("musicbrainz", "wikidata"), default=None)
    ap.add_argument("--raiz", action="append", default=None, help="Género raíz de MusicBrainz (por defecto electronic)")
    ap.add_argument("--max", type=int, default=1500, help="Tope de géneros del árbol de MusicBrainz")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = Fetcher()
    if args.solo in (None, "wikidata"):
        wikidata(f)
    if args.solo in (None, "musicbrainz"):
        musicbrainz(f, args.raiz or ["electronic"], args.max)
    return 0


if __name__ == "__main__":
    sys.exit(main())
