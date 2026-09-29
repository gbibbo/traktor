"""
PURPOSE: Generar la página autónoma "Revisión de playlists" para que Gabriel evalúe, escuchando, si la
         organización en carpetas y playlists tiene sentido antes de importarla en Rekordbox/Traktor.
         La página muestra un mapa 2D (UMAP de Phase 2) con la carpeta y la playlist elegidas, cada
         playlist en su orden (BPM, tonalidad, energía, Vocal, género del tag y carpeta de origen),
         escucha de la secuencia (un fragmento de cada tema) y veredictos por playlist y por tema que
         se guardan en el navegador y se exportan a CSV.
         Se escribe DENTRO de la carpeta de audio del dataset (rutas relativas: el navegador reproduce
         los archivos locales sin servidor). Admite varias organizaciones (una por config de Phase 2):
         con --blind se muestran como A/B en orden aleatorio y la correspondencia se guarda aparte.
CHANGELOG:
  - 2026-09-27: Creación inicial. Veredictos guardados por dataset + config_hash (no por generación).
  - 2026-09-29: --org-name: organizaciones estables (organize.py); marca los temas agregados ('nuevo')
                y los movidos por una semilla ('semilla'). Nombre, versión, alcances, historial y
                semillas aplicadas; 'nombre@N' para una versión anterior; sin argumentos, todas las
                organizaciones con nombre. La página arma semillas y las exporta para
                organize.py link --from-file.
  - 2026-09-29: make_data() reutilizable por la app local (src/v4/ui/review_app.py); fusiones con
                nombre y color en vez de "semillas".
  - 2026-09-29: Mezcla de cada tema (campo "m"): tag Remixer del archivo o el paréntesis del título o
                del nombre que nombra una versión, para distinguir dos versiones del mismo tema.
  - 2026-09-29: Álbum por tema (campo "al", para los filtros de búsqueda), clave l1/l2 de cada playlist
                (orden a mano desde la app) y el historial de reorder / unlink.
"""
import argparse
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_dataset_audio_root  # noqa: E402

TEMPLATE = Path(__file__).with_name("template.html")
PAGE_NAME = "_revision_playlists.html"
_VOCAL = re.compile(r"(?<!\w)Vocal(?!\w)", re.IGNORECASE)
_VERSION = re.compile(r"\b(mix|remix|rmx|edit|dub|version|rework|remaster(ed)?|vip|bootleg|instrumental|variation)\b",
                      re.IGNORECASE)
_BRACKETS = re.compile(r"[(\[]([^()\[\]]+)[)\]]")
_PLAIN_MIX = {"original mix", "original", "original version"}  # la versión por defecto no se muestra
_MIX_CACHE: Dict[tuple, str] = {}


def _letter(n: int) -> str:
    """0→'A', 25→'Z', 26→'AA' (igual que Phase 3/5)."""
    out = []
    while True:
        out.append(chr(ord("A") + n % 26))
        n = n // 26 - 1
        if n < 0:
            return "".join(reversed(out))


def short_folder(folder: str) -> str:
    """Carpeta de origen legible: sin el prefijo '#1 BIBO/PRO/' y como mucho las 2 últimas partes."""
    parts = [p for p in str(folder or "").split("/") if p]
    if parts[:2] == ["#1 BIBO", "PRO"]:
        parts = parts[2:]
    return " / ".join(parts[-2:])


def _num(v, nd: Optional[int] = None):
    if v is None or pd.isna(v):
        return None
    return round(float(v), nd) if nd is not None else int(v)


def tag_mix(path) -> str:
    """Remixer del tag (ID3 TPE4, MP4 REMIXER, Vorbis REMIXER/MIXARTIST), '' si no hay o no se puede leer.
    Cacheado por ruta, fecha y tamaño: la app arma la página en cada visita."""
    try:
        st = Path(path).stat()
    except (OSError, TypeError):
        return ""
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _MIX_CACHE:
        out = ""
        try:
            import mutagen
            from mutagen.id3 import ID3
            try:
                tags = ID3(str(path))  # MP3: solo el bloque de tags, sin recorrer el audio
            except Exception:  # noqa: BLE001 (WAV/AIFF con ID3 en un chunk, MP4, FLAC)
                tags = getattr(mutagen.File(str(path)), "tags", None)
            if tags is not None and hasattr(tags, "getall"):  # ID3 (MP3, WAV, AIFF)
                frame = tags.get("TPE4")
                out = str(frame.text[0]) if frame is not None and frame.text else ""
            elif tags is not None:  # MP4 o Vorbis
                for k in ("----:com.apple.iTunes:REMIXER", "remixer", "mixartist"):
                    vals = tags.get(k)
                    if vals:
                        v = vals[0]
                        out = v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)
                        break
        except Exception:  # noqa: BLE001 (un archivo raro no frena la página)
            out = ""
        _MIX_CACHE[key] = out.strip()
    return _MIX_CACHE[key]


def version_label(tag: Optional[str], title: str, filename: str) -> str:
    """Mezcla que distingue versiones del mismo tema ('Adana Twins Remix Two'): el tag Remixer o, si falta,
    el primer paréntesis o corchete del título o del nombre de archivo que nombra una versión.
    '' si es la original o si el título ya la dice."""
    mix = (tag or "").strip()
    if not mix:
        for text in (title, Path(filename).stem if filename else ""):
            found = [m.strip() for m in _BRACKETS.findall(text or "") if _VERSION.search(m)]
            if found:
                mix = found[0]
                break
    if not mix or mix.lower() in _PLAIN_MIX or mix.lower() in (title or "").lower():
        return ""
    return mix


def build_tracks(catalog: pd.DataFrame, bpm_key: pd.DataFrame, uids: List[str]) -> List[Dict]:
    """Un registro compacto por tema, en el orden de uids."""
    cat = catalog.drop_duplicates("track_uid").set_index("track_uid")
    bk = bpm_key.drop_duplicates("track_uid").set_index("track_uid")
    out = []
    for u in uids:
        r = cat.loc[u]
        b = bk.loc[u] if u in bk.index else None
        comment = r.get("tag_comment")
        title = "" if pd.isna(r.get("title")) else str(r.get("title"))
        tag = r.get("tag_remixer")  # columna del catálogo si existe; si no, el tag del archivo
        if tag is None or pd.isna(tag):
            tag = tag_mix(r["source_path"]) if "source_path" in r.index and not pd.isna(r["source_path"]) else ""
        out.append({
            "u": u[:16],
            "p": str(r["rel_path"]),
            "a": "" if pd.isna(r.get("artist")) else str(r.get("artist")),
            "t": title,
            "m": version_label(tag, title, Path(str(r["rel_path"])).name),
            "g": "" if pd.isna(r.get("tag_genre")) else str(r.get("tag_genre")),
            "al": "" if pd.isna(r.get("tag_album")) else str(r.get("tag_album")),
            "fs": short_folder(r.get("folder", "")),
            "b": _num(b["bpm"], 1) if b is not None else None,
            "k": None if b is None or pd.isna(b["key"]) else str(b["key"]),
            "e": _num(b["energy"]) if b is not None and "energy" in b.index else None,
            "v": bool(isinstance(comment, str) and _VOCAL.search(comment)),
        })
    return out


def load_org(artifacts: Path, config_hash: str) -> Dict:
    """Carpetas (L1) y playlists (L2) ordenadas de una corrida de Phases 2-4, más el UMAP."""
    cdir = artifacts / "clustering"
    ordered = pd.read_parquet(cdir / f"ordered_{config_hash}.parquet")
    names = json.loads((cdir / f"names_{config_hash}.json").read_text(encoding="utf-8"))
    cfg = json.loads((cdir / f"config_{config_hash}.json").read_text(encoding="utf-8"))
    if not {"umap_x", "umap_y"} <= set(ordered.columns) or not ordered[["umap_x", "umap_y"]].abs().to_numpy().any():
        raise ValueError(f"ordered_{config_hash} no tiene UMAP: correr phase2_cluster.py sin --skip-umap")
    return {"hash": config_hash, "rep": cfg.get("rep", "mert"), "ordered": ordered, "names": names}


def _when(created: str) -> str:
    try:
        d = dt.datetime.fromisoformat(created)
        return d.strftime("%d/%m %H:%M")
    except (TypeError, ValueError):
        return ""


def _n(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def history_lines(meta: Dict) -> List[str]:
    """Cambios de la organización en palabras simples, con fecha, para la barra lateral."""
    out = []
    by_v = {h["version"]: h for h in meta.get("history", [])}
    lineage, cur = set(), meta.get("current_version")
    while cur is not None and cur in by_v and cur not in lineage:  # versiones vigentes: la actual y sus padres
        lineage.add(cur)
        cur = by_v[cur].get("parent")
    for h in meta.get("history", []):
        v, a = h["version"], h["action"]
        if a == "import":
            txt = "organización inicial"
        elif a == "build":
            txt = f"reorganizada desde cero ({h['n_tracks']} temas, {h['n_playlists']} playlists)"
        elif a == "add":
            scope = h.get("scope")
            scope = scope if isinstance(scope, str) else ", ".join(x or "toda la biblioteca" for x in scope or [])
            parts = [f"{h.get('add', 0)} a playlists que ya existían"]
            if h.get("add-new"):
                parts.append(f"{h['add-new']} en playlists nuevas")
            if h.get("add-far"):
                parts.append(f"{h['add-far']} sin una playlist parecida")
            txt = f"se agregaron {_n(h.get('added', 0), 'tema', 'temas')} de «{scope}» ({', '.join(parts)})"
        elif a == "link":
            groups = h.get("groups") or ([h["group"]] if h.get("group") else [])
            moved = len(h.get("moved", []))
            txt = f"{_n(len(groups), 'fusión aplicada', 'fusiones aplicadas')} ({_n(moved, 'tema movido', 'temas movidos')})"
        elif a == "reorder":
            txt = f"orden cambiado a mano en «{h.get('playlist_name') or 'una playlist'}»"
        elif a == "unlink":
            txt = f"se eliminó la fusión «{h.get('fusion', '')}» (sus temas no se movieron)"
        elif a == "link-rebuild":
            txt = f"reorganizada desde cero respetando las fusiones ({h['n_playlists']} playlists)"
        else:
            txt = a
        when = _when(h.get("created", ""))
        undone = lineage and v not in lineage
        out.append(f"Versión {v}{' · ' + when if when else ''} · {txt}{' (deshecho)' if undone else ''}")
    return out


def load_named_org(artifacts: Path, spec: str) -> Dict:
    """Organización estable (orgs/<nombre>/, 'nombre@N' = versión N) con el formato de load_org más
    su meta, semillas aplicadas y el config_hash de origen (para trasladar veredictos)."""
    from src.v4.pipeline.organize import OrgStore, load_for_export, org_scopes
    name, _, ver = spec.partition("@")
    df, names, meta, v = load_for_export(artifacts, name, int(ver.lstrip("v")) if ver else None)
    current = v == meta["current_version"]
    return {"hash": f"org:{name}" if current else f"org:{name}@v{v}", "rep": f"{name} v{v}", "ordered": df,
            "names": names, "cli": name,
            "meta": {"name": name, "version": v, "current": current, "scopes": org_scopes(meta),
                     "history": history_lines(meta)},
            "fusions": OrgStore(artifacts, name).fusions(v),
            "legacy": [meta["source_hash"]] if current and meta.get("source_hash") else []}


FLAG_TEXT = {"add": "nuevo", "add-far": "nuevo", "add-new": "nuevo", "link": "semilla"}


def org_payload(org: Dict, uid_index: Dict[str, int], label: str, n_suggested: int = 8) -> Dict:
    """id = config_hash: los veredictos guardados en el navegador siguen a la organización aunque se
    regenere la página o cambie la letra A/B."""
    org_id = org["hash"]
    df = org["ordered"].sort_values(["label_l1", "label_l2", "position"])
    folders = []
    for l1, g1 in df.groupby("label_l1", sort=True):
        letter = "Ruido" if l1 < 0 else _letter(int(l1))
        l1_name = org["names"].get(f"l1_{l1}", f"Group {letter}")
        fname = f"{letter} · {l1_name}" if not l1_name.startswith("Group") else letter
        pls = []
        for l2, g2 in g1.groupby("label_l2", sort=True):
            pname = org["names"].get(f"l1_{l1}_l2_{l2}", f"{letter}{int(l2) + 1}" if l2 >= 0 else f"{letter} varios")
            pls.append({"id": f"{org_id}:{l1}:{l2}", "key": [int(l1), int(l2)], "name": pname,
                        "tracks": [uid_index[u] for u in g2["track_uid"]], "suggested": False})
        folders.append({"id": f"{org_id}:{l1}", "name": fname, "playlists": pls})
    # Sugeridas: la playlist más grande de cada una de las n carpetas más grandes
    by_size = sorted(folders, key=lambda f: -sum(len(p["tracks"]) for p in f["playlists"]))
    for f in by_size[:n_suggested]:
        max(f["playlists"], key=lambda p: len(p["tracks"]))["suggested"] = True
    xy_df = df.set_index("track_uid")[["umap_x", "umap_y"]]
    order = list(xy_df.index)
    flags = {}
    if "origin" in df.columns:
        flags = {str(uid_index[u]): FLAG_TEXT[o] for u, o in zip(df["track_uid"], df["origin"]) if o in FLAG_TEXT}
    extra = {}
    if org.get("cli"):
        extra = {"cli": org["cli"], "meta": org["meta"], "legacy": org.get("legacy", []),
                 "fusions": [{"name": f["name"], "color": f["color"],
                              "tracks": [uid_index[u] for u in f["tracks"] if u in uid_index]}
                             for f in org.get("fusions", [])]}
    return {"id": org_id, "label": label, "folders": folders, "flags": flags, **extra,
            "trackIdx": [uid_index[u] for u in order],
            "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in xy_df.to_numpy()]}


def render(data: Dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "/*__DATA__*/null" in html
    return html.replace("/*__DATA__*/null", payload)


def make_data(artifacts: Path, dataset: str, specs: List[str], hashes: List[str], blind: bool = False,
              seed: Optional[int] = None) -> Dict:
    """Datos de la página: organizaciones con nombre (specs) y corridas por hash (hashes)."""
    orgs = [load_named_org(artifacts, n) for n in specs]
    orgs += [load_org(artifacts, h) for h in hashes]
    catalog = pd.read_parquet(artifacts / "catalog.parquet")
    bpm_key = pd.read_parquet(artifacts / "features" / "bpm_key.parquet")
    uids = list(dict.fromkeys(u for o in orgs for u in o["ordered"]["track_uid"]))
    uid_index = {u: i for i, u in enumerate(uids)}
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M")

    if blind and len(orgs) > 1:
        rng = random.Random(seed)
        rng.shuffle(orgs)
        labels = [f"Organización {_letter(i)}" for i in range(len(orgs))]
        key = {lab: {"config_hash": o["hash"], "rep": o["rep"]} for lab, o in zip(labels, orgs)}
        key_path = artifacts / "evaluation" / f"review_key_{run_id}.json"
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
        print(f"[INFO] A ciegas: correspondencia en {key_path} (no está en la página)")
    else:
        labels = [(o["rep"] if o["hash"].startswith("org:") else f"{o['rep']} ({o['hash']})") if len(orgs) > 1
                  else (o["rep"] if o["hash"].startswith("org:") else "Organización actual") for o in orgs]
    return {"run_id": run_id, "dataset": dataset, "app": None,
            "tracks": build_tracks(catalog, bpm_key, uids),
            "orgs": [org_payload(o, uid_index, lab) for o, lab in zip(orgs, labels)]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Página HTML para revisar la organización escuchando")
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--config", default=None)
    parser.add_argument("--org", action="append", default=[],
                        help="config_hash de Phase 2 (repetible). Default: el ordered_*.parquet más reciente.")
    parser.add_argument("--org-name", action="append", default=[],
                        help="Organización estable de organize.py; 'nombre@N' = versión N (repetible; se combina con --org)")
    parser.add_argument("--blind", action="store_true", help="Mostrar las organizaciones como A/B en orden aleatorio")
    parser.add_argument("--seed", type=int, default=None, help="Semilla del orden a ciegas")
    parser.add_argument("--out", default=None, help=f"Default: <carpeta de audio>/{PAGE_NAME}")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    specs = args.org_name
    if not specs and not args.org:  # sin argumentos: todas las organizaciones con nombre
        specs = sorted(p.parent.name for p in (artifacts / "orgs").glob("*/org.json"))
    hashes = args.org or ([] if specs else [sorted((artifacts / "clustering").glob("ordered_*.parquet"),
                                                   key=lambda p: p.stat().st_mtime)[-1].stem.replace("ordered_", "")])
    data = make_data(artifacts, args.dataset_name, specs, hashes, args.blind, args.seed)
    uids = data["tracks"]
    orgs = data["orgs"]
    out = Path(args.out) if args.out else resolve_dataset_audio_root(args.dataset_name, config) / PAGE_NAME
    out.write_text(render(data), encoding="utf-8")
    n_pl = sum(len(f["playlists"]) for o in data["orgs"] for f in o["folders"])
    print(f"[INFO] {len(uids)} temas, {len(orgs)} organización(es), {n_pl} playlists → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
