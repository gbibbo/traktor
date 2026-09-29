"""
PURPOSE: Organizaciones estables de la biblioteca (plan docs/plans/20260929_organizaciones_estables.md).
         Una organización tiene nombre, versiones y los modelos (PCA, UMAP, estadísticas de BPM) con
         que se construyó, en artifacts/v4/datasets/<dataset>/orgs/<nombre>/. Permite:
           - ingest: catálogo, BPM/tonalidad, embeddings (y etiqueta Vocal) solo de una subcarpeta;
           - import: convertir una corrida de Phases 2-4 (config_hash) en organización v1 sin cambios;
           - build: organizar desde cero una subcarpeta o toda la biblioteca, respetando semillas;
           - add: agregar música nueva con todo lo existente congelado (nadie cambia de playlist, de
             orden relativo ni de lugar en el mapa);
           - link: exigir que dos o más temas vayan en la misma playlist, moviendo solo ese grupo
             (congelado) o rehaciendo todo con las semillas (--rebuild);
           - show: resumen de versiones, playlists y semillas;
           - reorder / remove_fusion (desde la app): orden a mano de una playlist y quitar una fusión;
           - remove_tracks: sacar las copias de temas repetidos (dedupe.py apply).
         Export y página de revisión: phase5_export.py / build_review_page.py con --org-name.
CHANGELOG:
  - 2026-09-29: Creación inicial.
  - 2026-09-29: link --from-file: aplica las semillas exportadas desde la página de revisión (varios
                grupos en una sola versión).
  - 2026-09-29: Fusiones (antes "semillas") con nombre y color, guardadas dentro de cada versión
                (v<N>/constraints.json): deshacer una versión deshace sus fusiones. Borradores de
                fusiones (fusion_drafts.json) para la app, set_current/undo, y parent en el historial.
  - 2026-09-29: reorder (orden a mano de una playlist) y remove_fusion (quitar una fusión aplicada sin
                mover temas), cada uno como versión nueva, para la app.
  - 2026-09-29: Temas repetidos (src/v4/common/duplicates.py): Library excluye las copias descartadas
                y remove_tracks las saca de la organización (versión 'dedupe').
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.catalog import load_catalog, scope_prefix  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.duplicates import dropped  # noqa: E402
from src.v4.common.embedding_utils import load_track_embeddings  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402
from src.v4.pipeline.phase2_cluster import _l2_normalize, _ward_cluster, ward_two_level  # noqa: E402
from src.v4.pipeline.phase3_name import _cluster_to_letter, _top_genres  # noqa: E402
from src.v4.pipeline.phase4_order import essentia_to_camelot, insert_position, order_cluster_tracks  # noqa: E402

DEFAULT_PARAMS = {
    "rep": "clap_full", "bpm_weight": 0.3, "pca_dim": 50, "n_l1": None, "l2_target_size": 35,
    "ordering_weights": {"embedding": 0.5, "bpm": 0.3, "key": 0.2, "energy": 0.0},
    "radius_pct": 90, "min_new_playlist": 12,
}


# ---------------------------------------------------------------------------
# Almacenamiento
# ---------------------------------------------------------------------------

class OrgStore:
    """Carpeta de una organización: org.json, models_v<N>/, v<N>/ (con constraints.json: las fusiones
    de esa versión) y fusion_drafts.json (fusiones en armado, aún no aplicadas)."""

    def __init__(self, artifacts: Path, name: str):
        self.artifacts = Path(artifacts)
        self.name = name
        self.dir = self.artifacts / "orgs" / name

    def exists(self) -> bool:
        return (self.dir / "org.json").exists()

    def meta(self) -> Dict:
        return json.loads((self.dir / "org.json").read_text(encoding="utf-8"))

    def save_meta(self, meta: Dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "org.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def fusions(self, version: Optional[int] = None) -> List[Dict]:
        """Fusiones aplicadas en la versión (actual por defecto): [{name, color, tracks}]."""
        if not self.exists():
            return []
        v = version or self.meta()["current_version"]
        p = self.dir / f"v{v}" / "constraints.json"
        if not p.exists():
            p = self.dir / "constraints.json"  # formato anterior (una sola lista para todas las versiones)
        if not p.exists():
            return []
        return normalize_fusions(json.loads(p.read_text(encoding="utf-8")))

    def constraints(self, version: Optional[int] = None) -> List[List[str]]:
        return [f["tracks"] for f in self.fusions(version)]

    def drafts(self) -> List[Dict]:
        p = self.dir / "fusion_drafts.json"
        return normalize_fusions(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else []

    def save_drafts(self, drafts: List[Dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "fusion_drafts.json").write_text(json.dumps({"fusions": drafts}, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")

    def set_current(self, version: int) -> None:
        meta = self.meta()
        entry = next(h for h in meta["history"] if h["version"] == version)
        meta["current_version"], meta["model_version"] = version, entry["model_version"]
        if "org_scopes" in entry:  # deshacer un 'add' también saca su carpeta de los alcances
            meta["scope"], meta["added_scopes"] = entry["org_scopes"][0], entry["org_scopes"][1:]
        self.save_meta(meta)

    def undo(self) -> Optional[int]:
        """Vuelve a la versión de la que salió la actual (ninguna versión se borra)."""
        meta = self.meta()
        cur = next(h for h in meta["history"] if h["version"] == meta["current_version"])
        parent = cur.get("parent")
        if parent is None:
            earlier = [h["version"] for h in meta["history"] if h["version"] < cur["version"]]
            parent = max(earlier) if earlier else None
        if parent is None:
            return None
        self.set_current(parent)
        return parent

    def load(self, version: Optional[int] = None) -> Tuple[pd.DataFrame, Dict, int]:
        meta = self.meta()
        v = version or meta["current_version"]
        vd = self.dir / f"v{v}"
        return (pd.read_parquet(vd / "assignments.parquet"),
                json.loads((vd / "names.json").read_text(encoding="utf-8")), v)

    def models(self, model_version: int) -> Dict:
        md = self.dir / f"models_v{model_version}"
        return {"pca": joblib.load(md / "pca.joblib") if (md / "pca.joblib").exists() else None,
                "umap": joblib.load(md / "umap.joblib"),
                "bpm_stats": json.loads((md / "bpm_stats.json").read_text(encoding="utf-8"))}

    def save_models(self, model_version: int, pca, reducer, bpm_stats: Dict) -> None:
        md = self.dir / f"models_v{model_version}"
        md.mkdir(parents=True, exist_ok=True)
        if pca is not None:
            joblib.dump(pca, md / "pca.joblib")
        joblib.dump(reducer, md / "umap.joblib")
        (md / "bpm_stats.json").write_text(json.dumps(bpm_stats, indent=2), encoding="utf-8")

    def commit(self, meta: Dict, assign: pd.DataFrame, names: Dict, action: str, detail: Dict,
               model_version: Optional[int] = None, fusions: Optional[List[Dict]] = None) -> int:
        """Escribe una versión nueva y la deja como actual. fusions=None conserva las de la versión actual."""
        parent = meta.get("current_version")
        if fusions is None:
            fusions = self.fusions(parent) if parent else []
        v = max([h["version"] for h in meta.get("history", [])], default=0) + 1
        vd = self.dir / f"v{v}"
        vd.mkdir(parents=True, exist_ok=True)
        (vd / "constraints.json").write_text(json.dumps({"fusions": fusions}, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
        assign = assign.sort_values(["l1", "l2", "position"]).reset_index(drop=True)
        assign.to_parquet(vd / "assignments.parquet", index=False)
        (vd / "names.json").write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")
        mv = model_version if model_version is not None else meta.get("model_version", v)
        meta.setdefault("history", []).append({
            "version": v, "created": dt.datetime.now().isoformat(timespec="seconds"), "action": action,
            "model_version": mv, "parent": parent,
            "org_scopes": [meta.get("scope", "")] + list(meta.get("added_scopes", [])), "n_tracks": int(len(assign)),
            "n_playlists": int(assign.groupby(["l1", "l2"]).ngroups), **detail})
        meta["current_version"], meta["model_version"] = v, mv
        self.save_meta(meta)
        return v


# ---------------------------------------------------------------------------
# Datos y espacio de features
# ---------------------------------------------------------------------------

class Library:
    """Catálogo, BPM/tonalidad/energía y embeddings de la representación, alineados por track_uid.
    Las copias descartadas de temas repetidos (duplicate_decisions.json) no entran a in_scope."""

    def __init__(self, artifacts: Path, rep: str):
        self.catalog = pd.read_parquet(artifacts / "catalog.parquet").drop_duplicates("track_uid").set_index("track_uid")
        self.bk = pd.read_parquet(artifacts / "features" / "bpm_key.parquet").drop_duplicates("track_uid").set_index("track_uid")
        uids, M = load_track_embeddings(artifacts, rep=rep)
        self.rep_index = {u: i for i, u in enumerate(uids)}
        self.M = M
        self.rep_uids = uids
        self.excluded = set(dropped(artifacts))

    def in_scope(self, scopes) -> List[str]:
        """Temas con embedding y en el catálogo bajo alguno de los prefijos ('' o lista vacía = todos)."""
        scopes = [scopes] if isinstance(scopes, str) else list(scopes)
        rel = self.catalog["rel_path"]
        if not scopes or "" in scopes:
            ok = set(rel.index)
        else:
            prefixes = tuple(scope_prefix(x) for x in scopes)
            ok = set(rel.index[rel.str.startswith(prefixes)])
        return [u for u in self.rep_uids if u in ok and u not in self.excluded]

    def emb(self, uids: List[str]) -> np.ndarray:
        return self.M[[self.rep_index[u] for u in uids]]

    def bpm_raw(self, uids: List[str]) -> pd.Series:
        return pd.to_numeric(self.bk.reindex(uids)["bpm"], errors="coerce")

    def order_arrays(self, uids: List[str]):
        """Arrays para el orden (Phase 4): embeddings normalizados, BPM (128 si falta), Camelot, energía."""
        bk = self.bk.reindex(uids)
        bpm = pd.to_numeric(bk["bpm"], errors="coerce").fillna(128.0).to_numpy(float)
        keys = [essentia_to_camelot(str(k)) if not pd.isna(k) else "?" for k in bk["key"]]
        energy = pd.to_numeric(bk["energy"], errors="coerce").to_numpy(float) if "energy" in bk.columns else None
        return _l2_normalize(self.emb(uids)), bpm, keys, energy


def bpm_stats(bpm: pd.Series) -> Dict:
    """Mismas estadísticas que phase2._append_bpm (faltantes = mediana; desvío poblacional)."""
    med = float(bpm.median())
    filled = bpm.fillna(med).to_numpy(np.float64)
    return {"median": med, "mean": float(filled.mean()), "std": float(filled.std() or 1.0)}


def features(lib: Library, uids: List[str], weight: float, stats: Dict) -> np.ndarray:
    """Embedding normalizado + weight * BPM estandarizado con estadísticas fijas (congeladas)."""
    X = _l2_normalize(lib.emb(uids))
    if weight <= 0:
        return X.astype(np.float32)
    bpm = lib.bpm_raw(uids).fillna(stats["median"]).to_numpy(np.float64)
    z = (bpm - stats["mean"]) / stats["std"]
    return np.hstack([X, (weight * z)[:, None]]).astype(np.float32)


def fit_space(F: np.ndarray, pca_dim: int):
    """PCA (la del L1 de Phase 2, semilla 0) y UMAP 2D (el de Phase 2, semilla 42), guardables."""
    from sklearn.decomposition import PCA
    import umap
    pca = None
    if 0 < pca_dim < F.shape[1]:
        pca = PCA(n_components=min(pca_dim, F.shape[0] - 1), whiten=False, random_state=0).fit(F)
    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(F) - 1), min_dist=0.1, random_state=42,
                        metric="cosine").fit(F)
    return pca, reducer, reducer.embedding_.astype(np.float32)


def project(pca, F: np.ndarray) -> np.ndarray:
    return (pca.transform(F) if pca is not None else F).astype(np.float32)


FUSION_COLORS = 8  # la interfaz asigna violeta, amarillo, verde, … por índice


def normalize_fusions(data) -> List[Dict]:
    """{'fusions': [...]}, {'groups': [[...]]} o una lista -> [{name, color, tracks}]."""
    items = data.get("fusions", data.get("groups", [])) if isinstance(data, dict) else data
    out = []
    for i, it in enumerate(items):
        if isinstance(it, dict):
            out.append({"name": it.get("name") or f"Fusión {i + 1}", "color": int(it.get("color", i)) % FUSION_COLORS,
                        "tracks": list(it["tracks"])})
        else:
            out.append({"name": f"Fusión {i + 1}", "color": i % FUSION_COLORS, "tracks": list(it)})
    return out


def merge_fusions(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Une fusiones que comparten temas (transitivo). La fusionada toma el nombre y color de la
    primera que aparece; las que no se tocan quedan igual."""
    groups = components([f["tracks"] for f in existing + new] + [[t, t] for f in existing + new for t in f["tracks"]])
    out = []
    for g in groups:
        first = next(f for f in existing + new if set(f["tracks"]) & set(g))
        out.append({"name": first["name"], "color": first["color"], "tracks": g})
    return out


def components(groups: List[List[str]]) -> List[List[str]]:
    """Une grupos que comparten temas (union-find); cada semilla transitiva queda en un solo grupo."""
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in groups:
        for u in g[1:]:
            parent[find(u)] = find(g[0])
        find(g[0])
    comps: Dict[str, List[str]] = {}
    for u in parent:
        comps.setdefault(find(u), []).append(u)
    return [sorted(c) for c in comps.values() if len(c) > 1]


# ---------------------------------------------------------------------------
# Nombres y orden
# ---------------------------------------------------------------------------

def name_playlists(assign: pd.DataFrame, catalog: pd.DataFrame) -> Dict[str, str]:
    """Mismo esquema que Phase 3: 'l1_<a>' y 'l1_<a>_l2_<b>' con votación de géneros de los tags."""
    cat = catalog.reset_index()
    names: Dict[str, str] = {"l1_-1": "Noise"}
    for l1, g1 in assign.groupby("l1"):
        letter = _cluster_to_letter(int(l1))
        genre = _top_genres(g1["track_uid"].tolist(), cat)
        l1_name = genre or f"Group {letter}"
        names[f"l1_{l1}"] = l1_name
        for l2, g2 in g1.groupby("l2"):
            g_l2 = _top_genres(g2["track_uid"].tolist(), cat)
            base = f"{letter}{int(l2) + 1}"
            names[f"l1_{l1}_l2_{l2}"] = f"{base} {g_l2}" if g_l2 and g_l2 != l1_name else base
    return names


def order_playlist(lib: Library, uids: List[str], weights: Dict) -> List[str]:
    E, bpm, keys, energy = lib.order_arrays(uids)
    order = order_cluster_tracks(list(range(len(uids))), E, bpm, keys, weights, energy=energy)
    return [uids[i] for i in order]


def insert_into(lib: Library, order: List[str], new_uid: str, weights: Dict) -> List[str]:
    """Inserta new_uid en la posición de menor costo; el orden relativo de los demás no cambia."""
    uids = order + [new_uid]
    E, bpm, keys, energy = lib.order_arrays(uids)
    pos = insert_position(list(range(len(order))), len(order), E, bpm, keys, weights, energy)
    return order[:pos] + [new_uid] + order[pos:]


def _renumber(assign: pd.DataFrame, playlists: Dict[Tuple[int, int], List[str]]) -> pd.DataFrame:
    """Aplica el orden de cada playlist a la columna position (y l1/l2 de los que se movieron)."""
    rows = {}
    for (l1, l2), order in playlists.items():
        for pos, u in enumerate(order):
            rows[u] = (l1, l2, pos)
    assign = assign.copy()
    assign["l1"] = assign["track_uid"].map(lambda u: rows[u][0])
    assign["l2"] = assign["track_uid"].map(lambda u: rows[u][1])
    assign["position"] = assign["track_uid"].map(lambda u: rows[u][2])
    return assign


def _playlists(assign: pd.DataFrame) -> Dict[Tuple[int, int], List[str]]:
    return {(int(a), int(b)): g.sort_values("position")["track_uid"].tolist()
            for (a, b), g in assign.groupby(["l1", "l2"])}


# ---------------------------------------------------------------------------
# Operaciones
# ---------------------------------------------------------------------------

def org_scopes(meta: Dict) -> List[str]:
    """Alcances de la organización: el de origen más los agregados con add ('' = toda la biblioteca)."""
    return [meta.get("scope", "")] + meta.get("added_scopes", [])


def build(store: OrgStore, lib: Library, params: Dict, scopes: List[str], action: str = "build",
          fusions: Optional[List[Dict]] = None) -> int:
    """Organización desde cero sobre los alcances, respetando las fusiones (las actuales si no se pasan)."""
    uids = lib.in_scope(scopes)
    if len(uids) < 2:
        raise ValueError(f"Alcance {scopes}: {len(uids)} temas con embedding; correr 'ingest' primero")
    stats = bpm_stats(lib.bpm_raw(uids))
    F = features(lib, uids, params["bpm_weight"], stats)
    pos = {u: i for i, u in enumerate(uids)}
    fusions = store.fusions() if fusions is None else fusions
    groups = [[pos[u] for u in c if u in pos] for c in components([f["tracks"] for f in fusions])]
    groups = [np.array(g) for g in groups if len(g) > 1]
    n_l1 = params["n_l1"] or max(1, int(round(len(uids) / 150)))
    l1, l2, _ = ward_two_level(F, F, n_l1, params["l2_target_size"], params["pca_dim"], groups)
    pca, reducer, xy = fit_space(F, params["pca_dim"])
    assign = pd.DataFrame({"track_uid": uids, "l1": l1, "l2": l2, "position": 0,
                           "x": xy[:, 0], "y": xy[:, 1], "origin": "build"})
    playlists = {k: order_playlist(lib, v, params["ordering_weights"]) for k, v in _playlists(assign).items()}
    assign = _renumber(assign, playlists)
    meta = store.meta() if store.exists() else {"name": store.name, "history": []}
    meta.update({"scope": scopes[0], "added_scopes": [x for x in scopes[1:] if x != scopes[0]], "params": params})
    next_v = max([h["version"] for h in meta.get("history", [])], default=0) + 1
    store.save_models(next_v, pca, reducer, stats)
    names = name_playlists(assign, lib.catalog)
    return store.commit(meta, assign, names, action,
                        {"scopes": scopes, "n_l1": n_l1, "n_seed_groups": len(groups)}, model_version=next_v,
                        fusions=fusions)


def import_hash(store: OrgStore, lib: Library, artifacts: Path, config_hash: str) -> int:
    """v1 = una corrida de Phases 2-4 tal cual; reajusta PCA/UMAP y verifica que el mapa coincida."""
    cdir = artifacts / "clustering"
    cfg = json.loads((cdir / f"config_{config_hash}.json").read_text(encoding="utf-8"))
    ordered = pd.read_parquet(cdir / f"ordered_{config_hash}.parquet")
    names = json.loads((cdir / f"names_{config_hash}.json").read_text(encoding="utf-8"))
    if cfg.get("method") != "ward" or not cfg.get("rep"):
        raise ValueError("Solo se importan corridas con --method ward y --rep")
    params = dict(DEFAULT_PARAMS, rep=cfg["rep"], bpm_weight=cfg.get("bpm_weight", 0.0), pca_dim=cfg["pca_dim"],
                  n_l1=cfg["n_l1"], l2_target_size=cfg["l2_target_size"])
    # Mismo orden de filas que Phase 2 (el de la representación): UMAP depende del orden
    present = set(ordered["track_uid"])
    uids = [u for u in lib.rep_uids if u in present]
    ordered = ordered.set_index("track_uid").loc[uids].reset_index()
    # Phase 2 estandariza el BPM sobre todos los temas de la representación
    stats = bpm_stats(lib.bpm_raw(lib.rep_uids))
    F = features(lib, uids, params["bpm_weight"], stats)
    pca, reducer, xy = fit_space(F, params["pca_dim"])
    ref = ordered[["umap_x", "umap_y"]].to_numpy(np.float32)
    diff = float(np.abs(xy - ref).max())
    if diff > 1e-3:
        raise ValueError(f"El UMAP reajustado no coincide con el guardado (máx. dif. {diff:.4f})")
    assign = pd.DataFrame({"track_uid": uids, "l1": ordered["label_l1"].astype(int), "l2": ordered["label_l2"].astype(int),
                           "position": ordered["position"].astype(int), "x": ref[:, 0], "y": ref[:, 1], "origin": "build"})
    meta = {"name": store.name, "scope": "", "params": params, "source_hash": config_hash, "history": []}
    store.save_models(1, pca, reducer, stats)
    return store.commit(meta, assign, names, "import", {"source_hash": config_hash, "umap_max_diff": diff}, model_version=1)


def add(store: OrgStore, lib: Library, scope: Optional[str] = None) -> Tuple[int, Dict[str, int]]:
    """Agrega los temas nuevos del alcance con todo lo existente congelado."""
    meta = store.meta()
    params = meta["params"]
    assign, names, _ = store.load()
    models = store.models(meta["model_version"])
    known = set(assign["track_uid"])
    cand = [u for u in lib.in_scope(scope if scope else org_scopes(meta)) if u not in known]
    if not cand:
        return meta["current_version"], {"added": 0}
    if scope and not any(x == "" or scope_prefix(scope).startswith(scope_prefix(x)) for x in org_scopes(meta)):
        meta.setdefault("added_scopes", []).append(scope)  # un build posterior también lo cubre
    w = params["bpm_weight"]
    old = assign["track_uid"].tolist()
    X_old = project(models["pca"], features(lib, old, w, models["bpm_stats"]))
    F_new = features(lib, cand, w, models["bpm_stats"])
    X_new = project(models["pca"], F_new)
    xy_new = models["umap"].transform(F_new).astype(np.float32)

    playlists = _playlists(assign)
    keys = list(playlists)
    row = {u: i for i, u in enumerate(old)}
    cents = np.stack([X_old[[row[u] for u in playlists[k]]].mean(axis=0) for k in keys])
    radii = np.array([np.percentile(np.linalg.norm(X_old[[row[u] for u in playlists[k]]] - c, axis=1), params["radius_pct"])
                      for k, c in zip(keys, cents)])
    radii = np.maximum(radii, np.median(radii))  # playlists chicas: radio mínimo = mediana
    d = np.linalg.norm(X_new[:, None, :] - cents[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    inside = d[np.arange(len(cand)), nearest] <= radii[nearest]
    origin = {}
    target: Dict[str, Tuple[int, int]] = {}
    for i, u in enumerate(cand):
        if inside[i]:
            target[u], origin[u] = keys[nearest[i]], "add"
    pool = [i for i in range(len(cand)) if not inside[i]]
    new_lists: Dict[Tuple[int, int], List[str]] = {}
    if len(pool) >= params["min_new_playlist"]:
        k = max(1, int(round(len(pool) / params["l2_target_size"])))
        labels = _ward_cluster(X_new[pool], k)
        l1_ids = sorted({a for a, _ in keys})
        l1_cents = np.stack([X_old[[row[u] for kk in keys if kk[0] == a for u in playlists[kk]]].mean(axis=0) for a in l1_ids])
        next_l2 = {a: max(b for aa, b in keys if aa == a) + 1 for a in l1_ids}
        for lab in sorted(set(labels)):
            members = [pool[j] for j in np.flatnonzero(labels == lab)]
            a = l1_ids[int(np.linalg.norm(l1_cents - X_new[members].mean(axis=0), axis=1).argmin())]
            key = (a, next_l2[a])
            next_l2[a] += 1
            muids = [cand[i] for i in members]
            new_lists[key] = order_playlist(lib, muids, params["ordering_weights"])
            for u in muids:
                target[u], origin[u] = key, "add-new"
    else:
        for i in pool:
            target[cand[i]], origin[cand[i]] = keys[nearest[i]], "add-far"

    ci, ki = {u: i for i, u in enumerate(cand)}, {k: i for i, k in enumerate(keys)}
    by_dist = sorted((u for u in cand if origin[u] != "add-new"), key=lambda u: float(d[ci[u], ki[target[u]]]))
    for u in by_dist:  # los más típicos primero
        playlists[target[u]] = insert_into(lib, playlists[target[u]], u, params["ordering_weights"])
    playlists.update(new_lists)

    new_rows = pd.DataFrame({"track_uid": cand, "l1": 0, "l2": 0, "position": 0,
                             "x": xy_new[:, 0], "y": xy_new[:, 1], "origin": [origin[u] for u in cand]})
    assign = _renumber(pd.concat([assign, new_rows], ignore_index=True), playlists)
    names = dict(names)
    fresh = name_playlists(assign[assign.set_index(["l1", "l2"]).index.isin(list(new_lists))], lib.catalog)
    for (a, b) in new_lists:
        base = fresh.get(f"l1_{a}_l2_{b}", f"{_cluster_to_letter(a)}{b + 1}")
        names[f"l1_{a}_l2_{b}"] = f"{base} (nuevos)"
    counts = pd.Series(origin).value_counts().to_dict()
    v = store.commit(meta, assign, names, "add", {"scope": scope or org_scopes(meta), "added": len(cand), **counts})
    return v, {"added": len(cand), **counts}


def resolve_track(query: str, assign: pd.DataFrame, catalog: pd.DataFrame) -> str:
    """uid (o su prefijo), ruta relativa exacta, o texto contenido en 'artista - título' / nombre de archivo."""
    uids = assign["track_uid"].tolist()
    q = query.strip()
    hits = [u for u in uids if u.startswith(q)] if len(q) >= 8 else []
    if not hits:
        cat = catalog.reindex(uids)
        rel = cat["rel_path"].fillna("")
        hits = [u for u, r in rel.items() if r == q.replace("\\", "/")]
    if not hits:
        cat = catalog.reindex(uids)
        text = (cat["artist"].fillna("") + " - " + cat["title"].fillna("") + " | " + cat["filename"].fillna("")).str.lower()
        hits = [u for u, t in text.items() if q.lower() in t]
    if len(hits) != 1:
        cat = catalog.reindex(hits[:10])
        options = "\n".join(f"  {u[:12]}  {r}" for u, r in zip(cat.index, cat["rel_path"]))
        raise ValueError(f"'{query}' coincide con {len(hits)} temas de la organización" + (f":\n{options}" if hits else ""))
    return hits[0]


def link(store: OrgStore, lib: Library, queries: List[str], rebuild: bool = False) -> Tuple[int, Dict]:
    """Semilla: los temas indicados (y todo lo ya vinculado a ellos) van a una misma playlist."""
    v, info = link_groups(store, lib, [queries], rebuild)
    return v, info["groups"][0] if info["mode"] == "frozen" else {"group": info["groups"][0], "mode": "rebuild"}


def link_groups(store: OrgStore, lib: Library, query_groups: List, rebuild: bool = False) -> Tuple[int, Dict]:
    """Varias semillas de una vez (una sola versión nueva). Congelado: por cada grupo se mueve solo lo
    necesario a la playlist, entre las que ya contienen algún miembro, que minimiza la distancia total
    de los que se mueven (la del primer tema gana empates). rebuild: build desde cero con todas."""
    meta = store.meta()
    assign, names, _ = store.load()
    current = store.fusions()
    new_groups, new_fusions = [], []
    for k, item in enumerate(query_groups):
        queries = item["tracks"] if isinstance(item, dict) else item
        uids = list(dict.fromkeys(resolve_track(q, assign, lib.catalog) for q in queries))
        if len(uids) < 2:
            raise ValueError(f"Hacen falta al menos dos temas distintos: {queries}")
        new_groups.append(uids)
        idx = len(current) + k
        new_fusions.append({"name": (item.get("name") if isinstance(item, dict) else None) or f"Fusión {idx + 1}",
                            "color": int(item.get("color", idx)) % FUSION_COLORS if isinstance(item, dict) else idx % FUSION_COLORS,
                            "tracks": uids})
    fusions = merge_fusions(current, new_fusions)
    groups = [f["tracks"] for f in fusions]
    comps = [next(c for c in groups if g[0] in c) for g in new_groups]
    if rebuild:
        v = build(store, lib, meta["params"], org_scopes(meta), action="link-rebuild", fusions=fusions)
        return v, {"groups": comps, "mode": "rebuild"}

    models = store.models(meta["model_version"])
    playlists = _playlists(assign)
    X = dict(zip(assign["track_uid"], project(models["pca"], features(
        lib, assign["track_uid"].tolist(), meta["params"]["bpm_weight"], models["bpm_stats"]))))
    results, all_moved = [], []
    for first, comp in ((g[0], c) for g, c in zip(new_groups, comps)):
        where = {u: k for k, order in playlists.items() for u in order}
        comp_in = [u for u in comp if u in where]
        cands = list(dict.fromkeys(where[u] for u in [first] + comp_in))

        def cost(k):
            c = np.mean([X[u] for u in playlists[k]], axis=0)
            return sum(float(np.linalg.norm(X[u] - c)) for u in comp_in if where[u] != k)
        target = min(cands, key=cost)
        moved = [u for u in comp_in if where[u] != target]
        for u in moved:
            playlists[where[u]] = [x for x in playlists[where[u]] if x != u]
            playlists[target] = insert_into(lib, playlists[target], u, meta["params"]["ordering_weights"])
        playlists = {k: v for k, v in playlists.items() if v}
        results.append({"group": comp, "moved": moved, "target": target})
        all_moved += moved
    assign = _renumber(assign, playlists)
    assign.loc[assign["track_uid"].isin(all_moved), "origin"] = "link"
    v = store.commit(meta, assign, names, "link", {"groups": [r["group"] for r in results], "moved": all_moved,
                                                     "targets": [list(r["target"]) for r in results]},
                     fusions=fusions)
    return v, {"groups": results, "mode": "frozen"}


def reorder(store: OrgStore, l1: int, l2: int, order: List[str]) -> int:
    """Orden a mano de una playlist: los mismos temas (uid o prefijo) en otro orden. Versión nueva
    ('reorder'); add y link congelado insertan después sin romper este orden relativo."""
    meta = store.meta()
    assign, names, _ = store.load()
    playlists = _playlists(assign)
    key = (int(l1), int(l2))
    if key not in playlists:
        raise ValueError(f"No existe la playlist {key}")
    current = playlists[key]
    new = []
    for q in order:
        hits = [u for u in current if u.startswith(str(q))]
        if len(hits) != 1:
            raise ValueError(f"'{q}' no identifica un tema de esa playlist")
        new.append(hits[0])
    if sorted(new) != sorted(current):
        raise ValueError("El orden nuevo tiene que tener exactamente los temas de la playlist")
    if new == current:
        return meta["current_version"]
    playlists[key] = new
    return store.commit(meta, _renumber(assign, playlists), names, "reorder",
                        {"playlist": list(key), "playlist_name": names.get(f"l1_{key[0]}_l2_{key[1]}", "")})


def remove_tracks(store: OrgStore, replace: Dict[str, str], action: str = "dedupe", detail: Optional[Dict] = None) -> int:
    """Saca temas de la organización sin mover al resto (copias de temas repetidos). replace = {tema que
    sale: tema que se queda}; en las fusiones, la copia que sale se reemplaza por la que se queda."""
    meta = store.meta()
    assign, names, _ = store.load()
    gone = [u for u in assign["track_uid"] if u in replace]
    if not gone:
        return meta["current_version"]
    playlists = {k: [u for u in order if u not in replace] for k, order in _playlists(assign).items()}
    playlists = {k: order for k, order in playlists.items() if order}
    assign = _renumber(assign[~assign["track_uid"].isin(gone)], playlists)
    fusions = []
    for f in store.fusions():
        tracks = list(dict.fromkeys(replace.get(u, u) for u in f["tracks"]))
        if len(tracks) >= 2:
            fusions.append({**f, "tracks": tracks})
    return store.commit(meta, assign, names, action, {"removed": gone, **(detail or {})}, fusions=fusions)


def remove_fusion(store: OrgStore, name: str) -> int:
    """Quita una fusión aplicada. Sus temas quedan en las playlists donde están; solo dejan de estar
    obligados a ir juntos (un build posterior puede separarlos). Versión nueva ('unlink')."""
    meta = store.meta()
    assign, names, _ = store.load()
    fusions = store.fusions()
    idx = next((i for i, f in enumerate(fusions) if f["name"] == name), None)
    if idx is None:
        raise ValueError(f"No hay una fusión aplicada llamada «{name}»")
    gone = fusions.pop(idx)
    return store.commit(meta, assign, names, "unlink", {"fusion": name, "tracks": gone["tracks"]}, fusions=fusions)


def read_seed_file(path: Path, org_name: str) -> List[Dict]:
    """Fusiones exportadas por la página de revisión: {"org": ..., "groups"|"fusions": [{name?, color?, tracks}]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("org") and data["org"] != org_name:
        raise ValueError(f"El archivo es de la organización '{data['org']}', no de '{org_name}'")
    items = data.get("fusions", data.get("groups", []))
    groups = [{"name": g.get("name"), "color": g.get("color"), "tracks": g["tracks"]} if isinstance(g, dict)
              else {"tracks": g} for g in items]
    groups = [{k: v for k, v in g.items() if v is not None} for g in groups if len(g["tracks"]) >= 2]
    if not groups:
        raise ValueError(f"{path}: no hay grupos de al menos dos temas")
    return groups


def load_for_export(artifacts: Path, name: str, version: Optional[int] = None):
    """(DataFrame con label_l1/label_l2/position/umap_x/umap_y/origin, nombres, meta, versión)."""
    store = OrgStore(artifacts, name)
    assign, names, v = store.load(version)
    df = assign.rename(columns={"l1": "label_l1", "l2": "label_l2", "x": "umap_x", "y": "umap_y"})
    return df, names, store.meta(), v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def ingest(dataset: str, scope: str, rep: str, vocals: bool) -> None:
    """Catálogo, tags, embeddings y (opcional) etiqueta Vocal solo de la subcarpeta."""
    py, pipe = sys.executable, REPO_ROOT / "src" / "v4" / "pipeline"
    backend = rep.split("_")[0]
    _run([py, str(pipe / "phase0_ingest.py"), "--dataset-name", dataset, "--scope", scope])
    _run([py, str(pipe / "phase1_tags.py"), "--dataset-name", dataset, "--estimate-missing"])
    _run([py, str(pipe / "extract_representations.py"), "--dataset-name", dataset, "--models", backend,
          "--folder", scope, "--min-duration", "90", "--max-duration", "900"])
    if vocals and backend == "clap":
        _run([py, str(pipe / "tag_vocals.py"), "--dataset-name", dataset, "--method", "clap", "--write-tags"])


def show(store: OrgStore) -> None:
    meta = store.meta()
    assign, names, v = store.load()
    print(f"Organización '{store.name}' (alcance: {', '.join(x or 'toda la biblioteca' for x in org_scopes(meta))}) · versión {v} · "
          f"modelos de v{meta['model_version']}")
    for h in meta["history"]:
        extra = {k: h[k] for k in h if k not in ("version", "created", "action", "model_version")}
        print(f"  v{h['version']}  {h['created']}  {h['action']:<13} {extra}")
    print(f"  {len(assign)} temas, {assign.groupby(['l1', 'l2']).ngroups} playlists, origen: "
          f"{assign['origin'].value_counts().to_dict()}")
    for f in store.fusions():
        print(f"  {f['name']}: {', '.join(u[:12] for u in f['tracks'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Organizaciones estables: ingest / import / build / add / link / show")
    parser.add_argument("command", choices=("ingest", "import", "build", "add", "link", "show"))
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--config", default=None)
    parser.add_argument("--name", help="Nombre de la organización")
    parser.add_argument("--scope", default=None, help="Subcarpeta (relativa a la carpeta de audio)")
    parser.add_argument("--from-hash", default=None, help="import: config_hash de Phase 2-4")
    parser.add_argument("--track", action="append", default=[], help="link: tema (uid, ruta o texto 'artista - título'); repetible")
    parser.add_argument("--rebuild", action="store_true", help="link: rehacer todo desde cero con las semillas")
    parser.add_argument("--from-file", default=None, help="link: JSON de semillas exportado desde la página de revisión")
    parser.add_argument("--rep", default=DEFAULT_PARAMS["rep"])
    parser.add_argument("--bpm-weight", type=float, default=DEFAULT_PARAMS["bpm_weight"])
    parser.add_argument("--n-l1", type=int, default=None, help="build: carpetas (default N/150)")
    parser.add_argument("--l2-target-size", type=int, default=DEFAULT_PARAMS["l2_target_size"])
    parser.add_argument("--no-vocals", action="store_true", help="ingest: no escribir la etiqueta Vocal")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    if args.command == "ingest":
        if not args.scope:
            parser.error("ingest necesita --scope")
        ingest(args.dataset_name, args.scope, args.rep, not args.no_vocals)
        return 0
    if not args.name:
        parser.error("falta --name")
    store = OrgStore(artifacts, args.name)
    if args.command == "show":
        show(store)
        return 0
    if args.command == "import":
        if store.exists():
            parser.error(f"la organización '{args.name}' ya existe")
        lib = Library(artifacts, json.loads((artifacts / "clustering" / f"config_{args.from_hash}.json").read_text())["rep"])
        v = import_hash(store, lib, artifacts, args.from_hash)
    elif args.command == "build":
        params = dict(store.meta()["params"]) if store.exists() else dict(DEFAULT_PARAMS)
        if not store.exists():
            params.update(rep=args.rep, bpm_weight=args.bpm_weight, n_l1=args.n_l1, l2_target_size=args.l2_target_size)
        elif args.n_l1:
            params["n_l1"] = args.n_l1
        scopes = [args.scope] if args.scope is not None else (org_scopes(store.meta()) if store.exists() else [""])
        v = build(store, Library(artifacts, params["rep"]), params, scopes)
    elif args.command == "add":
        v, info = add(store, Library(artifacts, store.meta()["params"]["rep"]), args.scope)
        print(f"[INFO] Agregados: {info}")
    else:  # link
        lib = Library(artifacts, store.meta()["params"]["rep"])
        if args.from_file:
            v, info = link_groups(store, lib, read_seed_file(Path(args.from_file), args.name), args.rebuild)
        else:
            v, info = link_groups(store, lib, [args.track], args.rebuild)
        if info["mode"] == "frozen":
            for r in info["groups"]:
                print(f"[INFO] Semilla de {len(r['group'])} temas → playlist {r['target']}; movidos: {len(r['moved'])}")
        else:
            print(f"[INFO] Reconstruida desde cero con {len(store.fusions())} fusiones")
    show(store)
    print(f"[INFO] Versión actual: v{v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
