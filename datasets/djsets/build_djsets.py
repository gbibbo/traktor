"""
PURPOSE: Arma el dataset DJSets v1 en una sola carpeta a partir de lo que dejaron D5 y D6 (tracklists_1001.py,
         beatport_previews.py, youtube_temas.py, descarga_continua.py). Es la parte propia de este proyecto; el lector
         que acompaña al dataset es djsets.py, que no depende del repositorio.
           build    -> <out>/metadata/*.csv, dataset_info.json, checksums.md5, README.md y djsets.py, desde los sets
                       y los manifiestos locales (--root, en este PC D:\\TRAKTOR ML datos\\dj_sets) y el listado con md5
                       de las tandas en el Drive. No baja audio: cada tanda .tar ya subida es un shard del dataset.
           publish  -> <dest> (por defecto erclm:traktor_ml/djsets_v1): copia los shards dentro del Drive con nombres
                       nuevos (copia del lado del servidor, sin pasar por este PC), copia los originales a raw/ y sube
                       lo armado por build; al final compara el md5 de cada archivo del destino con checksums.md5.
                       Nunca borra nada, ni en el origen ni en el destino; se puede repetir y salta lo que ya está.
         Reglas: AGENTS.md («Compute»): nada fuera de erclm:traktor_ml/; el dataset es privado (audio con derechos de
         sus dueños), su código va a Git y sus datos solo al Drive y a las copias de Gabriel.
CHANGELOG:
  - 2026-11-13: Creación (v1).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AQUI))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.dj_sets import beatport_previews as bpv  # noqa: E402
import djsets  # noqa: E402

VERSION = "1.0"
NOMBRE = "djsets_v1"
REMOTO = "erclm:traktor_ml"
FUENTES = {"beatport": "beatport_previews", "youtube": "youtube_temas"}   # fuente -> carpeta de las corridas
EXCLUIR_CORRIDA = re.compile(r"^(prueba|control_suave|tmp|cache)")
N_CARPETAS = 100
SPLIT = (("train", 80), ("validation", 90), ("test", 100))

COL_SETS = ["set_id", "title", "dj", "date", "minutes", "use", "kind", "b2b", "night_group", "n_rows", "n_identified",
            "n_tracks", "n_tracks_with_audio", "share_with_cue", "recordings", "listing_genres", "listing_pct_identified",
            "seed_queries", "url"]
COL_SET_TRACKS = ["set_id", "position", "number", "cue_s", "with_previous", "identified", "track_id", "id_1001",
                  "name", "artist", "label", "genre_1001", "duration_1001_s", "edition", "buy_button"]
COL_TRACKS = ["track_id", "id_1001", "name", "artist", "title", "label_1001", "genre_1001", "duration_1001_s", "n_sets",
              "n_rows", "buy_button", "has_audio", "audio_status", "audio_attempts", "audio_source", "recording_id", "audio_shard",
              "audio_member", "audio_file", "audio_ext", "audio_bytes", "audio_md5", "audio_duration_s",
              "audio_is_full_track", "preview_start_s", "beatport_id", "beatport_via", "beatport_match_level",
              "beatport_soft_mix", "bp_track_name", "bp_mix_name", "bp_artists", "bp_bpm", "bp_key", "bp_genre",
              "bp_label", "bp_isrc", "bp_release", "bp_release_date", "bp_length_s", "youtube_id", "youtube_title",
              "youtube_channel", "youtube_topic", "youtube_codec", "youtube_abr_kbps", "youtube_sample_rate"]
COL_SHARDS = ["shard", "source", "n_audio", "bytes", "md5", "created", "origin"]


# ---------------------------------------------------------------- utilidades


def escribir_csv(ruta: Path, columnas: Sequence[str], filas: Sequence[Dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=list(columnas), extrasaction="raise", lineterminator="\n")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: "" if fila.get(c) is None else fila.get(c) for c in columnas})


def texto(valor) -> str:
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return "1" if valor else "0"
    if isinstance(valor, (list, tuple)):
        return "; ".join(str(v) for v in valor)
    return str(valor)


def carpeta_de(recording_id: str) -> str:
    return f"{int(hashlib.md5(recording_id.encode()).hexdigest()[:8], 16) % N_CARPETAS:02d}"


def clave_dj(dj: str) -> str:
    """El primer DJ del nombre («A & B», «A b2b B», «A vs. B», «A, B»), en minúsculas."""
    primero = re.split(r"\s+(?:&|b2b|B2B|vs\.?|x)\s+|,\s*", dj or "")[0]
    return primero.strip().lower()


def split_de(dj: str) -> str:
    cubeta = int(hashlib.md5(clave_dj(dj).encode()).hexdigest()[:8], 16) % 100
    return next(nombre for nombre, tope in SPLIT if cubeta < tope)


def listar_remoto(carpeta: str, remoto: str = REMOTO) -> Dict[str, dict]:
    salida = subprocess.run(["rclone", "lsjson", "-R", "--hash", "--files-only", f"{remoto}/{carpeta}"],
                            capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return {e["Path"]: e for e in json.loads(salida)}


# ---------------------------------------------------------------- lectura de D5 y D6


def leer_sets(root: Path) -> Tuple[List[Dict[str, str]], List[dict]]:
    """(filas de candidatos.csv sin veto, sets bajados en ese orden)."""
    carpeta = root / "1001tracklists"
    candidatos = [f for f in bpv.leer_tabla(carpeta / "candidatos.csv") if not f.get("veto", "").strip()]
    sets = []
    for fila in candidatos:
        ruta = carpeta / "sets" / f"{fila['id']}.json"
        if not ruta.exists():
            raise FileNotFoundError(f"set sin bajar: {ruta}")
        sets.append(json.loads(ruta.read_text(encoding="utf-8")))
    return candidatos, sets


def bajar_manifiestos(cache: Path, remoto: str = REMOTO) -> None:
    for carpeta in FUENTES.values():
        subprocess.run(["rclone", "copy", f"{remoto}/{carpeta}", str(cache / carpeta), "--include", "tanda_*.json"], check=True)


def leer_tandas(root: Path, cache: Path, listados: Dict[str, Dict[str, dict]]) -> Tuple[List[dict], List[str]]:
    """Las tandas del Drive con .tar y manifiesto (el del Drive manda; cache/ tiene su copia), en orden de creación.
    Cada una: fuente, carpeta, corrida, tanda, creado, manifest, tar (entrada del listado con md5). Si la copia local en
    <root> existe, tiene que decir lo mismo (el formato del JSON puede cambiar). Avisos: tandas incompletas."""
    tandas, avisos = [], []
    for fuente, carpeta in FUENTES.items():
        listado = listados[carpeta]
        for rel_json in sorted(r for r in listado if r.endswith(".json") and "/tanda_" in r):
            corrida, nombre = rel_json.split("/", 1)
            if EXCLUIR_CORRIDA.match(corrida):
                continue
            rel = rel_json[: -len(".json")]
            tar = listado.get(rel + ".tar")
            if tar is None:
                avisos.append(f"{carpeta}/{rel}: manifiesto sin .tar en el Drive (no entra)")
                continue
            copia = cache / carpeta / rel_json
            if djsets.md5_archivo(copia) != listado[rel_json]["Hashes"]["md5"]:
                raise RuntimeError(f"{copia}: la copia bajada no coincide con el Drive")
            datos = json.loads(copia.read_text(encoding="utf-8"))
            local = root / carpeta / rel_json
            if local.exists() and json.loads(local.read_text(encoding="utf-8")) != datos:
                raise RuntimeError(f"{local}: dice otra cosa que el manifiesto del Drive")
            if not local.exists():
                avisos.append(f"{carpeta}/{rel_json}: sin copia local (se usa la del Drive)")
            tandas.append({"fuente": fuente, "carpeta": carpeta, "corrida": corrida, "tanda": Path(nombre).stem,
                           "creado": datos.get("creado", ""), "manifest": datos, "tar": tar})
    usados = {f"{t['carpeta']}/{t['corrida']}/{t['tanda']}.tar" for t in tandas}
    for carpeta, listado in listados.items():
        for ruta in listado:
            if ruta.endswith(".tar") and f"{carpeta}/{ruta}" not in usados:
                avisos.append(f"{carpeta}/{ruta}: .tar que no entra (sin manifiesto o corrida de prueba)")
    tandas.sort(key=lambda t: (list(FUENTES).index(t["fuente"]), t["creado"], t["corrida"], t["tanda"]))
    return tandas, avisos


# ---------------------------------------------------------------- armado


def armar(root: Path, cache: Path, listados: Dict[str, Dict[str, dict]]) -> dict:
    candidatos, sets = leer_sets(root)
    poblacion = bpv.elegir(sets, n=10 ** 9, todos=True)
    claves = {f["clave"]: f for f in poblacion}
    tandas, avisos = leer_tandas(root, cache, listados)

    shards, por_fuente = [], Counter()
    for t in tandas:
        nombre = f"audio/{t['fuente']}/{t['fuente']}_{por_fuente[t['fuente']]:03d}.tar"
        por_fuente[t["fuente"]] += 1
        t["shard"] = nombre
        shards.append({"shard": nombre, "source": t["fuente"], "n_audio": t["manifest"].get("ok", 0),
                       "bytes": t["tar"]["Size"], "md5": t["tar"]["Hashes"]["md5"], "created": t["creado"],
                       "origin": f"{t['carpeta']}/{t['corrida']}/{t['tanda']}.tar"})

    # Audio por clave: la primera tanda (Beatport antes que YouTube) que lo resolvió. Metadatos de Beatport también
    # de los temas sin audio, si alguna tanda los trajo.
    audio, beatport_meta, estados, fuera = {}, {}, defaultdict(list), Counter()
    vistos_recording: Dict[str, Tuple[str, str]] = {}
    duplicados = 0
    for t in tandas:
        entradas = t["manifest"]["temas"]
        por_clave = {e["clave_1001"]: e for e in entradas}
        for e in entradas:
            clave = e["clave_1001"]
            if clave not in claves:
                fuera[e["estado"]] += 1
                continue
            estados[clave].append(f"{t['fuente']}:{e['estado']}")
            if e.get("detalles") and clave not in beatport_meta:
                beatport_meta[clave] = (e["detalles"], e)
            if clave in audio:
                continue
            base = e
            if e["estado"] == "repetido":
                base = por_clave.get(e.get("igual_a", ""))
                if not base or base["estado"] != "ok":
                    continue
            elif e["estado"] != "ok":
                continue
            if t["fuente"] == "beatport":
                recording = f"bp_{base['beatport_id']}"
                ext = "mp3"
            else:
                recording = f"yt_{base['video_id']}"
                ext = base.get("ext") or Path(base["archivo"]).suffix.lstrip(".")
            if recording in vistos_recording and vistos_recording[recording][0] != t["shard"]:
                duplicados += 1  # el mismo audio en otra tanda: se usa la primera copia
                shard, miembro = vistos_recording[recording]
            else:
                shard, miembro = t["shard"], base["archivo"]
                vistos_recording.setdefault(recording, (shard, miembro))
            audio[clave] = {"fuente": t["fuente"], "entrada": base, "propia": e, "recording": recording, "ext": ext,
                            "shard": shard, "member": miembro}

    # Nombres de archivo desempaquetado sin choques aun en discos que no distinguen mayúsculas (FAT32, NTFS, SMB).
    exts = {a["recording"]: a["ext"] for a in audio.values()}
    archivo_de, ocupados, choques = {}, set(), 0
    for recording in sorted(exts):
        fuente = "beatport" if recording.startswith("bp_") else "youtube"
        carpeta = f"audio_files/{fuente}/{carpeta_de(recording)}"
        nombre = f"{recording}.{exts[recording]}"
        if f"{carpeta}/{nombre}".lower() in ocupados:
            choques += 1
            nombre = f"{recording}__{hashlib.md5(recording.encode()).hexdigest()[:6]}.{exts[recording]}"
        ocupados.add(f"{carpeta}/{nombre}".lower())
        archivo_de[recording] = f"{carpeta}/{nombre}"

    # Filas de los sets.
    aparece_sets, aparece_filas, primera = defaultdict(set), Counter(), {}
    filas_sets_temas, filas_sets = [], []
    meta_set = {f["id"]: f for f in candidatos}
    for dato in sets:
        n_tracks, con_audio, con_cue, n_id = set(), set(), 0, 0
        for pos, tema in enumerate(dato["temas"], 1):
            nombre = tema["nombre"] or tema["visible"]
            clave = None
            if tema["identificado"]:
                n_id += 1
                posible = tema["id_texto"] or "nombre:" + bpv.bp.norm(nombre)
                clave = posible if posible in claves else None
            if clave:
                aparece_sets[clave].add(dato["id"])
                aparece_filas[clave] += 1
                primera.setdefault(clave, tema)
                n_tracks.add(clave)
                if clave in audio:
                    con_audio.add(clave)
            # Como tracklists_1001.medir: un 0 fuera de la primera fila es «sin tiempo» en la página.
            segundo = tema.get("segundo")
            cue = segundo if segundo is not None and (segundo > 0 or pos == 1) else None
            con_cue += cue is not None
            filas_sets_temas.append({
                "set_id": dato["id"], "position": pos, "number": tema.get("numero", ""), "cue_s": texto(cue),
                "with_previous": texto(bool(tema.get("junto_al_anterior"))), "identified": texto(bool(tema["identificado"])),
                "track_id": clave or "", "id_1001": tema.get("id_texto", ""), "name": nombre, "artist": tema.get("artista", ""),
                "label": tema.get("sello", ""), "genre_1001": tema.get("genero", ""), "duration_1001_s": texto(tema.get("duracion_s")),
                "edition": tema.get("edicion", ""), "buy_button": texto("affiliate" in (tema.get("fuentes") or []))})
        m = meta_set.get(dato["id"], {})
        n = len(dato["temas"])
        filas_sets.append({
            "set_id": dato["id"], "title": dato.get("titulo", ""), "dj": m.get("dj", ""), "date": dato.get("fecha", ""),
            "minutes": texto(dato.get("minutos")), "use": m.get("uso", dato.get("uso", "")), "kind": m.get("tipo", ""),
            "b2b": texto(m.get("b2b") == "si"), "night_group": m.get("grupo_noche", ""), "n_rows": n, "n_identified": n_id,
            "n_tracks": len(n_tracks), "n_tracks_with_audio": len(con_audio),
            "share_with_cue": f"{con_cue / n:.3f}" if n else "", "recordings": len(dato.get("grabaciones") or []),
            "listing_genres": m.get("generos", ""), "listing_pct_identified": m.get("pct_id", ""),
            "seed_queries": m.get("consultas", ""), "url": dato.get("url", "")})

    # Tabla de temas.
    filas_temas = []
    for clave, fila in claves.items():
        tema = primera.get(clave, {})
        a = audio.get(clave)
        meta, entrada_meta = beatport_meta.get(clave, (None, None))
        if a and a["fuente"] == "beatport":
            meta, entrada_meta = a["entrada"].get("detalles") or meta, a["propia"] if a["propia"]["estado"] == "ok" else a["entrada"]
        r = {"track_id": clave, "id_1001": clave if not clave.startswith("nombre:") else "", "name": fila["nombre"],
             "artist": fila["artista"], "title": fila["titulo"], "label_1001": tema.get("sello", ""),
             "genre_1001": tema.get("genero", ""), "duration_1001_s": texto(tema.get("duracion_s")),
             "n_sets": len(aparece_sets[clave]), "n_rows": aparece_filas[clave], "buy_button": texto(fila["con_boton"] == "si"),
             "has_audio": texto(bool(a)), "audio_status": "ok" if a else (estados[clave][-1] if estados[clave] else "no_intentado"),
             "audio_attempts": " > ".join(estados[clave])}
        if meta:
            r.update({"beatport_id": texto(meta.get("track_id")), "bp_track_name": texto(meta.get("track_name")),
                      "bp_mix_name": texto(meta.get("mix_name")), "bp_artists": texto(meta.get("artists")),
                      "bp_bpm": texto(meta.get("bpm")), "bp_key": texto(meta.get("key")), "bp_genre": texto(meta.get("genre")),
                      "bp_label": texto(meta.get("label")), "bp_isrc": texto(meta.get("isrc")),
                      "bp_release": texto(meta.get("release_name")), "bp_release_date": texto(meta.get("release_date")),
                      "bp_length_s": f"{meta['track_length_ms'] / 1000:.1f}" if meta.get("track_length_ms") else "",
                      "beatport_match_level": texto(entrada_meta.get("nivel_busqueda")),
                      "beatport_soft_mix": texto(entrada_meta.get("mezcla_suave"))})
        if a:
            e = a["entrada"]
            r.update({"audio_source": a["fuente"], "recording_id": a["recording"], "audio_shard": a["shard"],
                      "audio_member": a["member"], "audio_file": archivo_de[a["recording"]], "audio_ext": a["ext"],
                      "audio_bytes": e["bytes"], "audio_md5": e["md5"]})
            if a["fuente"] == "beatport":
                d = e.get("detalles") or {}
                inicio, fin = d.get("sample_start_ms"), d.get("sample_end_ms")
                r.update({"beatport_via": texto(a["propia"].get("via") or e.get("via")), "audio_is_full_track": "0",
                          "audio_duration_s": f"{(fin - inicio) / 1000:.1f}" if inicio is not None and fin else "",
                          "preview_start_s": f"{inicio / 1000:.1f}" if inicio is not None else ""})
            else:
                r.update({"audio_is_full_track": "1", "audio_duration_s": texto(e.get("duracion_s")),
                          "youtube_id": e["video_id"], "youtube_title": texto(e.get("titulo_youtube")),
                          "youtube_channel": texto(e.get("canal")), "youtube_topic": texto(bool(e.get("topic"))),
                          "youtube_codec": texto(e.get("acodec")), "youtube_abr_kbps": texto(e.get("abr")),
                          "youtube_sample_rate": texto(e.get("asr"))})
        filas_temas.append(r)
    filas_temas.sort(key=lambda r: r["track_id"])

    split = [{"set_id": f["set_id"], "dj": f["dj"], "split_key": clave_dj(f["dj"]), "split": split_de(f["dj"])} for f in filas_sets]
    resumen = {
        "sets": len(filas_sets), "set_rows": len(filas_sets_temas), "tracks": len(filas_temas),
        "tracks_with_audio": sum(r["has_audio"] == "1" for r in filas_temas),
        "tracks_with_audio_by_source": dict(Counter(r["audio_source"] for r in filas_temas if r["has_audio"] == "1")),
        "recordings": len(exts), "audio_bytes": sum(int(s["bytes"]) for s in shards),
        "shards": dict(Counter(s["source"] for s in shards)), "audio_status": dict(Counter(r["audio_status"] for r in filas_temas)),
        "tracks_with_beatport_metadata": sum(bool(r.get("beatport_id")) for r in filas_temas),
        "consecutive_pairs": None, "duplicate_audio_copies_ignored": duplicados, "case_collisions_renamed": choques,
        "manifest_entries_outside_population": dict(fuera),
        "split_sets": dict(Counter(s["split"] for s in split))}
    return {"sets": filas_sets, "set_tracks": filas_sets_temas, "tracks": filas_temas, "shards": shards, "split": split,
            "resumen": resumen, "avisos": avisos, "tandas": tandas}


def commit_actual() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def escribir(out: Path, datos: dict, commit: str = "") -> None:
    """Escribe en out/ todo lo que no es audio: metadata/, dataset_info.json, README.md, djsets.py, checksums.md5."""
    out.mkdir(parents=True, exist_ok=True)
    meta = out / djsets.METADATA
    escribir_csv(meta / djsets.FILES["sets"], COL_SETS, datos["sets"])
    escribir_csv(meta / djsets.FILES["set_tracks"], COL_SET_TRACKS, datos["set_tracks"])
    escribir_csv(meta / djsets.FILES["tracks"], COL_TRACKS, datos["tracks"])
    escribir_csv(meta / djsets.FILES["shards"], COL_SHARDS, datos["shards"])
    escribir_csv(meta / djsets.FILES["split"], ["set_id", "dj", "split_key", "split"], datos["split"])
    armado = djsets.DJSets(out)
    datos["resumen"]["consecutive_pairs"] = len(armado.pairs())
    datos["resumen"]["consecutive_pairs_with_audio"] = len(armado.pairs(with_audio=True))
    info = {"name": "DJSets", "version": VERSION, "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "builder": "datasets/djsets/build_djsets.py", "builder_commit": commit,
            "counts": datos["resumen"], "sources": {
                "sets": "1001Tracklists (tracklists_1001.py, cuenta de Gabriel, 8 s entre páginas), 2026-11-09 a 11-12",
                "beatport": "previews MP3 de Beatport (beatport_previews.py): búsqueda por nombre y botón de compra de 1001Tracklists",
                "youtube": "audio completo de YouTube (youtube_temas.py, yt-dlp, sin recodificar) para lo que Beatport no dio"},
            "terms": "Privado, solo investigación de Gabriel. No se redistribuye. El audio es de sus dueños."}
    (out / djsets.INFO).write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shutil.copy2(AQUI / "djsets.py", out / "djsets.py")
    shutil.copy2(AQUI / "README.md", out / "README.md")
    (out / "build_avisos.txt").write_text("".join(a + "\n" for a in datos["avisos"]), encoding="utf-8")
    lineas = [f"{djsets.md5_archivo(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*"))
              if p.is_file() and p.name != djsets.CHECKSUMS and not p.relative_to(out).as_posix().startswith(("audio", "raw"))]
    lineas += [f"{s['md5']}  {s['shard']}" for s in datos["shards"]]
    (out / djsets.CHECKSUMS).write_text("\n".join(lineas) + "\n", encoding="utf-8")


def cmd_build(args) -> int:
    out = Path(args.out)
    listados = {c: listar_remoto(c, args.remoto) for c in FUENTES.values()}
    cache = Path(args.cache) if args.cache else out.parent / (out.name + "_manifiestos")
    bajar_manifiestos(cache, args.remoto)
    datos = armar(Path(args.root), cache, listados)
    escribir(out, datos, commit_actual())
    print(json.dumps(datos["resumen"], ensure_ascii=False, indent=1))
    print(f"avisos: {len(datos['avisos'])} (build_avisos.txt)")
    for a in datos["avisos"][:20]:
        print("  " + a)
    return 0


# ---------------------------------------------------------------- publicación


def rclone(*partes: str) -> None:
    subprocess.run(["rclone", *partes], check=True)


def cmd_publish(args) -> int:
    out, dest = Path(args.out), args.dest.rstrip("/")
    if not dest.startswith(REMOTO + "/"):
        raise SystemExit(f"destino fuera de {REMOTO}/: {dest}")
    shards = djsets._leer_csv(out / djsets.METADATA / djsets.FILES["shards"])
    try:
        ya = {e["Path"]: e["Hashes"]["md5"] for e in json.loads(subprocess.run(
            ["rclone", "lsjson", "-R", "--hash", "--files-only", f"{dest}/audio"], capture_output=True, text=True,
            encoding="utf-8").stdout or "[]")}
    except json.JSONDecodeError:
        ya = {}
    for i, s in enumerate(shards, 1):
        rel = s["shard"][len("audio/"):]
        if ya.get(rel) == s["md5"]:
            continue
        print(f"[{i}/{len(shards)}] {s['origin']} -> {s['shard']}", flush=True)
        rclone("copyto", f"{args.remoto}/{s['origin']}", f"{dest}/{s['shard']}")
    corridas = sorted({"/".join(s["origin"].split("/")[:2]) for s in shards})
    print("raw: sets de 1001Tracklists y manifiestos de cada corrida", flush=True)
    rclone("copy", f"{args.remoto}/dj_sets/1001tracklists", f"{dest}/raw/1001tracklists")
    for corrida in corridas:
        rclone("copy", f"{args.remoto}/dj_sets/{corrida}", f"{dest}/raw/{corrida}")
    print("metadatos", flush=True)
    rclone("copy", str(out), dest, "--exclude", "audio/**", "--exclude", "audio_files/**")
    return verificar(out, dest)


def verificar(out: Path, dest: str) -> int:
    remoto = {e["Path"]: e["Hashes"]["md5"] for e in json.loads(subprocess.run(
        ["rclone", "lsjson", "-R", "--hash", "--files-only", dest, "--exclude", "raw/**"], capture_output=True, text=True,
        encoding="utf-8", check=True).stdout)}
    malos = 0
    for linea in (out / djsets.CHECKSUMS).read_text(encoding="utf-8").splitlines():
        esperado, nombre = linea.split(None, 1)
        if remoto.get(nombre) != esperado:
            malos += 1
            print(f"distinto o ausente en el destino: {nombre}")
    print(f"verificados {len(remoto)} archivos del destino; {malos} con problemas")
    return 1 if malos else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arma y publica el dataset DJSets")
    parser.add_argument("--remoto", default=REMOTO)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True, help="carpeta de datos de sets (en este PC, D:\\TRAKTOR ML datos\\dj_sets)")
    b.add_argument("--out", required=True, help="carpeta local donde se arma (metadatos; sin audio)")
    b.add_argument("--cache", help="dónde bajar los manifiestos del Drive (por defecto <out>_manifiestos)")
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--dest", default=f"{REMOTO}/{NOMBRE}")
    v = sub.add_parser("verify")
    v.add_argument("--out", required=True)
    v.add_argument("--dest", default=f"{REMOTO}/{NOMBRE}")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "publish":
        return cmd_publish(args)
    return verificar(Path(args.out), args.dest)


if __name__ == "__main__":
    sys.exit(main())
