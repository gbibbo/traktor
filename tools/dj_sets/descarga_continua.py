"""
PURPOSE: D5/D6 sin pausas muertas (Gabriel, 2026-11-10: «que la descarga completa del dataset no se detenga en seco»;
         si hay que esperar más entre pedidos, se espera). Encadena solo, en tres carriles que corren a la vez, los
         pasos que hasta ahora se lanzaban a mano, con los mismos comandos de siempre:
           1001      navegador con la sesión de Gabriel, de a uno: los sets que faltan (tracklists_1001.py bajar, 200
                     páginas por corrida y 45 min entre corridas) y después el enlace del botón de compra de los temas
                     que la búsqueda no resolvió (beatport_previews.py enlaces --solo-sin-id, 200 por corrida).
           beatport  los lotes (bajar de a una tanda; elegir arma uno nuevo cuando hay sets nuevos) y, por cada lote
                     terminado, su rescate res_<lote>: rescate y buscar enseguida; bajar cuando los enlaces de 1001 ya
                     están pedidos. Entre tanda y tanda atiende lo que haya aparecido.
           youtube   las corridas yt_* (bajar); mientras un lote se baja, sus temas sin resolver sin botón de compra
                     (yt_<lote>_NN, de a 200 o más: Beatport ya no los da y no hay enlace que esperar); y por cada
                     rescate bajado, yt_res_<lote> con lo que sigue sin resolver (youtube_temas.py elegir). Cada
                     verificación anti-robots alarga la pausa entre pedidos (x1,5, tope 30 s; queda guardada en
                     youtube_temas/pausa_s.txt).
           copia     al arrancar y cada 2 h, rclone copy de <root> entero a erclm:traktor_ml/dj_sets/ (menos la caché de
                     Beatport y lo temporal): nada del dataset queda solo en el pendrive (Gabriel, 2026-11-11).
         Nunca se rinde: si un sitio corta (salida 2), espera (1001: 2 h, 3 h, 4 h; Beatport: 30 min a 2 h; YouTube:
         1 h a 2 h) y vuelve a probar, sin límite de intentos; un error del programa (otra salida) se reintenta a los
         10 min (hasta 1 h). No esquiva verificaciones ni usa otra cuenta.
         Parar: crear el archivo <root>/PARAR (corta lo que esté corriendo; las tandas subidas quedan y todo se
         retoma al volver a lanzar). Termina solo cuando no queda nada. Mantiene el PC despierto mientras corre;
         los procesos van con prioridad baja (Traktor y Rekordbox primero). Registro: <root>/descarga_continua.log.
           correr    -> los carriles (una sola instancia: descarga_continua.pid)
           guardian  -> relanza correr si se murió sin terminar, cada 5 min (no si existe PARAR); lanzarlo en vez
                        de correr deja todo andando aunque correr se caiga
           estado    -> qué falta en cada corrida, sin red
         lote_01..03 ya los rescató rescate_01 (sin enlaces): ese rescate es un origen más (res_rescate_01).
CHANGELOG:
  - 2026-11-10: Creación.
  - 2026-11-10: Beatport de a una tanda (el rescate de rescate_01 esperaba a que terminaran las 34 de lote_04) y
                YouTube adelanta los temas sin botón de los lotes (se quedaba sin trabajo hasta la madrugada).
                Las esperas ante un sitio quedan en espera_<carril>.txt: un reinicio no las acorta. La pausa de
                YouTube sube solo si la corrida bajó al menos 10 min antes del bloqueo (si no, es el mismo bloqueo).
  - 2026-11-11: un bloqueo de YouTube después de una corrida que bajó al menos 10 min cuenta como el primero (espera
                1 h), no como uno más seguido.
  - 2026-11-11: carril copia (Gabriel: los datos del dataset no pueden estar solo en el pendrive; los sets de
                1001Tracklists, las listas de las corridas y D7 vivían solo en D:).
  - 2026-11-11: guardian (correr murió a las 13:28 sin dejar rastro y la descarga quedó parada 19 min); escribir el
                registro y mirar el disco ya no pueden tumbar correr; descarga_terminada.txt al terminar.
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.dj_sets import beatport_previews as bpv  # noqa: E402
from tools.dj_sets import youtube_temas as yt  # noqa: E402

TANDA_BP, TANDA_YT = bpv.TANDA, yt.TANDA
YA_RESCATADOS = {"lote_01", "lote_02", "lote_03"}  # rescate_01 (2026-11-10) tomó sus temas sin resolver
ORIGEN = re.compile(r"^(piloto|lote_\d+|rescate_\d+)$")
LOTE = re.compile(r"^(piloto|lote_\d+)$")
VACIA = "sin_temas.txt"  # marca de un rescate o una corrida de YouTube que no tuvo temas
SCRIPTS = {"sets": "tools/dj_sets/tracklists_1001.py", "previews": "tools/dj_sets/beatport_previews.py",
           "youtube": "tools/dj_sets/youtube_temas.py"}
NAVEGADOR = Path.home() / ".traktor_ml"
PAUSA_YT_INICIAL, PAUSA_YT_TOPE = 6.0, 30.0
# Gabriel (2026-11-11): los datos del dataset no pueden estar solo en el pendrive. Copia de <root> entero al Drive
# (rclone copy: sube lo nuevo o cambiado y nunca borra allá), menos la caché de búsquedas de Beatport (se rehace) y lo
# temporal (el audio de las tandas ya va al Drive por su cuenta).
COPIA_DRIVE, COPIA_CADA_S = "dj_sets", 2 * 3600
FIN_MARCA = "descarga_terminada.txt"  # la deja correr al terminar todo; el guardián deja de relanzar
EXCLUIR_COPIA = ("beatport_previews/cache/**", "beatport_previews/tmp/**", "youtube_temas/tmp/**", "descarga_continua.pid",
                 "espera_*.txt", "PARAR", "*.tmp")

Tarea = Tuple[str, List[str]]


# ---------------------------------------------------------------- estado en disco (sin red)


def leer(ruta: Path) -> List[Dict[str, str]]:
    """temas.csv; si otro proceso lo está reemplazando en ese momento, reintenta."""
    for _ in range(5):
        try:
            return bpv.leer_tabla(ruta) if ruta.exists() else []
        except PermissionError:
            time.sleep(2)
    return bpv.leer_tabla(ruta)


def completa(base: Path, tam: int) -> bool:
    """Todas las tandas de la corrida subidas (cada bajar deja la copia local tanda_NNNNN.json después de subir)."""
    filas = leer(base / "temas.csv")
    if not filas:
        return False
    hechas = {p.stem for p in base.glob("tanda_*.json")}
    return all(bpv.nombre_tanda(k) in hechas for k in range(math.ceil(len(filas) / tam)))


def con_lista(carpeta: Path) -> List[str]:
    if not carpeta.exists():
        return []
    return sorted(d.name for d in carpeta.iterdir() if d.is_dir() and (d / "temas.csv").exists())


def resuelta(base: Path) -> bool:
    """Ya tiene lista o se sabe que no tuvo temas."""
    return (base / "temas.csv").exists() or (base / VACIA).exists()


def marcar_vacia(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / VACIA).write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


def sets_en_lotes(bp_dir: Path) -> int:
    try:
        return int((bp_dir / "sets_en_lotes.txt").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def contar_sets_bajados(root: Path) -> int:
    return len(list((root / "1001tracklists" / "sets").glob("*.json")))


def tarea_beatport(root: Path, n_sets: int) -> Optional[Tarea]:
    """El paso siguiente del carril de Beatport: primero lo corto (rescates), después los lotes, al final un lote nuevo."""
    bp_dir = root / "beatport_previews"
    temporal = str(bp_dir / "tmp")
    for origen in con_lista(bp_dir):
        if ORIGEN.match(origen) and origen not in YA_RESCATADOS and not resuelta(bp_dir / f"res_{origen}") \
                and completa(bp_dir / origen, TANDA_BP):
            return "rescate", ["--corrida", f"res_{origen}", "rescate", "--de", origen]
    for res in (c for c in con_lista(bp_dir) if c.startswith("res_")):
        filas = leer(bp_dir / res / "temas.csv")
        if any(not f.get("nivel_busqueda") for f in filas):
            return "buscar", ["--corrida", res, "buscar"]
        if bpv.pendientes_de_enlace(filas, True):
            continue  # espera los enlaces del carril 1001
        if not completa(bp_dir / res, TANDA_BP):
            return "bajar", ["--corrida", res, "bajar", "--temporal", temporal, "--max-tandas", "1"]
    lotes = [c for c in con_lista(bp_dir) if LOTE.match(c)]
    for lote in lotes:  # de a una tanda: entre tanda y tanda se atiende lo que haya aparecido
        if not completa(bp_dir / lote, TANDA_BP):
            return "bajar", ["--corrida", lote, "bajar", "--temporal", temporal, "--max-tandas", "1"]
    if n_sets > sets_en_lotes(bp_dir):
        numeros = [int(c.split("_")[1]) for c in lotes if c.startswith("lote_")]
        nuevo = f"lote_{max(numeros, default=0) + 1:02d}"
        excluir = [x for lote in lotes for x in ("--excluir", lote)]
        return "elegir", ["--corrida", nuevo, "elegir", "--todos", "--n", "1000000", *excluir]
    return None


def tarea_1001(root: Path, sets_pendientes: Optional[int]) -> Optional[Tarea]:
    """Primero los sets; después los enlaces de los rescates ya buscados."""
    if sets_pendientes:
        return "sets", ["bajar", "--n", "200"]
    bp_dir = root / "beatport_previews"
    for res in (c for c in con_lista(bp_dir) if c.startswith("res_")):
        filas = leer(bp_dir / res / "temas.csv")
        if filas and all(f.get("nivel_busqueda") for f in filas) and bpv.pendientes_de_enlace(filas, True):
            return "enlaces", ["--corrida", res, "enlaces", "--solo-sin-id", "--n", "200"]
    return None


def siguiente_nombre(carpeta: Path, prefijo: str) -> str:
    n = 1
    while (carpeta / f"{prefijo}_{n:02d}").exists():
        n += 1
    return f"{prefijo}_{n:02d}"


def tarea_youtube(root: Path) -> Optional[Tarea]:
    """Las corridas de YouTube sin terminar; después, una nueva por cada rescate bajado; y mientras un lote se baja, sus
    temas sin resolver que no tienen botón de compra (Beatport ya no los va a dar ni hay enlace que esperar), de a 200
    o más (o los que queden cuando el lote termina)."""
    bp_dir, yt_dir = root / "beatport_previews", root / "youtube_temas"
    corridas = [c for c in con_lista(yt_dir) if c != "tmp"]
    for corrida in corridas:
        if not completa(yt_dir / corrida, TANDA_YT):
            return "bajar", ["--corrida", corrida, "bajar", "--temporal", str(yt_dir / "tmp")]
    excluir = [x for c in corridas for x in ("--excluir", c)]
    for res in (c for c in con_lista(bp_dir) if c.startswith("res_")):
        destino = yt_dir / f"yt_{res}"
        if not resuelta(destino) and completa(bp_dir / res, TANDA_BP):
            return "elegir", ["--corrida", destino.name, "elegir", "--de", res, *excluir]
    ya = {f["clave"] for c in corridas for f in leer(yt_dir / c / "temas.csv")}
    for lote in con_lista(bp_dir):
        if not re.match(r"^lote_\d+$", lote) or lote in YA_RESCATADOS:
            continue
        nuevas = [f for f in bpv.filas_sin_resolver(bp_dir / lote) if f.get("con_boton") != "si" and f["clave"] not in ya]
        if len(nuevas) >= TANDA_YT or (nuevas and completa(bp_dir / lote, TANDA_BP)):
            return "elegir", ["--corrida", siguiente_nombre(yt_dir, f"yt_{lote}"), "elegir", "--de", lote,
                              "--solo-sin-boton", *excluir]
    return None


def tandas_locales(root: Path, carril: str, corrida: str) -> int:
    carpeta = root / ("beatport_previews" if carril == "beatport" else "youtube_temas") / corrida
    return len(list(carpeta.glob("tanda_*.json")))


MIN_CORRIDA_YT_S = 600


def nueva_pausa_yt(pausa: float, segundos_corridos: float) -> float:
    """La pausa entre pedidos sube (x1,5, tope 30 s) solo si la corrida bajó un rato antes de la verificación
    anti-robots. Si la pide al arrancar, es el mismo bloqueo de antes, no uno nuevo (2026-11-10 17:30: un minuto
    después de la espera de una hora; subirla igual la habría llevado al tope sin motivo)."""
    if segundos_corridos < MIN_CORRIDA_YT_S:
        return pausa
    return min(round(pausa * 1.5, 1), PAUSA_YT_TOPE)


def leer_pausa_yt(yt_dir: Path) -> float:
    try:
        return float((yt_dir / "pausa_s.txt").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return PAUSA_YT_INICIAL


def espera_corte(carril: str, n: int) -> float:
    """Segundos de espera tras el corte número n (1, 2, ...) seguido de un sitio."""
    if carril == "1001":
        return min(1 + n, 4) * 3600
    if carril == "youtube":
        return min(60 + 30 * (n - 1), 120) * 60
    return min(30 * 2 ** (n - 1), 120) * 60


def espera_error(n: int) -> float:
    return min(10 * 2 ** (n - 1), 60) * 60


def espera_pendiente(root: Path, carril: str) -> float:
    """Segundos que faltan de la última espera ante un sitio guardada para el carril (0 si no hay)."""
    try:
        return max(0.0, float((root / f"espera_{carril}.txt").read_text(encoding="utf-8").strip()) - time.time())
    except (OSError, ValueError):
        return 0.0


# ---------------------------------------------------------------- procesos


def vivo(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel = ctypes.windll.kernel32
    proceso = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION (os.kill mataría el proceso)
    if not proceso:
        return False
    codigo = ctypes.c_ulong()
    kernel.GetExitCodeProcess(proceso, ctypes.byref(codigo))
    kernel.CloseHandle(proceso)
    return codigo.value == 259  # STILL_ACTIVE


def mantener_despierto(activo: bool) -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if activo else 0))


def perfil_ocupado() -> bool:
    """¿Hay un Chromium abierto con el perfil de 1001Tracklists (otra corrida, o Gabriel)? Dos a la vez no pueden."""
    if os.name != "nt":
        return False
    consulta = ("(Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { $_.CommandLine -like "
                "'*1001tracklists_perfil*' } | Measure-Object).Count")
    try:
        salida = subprocess.run(["powershell", "-NoProfile", "-Command", consulta], capture_output=True, text=True,
                                timeout=120).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False
    return salida.isdigit() and int(salida) > 0


class Orquesta:
    def __init__(self, root: Path, python_navegador: Path, navegadores: Path):
        self.root, self.python_navegador, self.navegadores = root, python_navegador, navegadores
        self.log_ruta, self.parar_ruta = root / "descarga_continua.log", root / "PARAR"
        self.parar = threading.Event()
        self.ocioso = {"1001": False, "beatport": False, "youtube": False}
        self.sets_pendientes: Optional[int] = None
        self._hijos: Dict[str, subprocess.Popen] = {}
        self._cerrojo = threading.Lock()

    def log(self, carril: str, texto: str) -> None:
        linea = f"{time.strftime('%m-%d %H:%M:%S')} [{carril}] {texto}"
        with self._cerrojo:  # escribir el registro nunca tumba el programa (D: es un pendrive: 2026-11-11 13:28)
            try:
                with self.log_ruta.open("a", encoding="utf-8") as archivo:
                    archivo.write(linea + "\n")
            except OSError:
                pass
            try:
                print(linea, flush=True)
            except (OSError, ValueError):
                pass

    def esperar(self, segundos: float, carril: str = "", motivo: str = "") -> bool:
        """Espera sin bloquear la parada. False si hay que parar."""
        if motivo:
            self.log(carril, f"ESPERA {segundos / 60:.0f} min: {motivo} (hasta {time.strftime('%H:%M', time.localtime(time.time() + segundos))})")
        return not self.parar.wait(segundos)

    def esperar_sitio(self, carril: str, segundos: float, motivo: str) -> bool:
        """Una espera ante un sitio (tras un corte, o entre corridas de 1001Tracklists): queda en espera_<carril>.txt
        para que un reinicio no la acorte (2026-11-10: reiniciar a las 16:32 habría vuelto a pedirle a YouTube
        enseguida, con la verificación anti-robots de las 16:28 todavía encima)."""
        try:
            (self.root / f"espera_{carril}.txt").write_text(f"{time.time() + segundos:.0f}", encoding="utf-8")
        except OSError:
            pass
        return self.esperar(segundos, carril, motivo)

    def retomar_espera(self, carril: str) -> bool:
        falta = espera_pendiente(self.root, carril)
        return self.esperar(falta, carril, "sigue la espera de antes del reinicio") if falta > 0 else True

    def entorno(self, navegador: bool) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k != "TRAKTOR_REMOTE"}  # el remoto de verdad: erclm:traktor_ml
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        if navegador:
            env["PLAYWRIGHT_BROWSERS_PATH"] = str(self.navegadores)
        return env

    def correr(self, carril: str, script: str, argumentos: List[str], navegador: bool = False) -> int:
        python = str(self.python_navegador) if navegador else sys.executable
        comando = [python, SCRIPTS[script], "--root", str(self.root), *argumentos]
        self.log(carril, "$ " + " ".join([SCRIPTS[script], *argumentos]))
        banderas = (subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
        proceso = subprocess.Popen(comando, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", env=self.entorno(navegador), creationflags=banderas)
        with self._cerrojo:
            self._hijos[carril] = proceso
        for linea in proceso.stdout:
            if linea.strip():
                self.log(carril, linea.rstrip())
        codigo = proceso.wait()
        with self._cerrojo:
            self._hijos.pop(carril, None)
        self.log(carril, f"salida {codigo}")
        return codigo

    def cortar_todo(self) -> None:
        with self._cerrojo:
            hijos = list(self._hijos.values())
        for proceso in hijos:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proceso.pid)], capture_output=True)
            else:
                proceso.kill()

    def contar_sets_pendientes(self) -> Optional[int]:
        try:
            salida = subprocess.run([sys.executable, SCRIPTS["sets"], "--root", str(self.root), "bajar", "--n", "100000", "--dry-run"],
                                    cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    env=self.entorno(False), timeout=600).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        encontrado = re.search(r"bajo (\d+)", salida or "")
        return int(encontrado.group(1)) if encontrado else None


# ---------------------------------------------------------------- carriles


def vigilar(o: Orquesta, nombre: str, carril) -> None:
    """Un error inesperado en un carril no lo apaga: se anota y el carril vuelve a empezar (retoma por el disco)."""
    errores = 0
    while not o.parar.is_set():
        try:
            carril(o)
            return  # los carriles solo vuelven cuando hay que parar
        except Exception:
            errores += 1
            o.log(nombre, "ERROR inesperado del carril:\n" + traceback.format_exc())
            if not o.esperar(espera_error(errores), nombre, f"el carril vuelve a empezar ({errores} seguido/s)"):
                return


def comando_copia(root: Path, rclone: str, destino: str) -> List[str]:
    excluir = [x for patron in EXCLUIR_COPIA for x in ("--exclude", patron)]
    return [rclone, "copy", str(root), destino, *excluir, "--transfers", "8", "--checkers", "16", "-q"]


def copiar_al_drive(o: Orquesta) -> bool:
    from src.v4.common.remote_storage import find_rclone, remote_root

    env = o.entorno(False)
    destino = f"{remote_root(env)}/{COPIA_DRIVE}"
    t0 = time.monotonic()
    banderas = (subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    proceso = subprocess.run(comando_copia(o.root, str(find_rclone(env)), destino), capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=env, creationflags=banderas)
    if proceso.returncode == 0:
        o.log("copia", f"copia de {o.root} al Drive ({destino}) al día, {time.monotonic() - t0:.0f} s")
        return True
    o.log("copia", f"ERROR en la copia al Drive (salida {proceso.returncode}): {(proceso.stderr or '').strip()[-400:]}")
    return False


def carril_copia(o: Orquesta) -> None:
    while not o.parar.is_set():
        copiar_al_drive(o)
        if not o.esperar(COPIA_CADA_S):
            return


def carril_1001(o: Orquesta) -> None:
    cortes = errores = 0
    avisado = False
    if not o.retomar_espera("1001"):
        return
    while not o.parar.is_set():
        if perfil_ocupado():
            o.ocioso["1001"] = False
            if not avisado:
                o.log("1001", "el navegador con el perfil de 1001Tracklists está abierto: espero a que se cierre")
                avisado = True
            if not o.esperar(300):
                return
            continue
        avisado = False
        o.sets_pendientes = o.contar_sets_pendientes()
        tarea = tarea_1001(o.root, o.sets_pendientes)
        if tarea is None:
            o.ocioso["1001"] = o.sets_pendientes == 0
            if not o.esperar(900):
                return
            continue
        o.ocioso["1001"] = False
        tipo, argumentos = tarea
        if tipo == "sets":
            o.log("1001", f"sets pendientes: {o.sets_pendientes}")
            if cortes >= 2:  # por si una página en particular es la que falla
                argumentos = [*argumentos, "--saltar", str(3 * (cortes - 1))]
        codigo = o.correr("1001", "sets" if tipo == "sets" else "previews", argumentos, navegador=True)
        if o.parar.is_set():
            return
        if codigo == 0:
            cortes = errores = 0
            if not o.esperar_sitio("1001", 45 * 60, "pausa entre corridas de 1001Tracklists"):
                return
        elif codigo == 2:
            cortes += 1
            if not o.esperar_sitio("1001", espera_corte("1001", cortes), f"1001Tracklists cortó ({cortes} seguido/s)"):
                return
        else:
            errores += 1
            if not o.esperar(espera_error(errores), "1001", f"error del programa ({errores} seguido/s)"):
                return


def sin_avance(o: Orquesta, carril: str, tipo: str, argumentos: List[str], antes: int) -> bool:
    """Un bajar que terminó bien sin subir ninguna tanda nueva (p. ej., tandas en el remoto sin copia local): espera en
    vez de dar vueltas en falso. False si hay que parar."""
    if tipo == "bajar" and tandas_locales(o.root, carril, argumentos[1]) == antes:
        return o.esperar(1800, carril, f"«{' '.join(argumentos)}» terminó bien sin subir ninguna tanda nueva")
    return True


def carril_beatport(o: Orquesta) -> None:
    cortes = errores = 0
    bp_dir = o.root / "beatport_previews"
    if not o.retomar_espera("beatport"):
        return
    while not o.parar.is_set():
        n_sets = contar_sets_bajados(o.root)
        tarea = tarea_beatport(o.root, n_sets)
        if tarea is None:
            o.ocioso["beatport"] = True
            if not o.esperar(900):
                return
            continue
        o.ocioso["beatport"] = False
        tipo, argumentos = tarea
        antes = tandas_locales(o.root, "beatport", argumentos[1])
        codigo = o.correr("beatport", "previews", argumentos)
        if o.parar.is_set():
            return
        if tipo == "elegir" and codigo in (0, 3):
            (bp_dir / "sets_en_lotes.txt").write_text(str(n_sets), encoding="utf-8")
        if tipo == "rescate" and codigo == 3:
            marcar_vacia(bp_dir / argumentos[1])
        if codigo in (0, 3):
            cortes = errores = 0
            if codigo == 0 and not sin_avance(o, "beatport", tipo, argumentos, antes):
                return
            continue
        if codigo == 2:
            cortes += 1
            if not o.esperar_sitio("beatport", espera_corte("beatport", cortes), f"Beatport cortó ({cortes} seguido/s)"):
                return
        else:
            errores += 1
            if not o.esperar(espera_error(errores), "beatport", f"error del programa ({errores} seguido/s)"):
                return


def carril_youtube(o: Orquesta) -> None:
    yt_dir = o.root / "youtube_temas"
    pausa = leer_pausa_yt(yt_dir)
    cortes = errores = 0
    if not o.retomar_espera("youtube"):
        return
    while not o.parar.is_set():
        tarea = tarea_youtube(o.root)
        if tarea is None:
            o.ocioso["youtube"] = True
            if not o.esperar(900):
                return
            continue
        o.ocioso["youtube"] = False
        tipo, argumentos = tarea
        if tipo == "bajar":
            argumentos = [*argumentos, "--pausa", f"{pausa:g}"]
        antes = tandas_locales(o.root, "youtube", argumentos[1])
        t0 = time.monotonic()
        codigo = o.correr("youtube", "youtube", argumentos)
        if o.parar.is_set():
            return
        if tipo == "elegir" and codigo == 3:
            marcar_vacia(yt_dir / argumentos[1])
        if codigo in (0, 3):
            cortes = errores = 0
            if codigo == 0 and not sin_avance(o, "youtube", tipo, argumentos, antes):
                return
            continue
        if codigo == 2:
            corrida_s = time.monotonic() - t0
            # Seguidas son las que llegan sin haber bajado nada entre medio (2026-11-11 05:50: tras 8 h bajando, el
            # bloqueo contó como el segundo seguido del de las 20:58 y esperó 90 min en vez de 60).
            cortes = 1 if corrida_s >= MIN_CORRIDA_YT_S else cortes + 1
            nueva = nueva_pausa_yt(pausa, corrida_s)
            if nueva != pausa:
                pausa = nueva
                yt_dir.mkdir(parents=True, exist_ok=True)
                (yt_dir / "pausa_s.txt").write_text(f"{pausa:g}", encoding="utf-8")
                motivo = f"verificación anti-robots ({cortes} seguida/s); la pausa entre pedidos pasa a {pausa:g} s"
            else:
                motivo = f"verificación anti-robots ({cortes} seguida/s), todavía al arrancar: misma pausa ({pausa:g} s), espera más"
            if not o.esperar_sitio("youtube", espera_corte("youtube", cortes), motivo):
                return
        else:
            errores += 1
            if not o.esperar(espera_error(errores), "youtube", f"error del programa ({errores} seguido/s)"):
                return


# ---------------------------------------------------------------- comandos


def todo_hecho(o: Orquesta) -> bool:
    if not all(o.ocioso.values()) or o.sets_pendientes != 0:
        return False
    return (tarea_1001(o.root, 0) is None and tarea_youtube(o.root) is None
            and tarea_beatport(o.root, contar_sets_bajados(o.root)) is None)


def cmd_correr(args) -> int:
    pid_ruta = args.root / "descarga_continua.pid"
    try:
        otro = int(pid_ruta.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        otro = 0
    if otro and otro != os.getpid() and vivo(otro):
        print(f"Ya corre otra instancia (PID {otro}). Para pararla: crear {args.root / 'PARAR'}.")
        return 2
    o = Orquesta(args.root, args.python_navegador, args.navegadores)
    if o.parar_ruta.exists():
        o.parar_ruta.unlink()
    pid_ruta.write_text(str(os.getpid()), encoding="utf-8")
    mantener_despierto(True)
    o.log("main", f"EMPIEZA (PID {os.getpid()}); para parar: crear {o.parar_ruta}")
    hilos = [threading.Thread(target=vigilar, args=(o, n, f), name=n, daemon=True)
             for n, f in (("1001", carril_1001), ("beatport", carril_beatport), ("youtube", carril_youtube),
                          ("copia", carril_copia))]
    for hilo in hilos:
        hilo.start()
    fin = False
    try:
        while True:
            time.sleep(15)
            try:  # un error al mirar el disco (D: es un pendrive) no corta todo: se anota y se vuelve a mirar
                if o.parar_ruta.exists():
                    o.log("main", "PARAR: corto lo que esté corriendo; las tandas subidas quedan")
                    break
                caidos = [h.name for h in hilos if not h.is_alive()]
                if caidos:
                    o.log("main", f"un carril se cayó ({', '.join(caidos)}); ver la salida del proceso. Paro todo para que se note.")
                    break
                if todo_hecho(o):
                    o.log("main", "FIN: no queda nada por bajar")
                    fin = True
                    break
            except Exception:
                o.log("main", "ERROR inesperado al revisar el estado:\n" + traceback.format_exc())
    finally:
        o.parar.set()
        o.cortar_todo()
        for hilo in hilos:
            hilo.join(timeout=60)
        if fin:
            copiar_al_drive(o)  # la última, con todo terminado
            try:
                (args.root / FIN_MARCA).write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
            except OSError:
                pass
        mantener_despierto(False)
        try:
            pid_ruta.unlink()
        except OSError:
            pass
        o.log("main", "TERMINA")
    return 0


def que_hace_el_guardian(root: Path, esta_vivo) -> Tuple[str, int]:
    """('fin' | 'sin_disco' | 'pausa' | 'corre' | 'relanzar', PID que figura en descarga_continua.pid o 0)."""
    if (root / FIN_MARCA).exists():
        return "fin", 0
    if not root.exists():
        return "sin_disco", 0
    if (root / "PARAR").exists():
        return "pausa", 0
    try:
        pid = int((root / "descarga_continua.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid = 0
    return ("corre" if pid and esta_vivo(pid) else "relanzar"), pid


def cmd_guardian(args) -> int:
    """Relanza `correr` si se murió sin terminar (2026-11-11 13:28: el proceso murió sin dejar rastro, quizá por un
    corte del pendrive D:, y la descarga quedó parada 19 min). No relanza si existe PARAR (pausa de Gabriel: borrarlo
    reanuda) ni después de terminar (descarga_terminada.txt). Mira cada 5 min; su registro y la salida de `correr`
    van a C: (~/.traktor_ml), no al pendrive."""
    carpeta = NAVEGADOR
    carpeta.mkdir(parents=True, exist_ok=True)
    registro, propio = carpeta / "descarga_guardian.log", carpeta / "descarga_guardian.pid"

    def anotar(texto: str) -> None:
        try:
            with registro.open("a", encoding="utf-8") as archivo:
                archivo.write(f"{time.strftime('%m-%d %H:%M:%S')} {texto}\n")
        except OSError:
            pass

    try:
        otro = int(propio.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        otro = 0
    if otro and otro != os.getpid() and vivo(otro):
        print(f"Ya corre otro guardián (PID {otro}).")
        return 2
    propio.write_text(str(os.getpid()), encoding="utf-8")
    anotar(f"EMPIEZA el guardián (PID {os.getpid()}) sobre {args.root}")
    while True:
        try:
            decision, pid = que_hace_el_guardian(args.root, vivo)
            if decision == "fin":
                anotar("la descarga terminó: el guardián también")
                return 0
            if decision == "sin_disco":
                anotar(f"no veo {args.root} (¿el pendrive?): espero")
            elif decision == "relanzar":
                banderas = 0
                if os.name == "nt":
                    banderas = (subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
                                | subprocess.CREATE_NEW_PROCESS_GROUP)
                with (carpeta / "descarga_continua.salida.txt").open("a", encoding="utf-8") as salida, \
                        (carpeta / "descarga_continua.errores.txt").open("a", encoding="utf-8") as errores:
                    nuevo = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(args.root), "correr",
                                              "--python-navegador", str(args.python_navegador), "--navegadores", str(args.navegadores)],
                                             cwd=REPO_ROOT, stdout=salida, stderr=errores, creationflags=banderas)
                anotar(f"descarga_continua no corría (PID anterior {pid or '-'}): la relanzo, PID {nuevo.pid}")
        except Exception:
            anotar("ERROR del guardián:\n" + traceback.format_exc())
        time.sleep(args.cada)


def cmd_estado(args) -> int:
    root = args.root
    bp_dir, yt_dir = root / "beatport_previews", root / "youtube_temas"
    print(f"sets bajados: {contar_sets_bajados(root)}; usados en lotes: {sets_en_lotes(bp_dir)}")
    for carpeta, tam in ((bp_dir, TANDA_BP), (yt_dir, TANDA_YT)):
        for corrida in con_lista(carpeta):
            if not (ORIGEN.match(corrida) or corrida.startswith(("res_", "yt_"))):
                continue
            filas = leer(carpeta / corrida / "temas.csv")
            total = math.ceil(len(filas) / tam)
            hechas = len([p for p in (carpeta / corrida).glob("tanda_*.json")])
            extra = ""
            if corrida.startswith("res_"):
                extra = (f"; sin buscar {sum(not f.get('nivel_busqueda') for f in filas)}, "
                         f"enlaces pendientes {len(bpv.pendientes_de_enlace(filas, True))}")
            print(f"  {carpeta.name}/{corrida}: {len(filas)} temas, tandas {hechas} de {total}{extra}")
    print(f"pausa de YouTube: {leer_pausa_yt(yt_dir):g} s")
    pid_ruta = root / "descarga_continua.pid"
    pid = int(pid_ruta.read_text(encoding="utf-8").strip()) if pid_ruta.exists() else 0
    print(f"descarga_continua: {'corriendo, PID ' + str(pid) if pid and vivo(pid) else 'no corre'}")
    log = root / "descarga_continua.log"
    if log.exists():
        print("últimas líneas del registro:")
        for linea in log.read_text(encoding="utf-8", errors="replace").splitlines()[-args.lineas:]:
            print("  " + linea)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=bpv.DEFAULT_ROOT, help="carpeta de datos de sets (en este PC, en D:)")
    comandos = parser.add_subparsers(dest="comando", required=True)
    p = comandos.add_parser("correr")
    p.add_argument("--python-navegador", type=Path, default=NAVEGADOR / "pw_venv" / "Scripts" / "python.exe",
                   help="Python con Playwright (fuera del repo)")
    p.add_argument("--navegadores", type=Path, default=NAVEGADOR / "pw_browsers", help="PLAYWRIGHT_BROWSERS_PATH")
    p.set_defaults(func=cmd_correr)
    p = comandos.add_parser("guardian", help="relanza correr si se murió sin terminar (no si existe PARAR)")
    p.add_argument("--python-navegador", type=Path, default=NAVEGADOR / "pw_venv" / "Scripts" / "python.exe")
    p.add_argument("--navegadores", type=Path, default=NAVEGADOR / "pw_browsers")
    p.add_argument("--cada", type=float, default=300, help="segundos entre revisiones")
    p.set_defaults(func=cmd_guardian)
    p = comandos.add_parser("estado")
    p.add_argument("--lineas", type=int, default=15)
    p.set_defaults(func=cmd_estado)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
