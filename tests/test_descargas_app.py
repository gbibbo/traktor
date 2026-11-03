"""
PURPOSE: Tests de la app de descargas (download_JIJIJI/app/descargas_app.py): estado de cada tema a
         partir de los archivos que escribe el orquestador, historial, comando de la descarga,
         ajustes, seguridad del servidor local y seguimiento de procesos huérfanos. Sin red.
CHANGELOG:
  - 2026-10-30: Creación inicial.
  - 2026-11-02: La salida de cada descarga queda en <runs>/logs/. Enlace de autorización de Spotify
                (copiable), «Conectar» cuando ya está conectado y carpeta de destino que no existe.
                Avance de la búsqueda; grabación pospuesta (comando, estado e interruptor).
  - 2026-11-02: Pestañas Colección (carpetas, conteos, elegir para playlists, carpetas agregadas) y
                Organización (lanzar review_app, motivos de error, tarea en curso, cierre con la app).
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "download_JIJIJI" / "app" / "descargas_app.py"
SPEC = importlib.util.spec_from_file_location("descargas_app", APP_PATH)
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)

PID = "0TestPlaylist0Id000001"


def spotify(n: int) -> dict:
    return {"spotify_id": f"id{n}", "title": f"Tema {n}", "artists": [f"Artista {n}"], "album": "Disco",
            "cover_url": f"https://i.scdn.co/image/{n}", "duration_ms": 200_000 + n}


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_run(root: Path, stamp: str = "20261030-153352", pid: str = PID, status: str = "running") -> Path:
    run = root / f"{stamp}_{pid}"
    write(run / "orchestration_report.json",
          {"status": status, "playlist_id": pid, "source_playlist": f"https://open.spotify.com/playlist/{pid}"})
    return run


def availability(run: Path, statuses: list[str], name: str = "Mi playlist") -> None:
    write(run / "soulseek_availability.json", {
        "playlist": {"name": name},
        "tracks": [{"position": i, "spotify": spotify(i), "status": s, "error": "timeout" if s == "error" else None}
                   for i, s in enumerate(statuses, 1)],
    })


def by_pos(state: dict) -> dict[int, dict]:
    return {t["pos"]: t for t in state["tracks"]}


def test_preflight_running(tmp_path):
    run = make_run(tmp_path)
    state = APP.run_state(run)
    assert state["status"] == "running" and state["phase"] == "preflight"
    assert state["tracks"] == [] and state["playlist_id"] == PID
    assert state["search"] is None  # sockseek todavía no empezó
    write(run / "preflight_status.json", {"tracks": 152, "timeout_s": 2100, "started_at": 1, "searched": 40, "found": 31})
    assert APP.run_state(run)["search"] == {"searched": 40, "found": 31, "total": 152, "skipped": 0}


def test_transfer_states(tmp_path):
    run = make_run(tmp_path)
    availability(run, ["available", "available", "missing", "error"])
    write(run / "soulseek_missing.json", {"tracks": [{"playlist_position": 3}]})
    write(run / "spotify_reconcile_report.json",
          {"tracks": [{"position": 1, "status": "finalized", "output": "C:/x/Artista 1 - Tema 1.mp3"}]})
    write(run / "recorded_missing" / "recording_progress.json", {"completed_count": 0, "tracks": []})
    write(run / "recorder_ready.json", {"status": "ready"})
    (run / "start_soulseek_downloads.signal").write_text("start\n", encoding="utf-8")

    state = APP.run_state(run)
    tracks = by_pos(state)
    assert state["phase"] == "transfer" and state["name"] == "Mi playlist"
    assert tracks[1]["state"] == "downloaded" and tracks[1]["detail"] == "Artista 1 - Tema 1.mp3"
    assert tracks[2]["state"] == "downloading"
    assert tracks[3]["state"] == "recording"
    assert tracks[4]["state"] == "to_record"          # la búsqueda falló: se graba al final
    assert state["counts"]["downloaded"] == 1 and state["counts"]["recording"] == 1
    assert state["recording_since"] is not None


def test_undownloaded_tracks_are_recorded_after_the_download(tmp_path):
    run = make_run(tmp_path)
    availability(run, ["available", "available", "available", "missing"])
    write(run / "soulseek_missing.json", {"tracks": [{"playlist_position": 4}]})
    write(run / "recorded_missing" / "recording_progress.json",
          {"completed_count": 1, "tracks": [{"playlist_position": 4}], "source_json": str(run / "soulseek_missing.json")})
    write(run / "spotify_reconcile_report.json", {"tracks": [
        {"position": 1, "status": "finalized"},
        {"position": 2, "status": "verification_failed",
         "candidates": [{"accepted": False, "reasons": ["artista mismatch (0.5)"]}]},
        {"position": 3, "status": "download_error", "error": "timeout"}]})
    tracks = by_pos(APP.run_state(run, alive=True))
    assert tracks[2]["state"] == "to_record" and "al terminar las descargas" in tracks[2]["detail"]

    write(run / "soulseek_failed.json", {"tracks": [{"playlist_position": 2}, {"playlist_position": 3}]})
    write(run / "recorded_failed" / "recording_progress.json",
          {"completed_count": 1, "tracks": [{"playlist_position": 2}], "source_json": str(run / "soulseek_failed.json")})
    write(run / "recorder_ready.json", {"status": "ready"})
    tracks = by_pos(APP.run_state(run, alive=True))
    assert [tracks[p]["state"] for p in (1, 2, 3, 4)] == ["downloaded", "recorded", "recording", "recorded"]


def test_partial_run_shows_what_is_missing(tmp_path):
    run = make_run(tmp_path, status="partial")
    write(run / "orchestration_report.json", {"status": "partial", "reason": "Spotify falló", "final_output": "D:/m/Lista"})
    diag = run / "diagnostics"
    write(diag / "soulseek_availability.json", {"playlist": {"name": "Lista"}, "tracks": [
        {"position": 1, "spotify": spotify(1), "status": "available"},
        {"position": 2, "spotify": spotify(2), "status": "available"}]})
    write(run / "soulseek_failed.json", {"tracks": [{"playlist_position": 2}]})
    write(run / "consolidation_report.json", {"partial": True, "playlist_directory": "D:/m/Lista", "audio_count": 1,
                                              "missing_spotify_ids": ["id2"], "tracks": [
        {"spotify_id": "id1", "origin": "soulseek", "destination": "D:/m/Lista/a.mp3"}]})
    state = APP.run_state(run)
    assert state["status"] == "partial" and state["reason"] == "Spotify falló"
    assert [t["state"] for t in state["tracks"]] == ["downloaded", "failed"]
    assert state["counts"]["failed"] == 1 and state["final_output"] == "D:/m/Lista"


def test_current_download_from_log_wins(tmp_path):
    run = make_run(tmp_path)
    availability(run, ["available", "available", "available"])
    state = APP.run_state(run, alive=True, current_download=3)
    assert by_pos(state)[3]["state"] == "downloading"
    assert by_pos(state)[1]["state"] == "queued"


def test_failed_run_explains_each_track(tmp_path):
    run = make_run(tmp_path, status="failed")
    availability(run, ["available", "available", "missing"])
    write(run / "soulseek_missing.json", {"tracks": [{"playlist_position": 3}]})
    write(run / "spotify_reconcile_report.json", {"tracks": [
        {"position": 1, "status": "verification_failed",
         "candidates": [{"accepted": False, "reasons": ["ISRC distinto: local=A Spotify=B"]}]},
        {"position": 2, "status": "verification_failed",
         "candidates": [{"accepted": False, "reasons": ["duración mismatch (-27.3s)"]}]},
    ]})
    write(run / "recorded_missing" / "recording_progress.json",
          {"completed_count": 1, "tracks": [{"playlist_position": 3, "output": "x.mp3"}]})
    state = APP.run_state(run)
    tracks = by_pos(state)
    assert state["status"] == "failed" and state["phase"] is None
    assert tracks[1]["state"] == "failed" and "ISRC distinto" in tracks[1]["detail"]
    assert "duración distinta (-27 s)" in tracks[2]["detail"]
    assert tracks[3]["state"] == "recorded"
    assert state["saved_to"] is None


def test_partial_save_marks_tracks(tmp_path):
    run = make_run(tmp_path, status="failed")
    availability(run, ["available", "available"])
    write(run / "spotify_reconcile_report.json", {"tracks": [{"position": 1, "status": "finalized"},
                                                             {"position": 2, "status": "download_error"}]})
    write(run / "consolidation_report.json", {"partial": True, "playlist_directory": "D:/Musica/Mi playlist",
                                              "audio_count": 1, "tracks": [
        {"spotify_id": "id1", "origin": "soulseek", "destination": "D:/Musica/Mi playlist/Artista 1 - Tema 1.mp3"}]})
    state = APP.run_state(run)
    assert state["saved_to"] == "D:/Musica/Mi playlist" and state["saved_count"] == 1
    assert by_pos(state)[1]["state"] == "downloaded" and by_pos(state)[2]["state"] == "failed"


def test_completed_run_reads_diagnostics(tmp_path):
    run = make_run(tmp_path, status="completed")
    diag = run / "diagnostics"
    write(diag / "soulseek_availability.json", {"playlist": {"name": "Lista"}, "tracks": [
        {"position": 1, "spotify": spotify(1), "status": "available"},
        {"position": 2, "spotify": spotify(2), "status": "missing"}]})
    write(run / "consolidation_report.json", {"tracks": [
        {"spotify_id": "id1", "origin": "soulseek", "destination": "D:/m/Lista/a.mp3"},
        {"spotify_id": "id2", "origin": "loopback", "destination": "D:/m/Lista/b.mp3"}]})
    state = APP.run_state(run)
    assert state["status"] == "done"
    assert [t["state"] for t in state["tracks"]] == ["downloaded", "recorded"]


def test_stale_running_run_counts_as_stopped(tmp_path):
    run = make_run(tmp_path)
    availability(run, ["available"])
    old = time.time() - 3 * 3600
    for path in run.rglob("*"):
        os.utime(path, (old, old))
    assert APP.run_state(run)["status"] == "stopped"
    assert APP.run_state(run, alive=True)["status"] == "running"


def test_history_keeps_latest_run_per_playlist(tmp_path):
    make_run(tmp_path, "20261029-100000", status="failed")
    make_run(tmp_path, "20261030-100000", status="completed")
    make_run(tmp_path, "20261030-120000", pid="otraPlaylist1", status="resumable")
    (tmp_path / "no_es_un_run").mkdir()
    items = APP.history(tmp_path, {PID: {"name": "Cacheada", "cover": "https://i.scdn.co/image/x"}})
    assert [(i["playlist_id"], i["status"]) for i in items] == [("otraPlaylist1", "paused"), (PID, "done")]
    assert items[1]["name"] == "Cacheada"


def test_failure_reasons_are_readable():
    row = {"status": "verification_failed", "candidates": [
        {"accepted": False, "reasons": ["título base mismatch (0.320)"]},
        {"accepted": False, "reasons": ["duración mismatch (+41.0s)"]},
        {"accepted": False, "reasons": ["título mismatch (0.700)"]}]}
    assert APP.friendly_failure(row) == ("El archivo de Soulseek no pasó la verificación: "
                                         "el título no coincide, duración distinta (+41 s)")


def test_an_unreadable_run_does_not_empty_the_history(tmp_path, monkeypatch):
    good = make_run(tmp_path, stamp="20261030-100000", pid="PlaylistBuena", status="completed")
    make_run(tmp_path, stamp="20261031-100000", pid="PlaylistBuena", status="completed")  # la más nueva, ilegible
    real = APP.run_state

    def run_state(run_dir, **kwargs):
        if run_dir.name.startswith("20261031"):
            raise ValueError("could not convert string to float: 'base '")
        return real(run_dir, **kwargs)
    monkeypatch.setattr(APP, "run_state", run_state)
    assert [item["run"] for item in APP.history(tmp_path, {})] == [good.name]  # se ve la anterior


def app_with_runs(tmp_path):
    write(tmp_path / "cfg" / "settings.json", {"library_root": str(tmp_path / "lib"), "runs_root": str(tmp_path / "runs")})
    return APP.App(tmp_path / "cfg")


def test_a_playlist_can_be_hidden_and_comes_back_when_downloaded_again(tmp_path):
    app = app_with_runs(tmp_path)
    make_run(tmp_path / "runs", stamp="20261030-100000", status="completed")
    assert [it["playlist_id"] for it in app.history()] == [PID]
    app.hide_playlist(PID, True)
    assert app.history() == [] and (tmp_path / "runs").exists()  # no se borra nada
    assert APP.App(tmp_path / "cfg").history() == []                 # sobrevive a reabrir la app
    app.hide_playlist(PID, False)
    assert [it["playlist_id"] for it in app.history()] == [PID]
    app.hide_playlist(PID, True)
    app.hidden[PID] = time.mktime(time.strptime("20261031", "%Y%m%d"))
    make_run(tmp_path / "runs", stamp="20261101-100000", status="running")  # se baja de nuevo
    assert [it["run"] for it in app.history()] == [f"20261101-100000_{PID}"]
    with pytest.raises(ValueError):
        app.hide_playlist("../x", True)


def test_deleting_work_files_warns_and_keeps_the_music(tmp_path):
    app = app_with_runs(tmp_path)
    run = make_run(tmp_path / "runs", stamp="20261030-100000", status="interrupted")
    (run / "soulseek").mkdir()
    (run / "soulseek" / "Artista - Tema.mp3").write_bytes(b"x" * 2048)
    music = tmp_path / "lib" / "Mi playlist" / "Artista - Otro.mp3"
    music.parent.mkdir(parents=True)
    music.write_bytes(b"y")
    other = make_run(tmp_path / "runs", stamp="20261030-100000", pid="OtraPlaylist01", status="completed")
    info = app.playlist_work(PID)
    assert (info["runs"], info["unsaved"], info["pending_recording"]) == (1, 1, False)
    app.job = type("Job", (), {"alive": True, "run_dir": run, "playlist_id": PID})()
    with pytest.raises(RuntimeError, match="se está descargando"):
        app.delete_playlist_work(PID)
    app.job = None
    assert app.delete_playlist_work(PID) == 1
    assert not run.exists() and other.exists() and music.exists()


def test_a_stopped_run_can_save_what_it_downloaded(tmp_path, monkeypatch):
    app = app_with_runs(tmp_path)
    run = make_run(tmp_path / "runs", status="interrupted")
    availability(run, ["available"])
    seen = {}

    def consolidate(run_dir, *args, **kwargs):
        seen["run"] = run_dir
        return tmp_path / "lib" / "Mi playlist", {}, {"audio_count": 1, "tracks": []}
    monkeypatch.setattr(APP.orch, "consolidate_audio", consolidate)
    assert app.save_partial(run.name)["status"] == "running"
    deadline = time.time() + 5
    while app.saving["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
    assert app.saving["status"] == "done" and seen["run"] == run


def test_the_log_of_any_run_can_be_read(tmp_path):
    app = app_with_runs(tmp_path)
    run = make_run(tmp_path / "runs", stamp="20261102-181456")
    other = make_run(tmp_path / "runs", stamp="20261102-174256")
    logs = tmp_path / "runs" / APP.JOB_LOG_DIR
    logs.mkdir()
    (logs / f"20261102-181455_{PID}.log").write_text(f"[18:14:56] Run      : {run}\n[18:16:38] Lanzando…\n", encoding="utf-8")
    (logs / f"20261102-174255_{PID}.log").write_text(f"[17:42:56] Run      : {other}\n", encoding="utf-8")
    (run / "sockseek_preflight.log").write_text("2026-11-02 18:16:40 [info] searching\n", encoding="utf-8")
    assert app.run_log(run.name, "job")["lines"][1] == "[18:16:38] Lanzando…"   # el de ese run, no el de otro
    assert app.run_log(other.name, "job")["lines"] == [f"[17:42:56] Run      : {other}"]
    assert app.run_log(run.name, "sockseek")["lines"] == ["2026-11-02 18:16:40 [info] searching"]
    assert app.run_log(other.name, "sockseek") == {"lines": [], "path": None}
    with pytest.raises(ValueError):
        app.run_log("../x", "job")


class FakeJob:
    def __init__(self, pid=PID, outcome="done", paused=None, stop_requested=False):
        self.playlist_id, self.alive, self.paused, self.stop_requested, self.error = pid, False, paused, stop_requested, None
        self.run_dir = None
        self._outcome = outcome

    def public(self):
        return {"outcome": self._outcome}


def test_queued_playlists_start_one_after_another(tmp_path, monkeypatch):
    app = app_with_runs(tmp_path)
    started, notes = [], []
    monkeypatch.setattr(APP, "notify", lambda title, text: notes.append(title))

    def start(url, fresh, defer):
        started.append((url.rsplit("/", 1)[-1], fresh, defer))
        app.job = FakeJob(pid=url.rsplit("/", 1)[-1])
        app.job.alive = True
        return {}
    monkeypatch.setattr(app, "start", start)
    app.enqueue("https://open.spotify.com/playlist/AAA", False, True)  # nada en curso: empieza ya
    assert started == [("AAA", False, True)] and app.queue == []
    app.enqueue("https://open.spotify.com/playlist/BBB", True, False)
    app.enqueue("https://open.spotify.com/playlist/CCC", False, False)
    assert [q["playlist_id"] for q in app.public_queue()] == ["BBB", "CCC"] and app.busy()
    with pytest.raises(RuntimeError, match="ya está en la cola"):
        app.enqueue("https://open.spotify.com/playlist/BBB", False, False)
    with pytest.raises(RuntimeError, match="ya se está descargando"):
        app.enqueue("https://open.spotify.com/playlist/AAA", False, False)
    app.dequeue("CCC")
    app.job.alive = False
    app._job_event(app.job, "end")  # terminó AAA: avisa y empieza BBB
    deadline = time.time() + 5
    while len(started) < 2 and time.time() < deadline:
        time.sleep(0.02)
    assert started[1] == ("BBB", True, False) and notes == ["Playlist lista"] and app.queue == []


def test_windows_notices_say_what_happened(tmp_path, monkeypatch):
    app = app_with_runs(tmp_path)
    app.playlists = {PID: {"name": "Mi playlist"}}
    notes = []
    monkeypatch.setattr(APP, "notify", lambda title, text: notes.append((title, text)))
    monkeypatch.setattr(app, "_start_next", lambda: None)
    app._job_event(FakeJob(outcome="paused", paused="Grabación pospuesta: 20 tema(s) por grabar"), "end")
    app._job_event(FakeJob(outcome="paused", paused="Spotify se detuvo"), "end")
    app._job_event(FakeJob(outcome="stopped", stop_requested=True), "end")  # la detuvo Gabriel: sin aviso
    app._job_event(FakeJob(), "auth")
    assert [t for t, _ in notes] == ["Grabación pendiente", "Descarga en pausa", "Spotify pide autorización"]
    assert "«Mi playlist»" in notes[0][1]


def test_a_stopped_download_does_not_start_the_queue(tmp_path, monkeypatch):
    app = app_with_runs(tmp_path)
    monkeypatch.setattr(APP, "notify", lambda *a: None)
    app.queue = [{"url": "https://open.spotify.com/playlist/BBB", "playlist_id": "BBB", "fresh": False, "defer": False}]
    started = []
    monkeypatch.setattr(app, "start", lambda *a: started.append(a))
    app._job_event(FakeJob(outcome="stopped", stop_requested=True), "end")
    time.sleep(0.2)
    assert started == [] and len(app.queue) == 1  # espera a «Empezar la cola ahora»
    app._start_next()
    assert started == [("https://open.spotify.com/playlist/BBB", False, False)] and app.queue == []


def test_a_run_stopped_before_searching_does_not_hide_the_previous_one(tmp_path):
    earlier = make_run(tmp_path, stamp="20261102-181456", status="interrupted")
    availability(earlier, ["available", "missing"])
    make_run(tmp_path, stamp="20261102-214736", status="interrupted")  # detenida a los 24 s, sin buscar
    assert [it["run"] for it in APP.history(tmp_path, {})] == [earlier.name]
    live = make_run(tmp_path, stamp="20261102-220000", status="running")  # una en curso sí se muestra
    assert [it["run"] for it in APP.history(tmp_path, {}, alive_run=live.name)] == [live.name]


def test_job_command(tmp_path):
    settings = {"library_root": "D:/Musica", "runs_root": "D:/runs"}
    cmd = APP.build_job_command(tmp_path / "stop", "https://open.spotify.com/playlist/abc", settings, fresh=False)
    assert cmd[2].endswith("run_job.py") and "--fresh" not in cmd
    assert cmd[cmd.index("--library-root") + 1] == "D:/Musica" and cmd[cmd.index("--output-root") + 1] == "D:/runs"
    assert APP.build_job_command(tmp_path / "stop", "u", settings, fresh=True)[-1] == "--fresh"
    assert APP.build_job_command(tmp_path / "stop", "u", settings, False, True)[-1] == "--defer-recording"


def test_postponed_recording_run(tmp_path):
    run = make_run(tmp_path, status="recording_deferred")
    report = json.loads((run / "orchestration_report.json").read_text(encoding="utf-8"))
    write(run / "orchestration_report.json", {**report, "pending_recordings": 2, "final_output": "D:/Musica/Mi playlist"})
    availability(run, ["available", "available", "missing"])
    write(run / "consolidation_report.json", {"tracks": [{"spotify_id": "id1", "origin": "soulseek",
                                                          "destination": "D:/Musica/Mi playlist/Tema 1.mp3"}]})
    write(run / "soulseek_missing.json", {"missing_count": 1, "tracks": [{"playlist_position": 3}]})
    state = APP.run_state(run)
    assert (state["status"], state["pending_recordings"], state["defer"]) == ("deferred", 2, False)
    assert by_pos(state)[1]["state"] == "downloaded" and by_pos(state)[3]["state"] == "to_record"


def test_recording_can_be_postponed_during_the_download(tmp_path):
    run = make_run(tmp_path)
    job = APP.DownloadJob(f"https://open.spotify.com/playlist/{PID}", {"library_root": "x", "runs_root": "y"}, False)
    with pytest.raises(RuntimeError, match="empezando"):
        job.set_defer(True)  # todavía no se sabe en qué carpeta trabaja
    job.run_dir = run
    job.set_defer(True)
    assert (run / APP.orch.DEFER_SIGNAL).exists() and APP.run_state(run, alive=True)["defer"] is True
    job.set_defer(False)
    assert not (run / APP.orch.DEFER_SIGNAL).exists() and APP.run_state(run, alive=True)["defer"] is False


def test_job_output_is_kept_in_a_log(tmp_path, monkeypatch):
    script = "print('Buscando en Soulseek…'); print('ERROR: algo falló'); raise SystemExit(3)"
    monkeypatch.setattr(APP, "build_job_command", lambda *args: [sys.executable, "-c", script])
    settings = {"library_root": str(tmp_path / "lib"), "runs_root": str(tmp_path / "runs")}
    job = APP.DownloadJob(f"https://open.spotify.com/playlist/{PID}", settings, fresh=False)
    job.start()
    deadline = time.time() + 30
    while job.alive and time.time() < deadline:
        time.sleep(0.05)
    assert job.returncode == 3 and job.error == "algo falló"
    assert job.log_path is not None and job.log_path.parent == tmp_path / "runs" / APP.JOB_LOG_DIR
    assert job.log_path.name.endswith(f"_{PID}.log")
    text = job.log_path.read_text(encoding="utf-8")
    assert f"Playlist : https://open.spotify.com/playlist/{PID}" in text
    assert "] Buscando en Soulseek…" in text and "] ERROR: algo falló" in text and "Fin: código de salida 3" in text
    assert APP.list_runs(tmp_path / "runs") == []  # la carpeta de logs no es un run


def run_fake_job(tmp_path, monkeypatch, script: str, settings: dict | None = None):
    monkeypatch.setattr(APP, "build_job_command", lambda *args: [sys.executable, "-c", script])
    settings = settings or {"library_root": str(tmp_path / "lib"), "runs_root": str(tmp_path / "runs")}
    job = APP.DownloadJob(f"https://open.spotify.com/playlist/{PID}", settings, fresh=False)
    job.start()
    deadline = time.time() + 30
    while job.alive and time.time() < deadline:
        time.sleep(0.05)
    return job


AUTH_LINES = "print('Abre esta URL si el navegador no se abre solo:\\n', 'https://accounts.spotify.com/authorize?x=1')"


def test_job_auth_link_is_shown_until_spotify_answers(tmp_path, monkeypatch):
    job = run_fake_job(tmp_path, monkeypatch, AUTH_LINES)
    assert job.auth_url == "https://accounts.spotify.com/authorize?x=1"
    job = run_fake_job(tmp_path, monkeypatch, AUTH_LINES + "; print('Leyendo la playlist en Spotify: 50/152 temas…')")
    assert job.auth_url is None


def test_missing_destination_is_explained(tmp_path, monkeypatch):
    blocker = tmp_path / "archivo"
    blocker.write_text("x", encoding="utf-8")
    settings = {"library_root": str(blocker / "musica"), "runs_root": str(tmp_path / "runs")}
    job = APP.DownloadJob(f"https://open.spotify.com/playlist/{PID}", settings, fresh=False)
    with pytest.raises(RuntimeError, match="No se puede usar la carpeta de destino"):
        job.start()
    assert job.proc is None


def test_spotify_connect_says_whether_a_browser_is_needed(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "test-client")
    token = tmp_path / "token.json"
    write(token, {"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 600,
                  "scope": "playlist-read-private playlist-read-collaborative playlist-modify-private"})
    preview = APP.SpotifyPreview(token)
    assert preview.connect() is False and not preview.connecting  # ya conectado: no se abre nada

    token.unlink()
    release = threading.Event()

    class FakeClient:
        _token = None

        def _load(self):
            return None

        def token(self):  # como dl.Spotify._login: avisa el enlace y espera la autorización
            preview._set_auth_url("https://accounts.spotify.com/authorize?y=2")
            release.wait(10)
            return "nuevo"

    preview._client = FakeClient()
    assert preview.connect() is True
    deadline = time.time() + 5
    while preview.auth_url is None and time.time() < deadline:
        time.sleep(0.02)
    assert preview.connecting and preview.auth_url == "https://accounts.spotify.com/authorize?y=2"
    release.set()
    while preview.connecting and time.time() < deadline + 5:
        time.sleep(0.02)
    assert not preview.connecting and preview.auth_url is None


def test_settings_merge_saved_values(tmp_path):
    write(tmp_path / "settings.json", {"library_root": "D:/Musica", "runs_root": "", "otra": 1})
    settings = APP.load_settings(tmp_path / "settings.json")
    assert settings["library_root"] == "D:/Musica"
    assert settings["runs_root"] == APP.default_settings()["runs_root"]


@pytest.fixture()
def server(tmp_path):
    write(tmp_path / "cfg" / "settings.json", {"library_root": str(tmp_path / "lib"), "runs_root": str(tmp_path / "runs")})
    app = APP.App(tmp_path / "cfg")
    app.port = APP.free_port(18790)
    httpd = ThreadingHTTPServer(("127.0.0.1", app.port), APP.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app
    httpd.shutdown()
    httpd.server_close()


def request(app, path, body=None, token=None, host=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{app.port}{path}", data=data, method="POST" if data else "GET")
    if host:
        req.add_header("Host", host)
    if token:
        req.add_header("X-App-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_server_security_and_settings(server, tmp_path):
    status, page = request(server, "/")
    assert status == 200 and server.token in page
    assert request(server, "/api/state", host="evil.example:80")[0] == 403
    assert request(server, "/api/stop", body={})[0] == 403                       # sin token
    assert request(server, "/api/stop", body={}, token="otro")[0] == 403
    status, body = request(server, "/api/settings", body={"library_root": "relativa"}, token=server.token)
    assert status == 400
    target = tmp_path / "nueva"
    status, body = request(server, "/api/settings", body={"library_root": str(target)}, token=server.token)
    assert status == 200 and target.is_dir()
    assert json.loads((tmp_path / "cfg" / "settings.json").read_text(encoding="utf-8"))["library_root"] == str(target.resolve())
    status, body = request(server, "/api/open-folder", body={"path": "C:/Windows"}, token=server.token)
    assert status == 400                                                          # fuera de las carpetas de descargas
    assert request(server, "/api/run?id=../../etc")[0] == 400


def mp3_with_tags(path: Path, spotify_id: str, title: str) -> Path:
    """Un MP3 de verdad (frames MPEG-1 Layer III de 128 kbps, ~1,3 s) con etiquetas y carátula."""
    from mutagen.id3 import APIC, ID3, TIT2, TPE1, TSRC, TXXX
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((b"\xff\xfb\x90\x64" + b"\x00" * 413) * 50)
    cover = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(cover, "PNG")
    tags = ID3()
    tags.add(TIT2(encoding=3, text=[title]))
    tags.add(TPE1(encoding=3, text=["Artista 1"]))
    tags.add(TSRC(encoding=3, text=["USRC10000001"]))
    tags.add(TXXX(encoding=3, desc="Spotify Track ID", text=[spotify_id]))
    tags.add(APIC(encoding=3, mime="image/png", type=3, desc="", data=cover.getvalue()))
    tags.save(path, v2_version=3)
    return path


def track_run(tmp_path: Path) -> tuple[Path, Path]:
    """Run con el tema 1 ya en la carpeta final (bajado de Soulseek) y el 2 sin archivo."""
    run = make_run(tmp_path / "runs", status="completed")
    rows = [{"position": i, "spotify": {**spotify(i), "isrc": "USRC10000001" if i == 1 else None},
             "status": "available", "top_candidates": [{"username": "peer1", "filename": "música\\Tema 1 (orig).flac"}]}
            for i in (1, 2)]
    write(run / "soulseek_availability.json", {"playlist": {"name": "Mi playlist"}, "tracks": rows})
    final = mp3_with_tags(tmp_path / "lib" / "Mi playlist" / "Artista 1 - Tema 1.mp3", "id1", "Tema 1")
    write(run / "consolidation_report.json", {"tracks": [{"spotify_id": "id1", "origin": "soulseek", "destination": str(final)}]})
    write(run / "spotify_reconcile_report.json", {"tracks": [{
        "position": 1, "status": "finalized", "output": str(final),
        "selected": {"path": "C:/tmp/verified_x/Tema 1 (orig).flac", "extended": False, "reasons": ["ISRC exacto", "base=1.000"]},
        "candidates": [
            {"path": "C:/tmp/verified_y/Tema 1 (Remix).flac", "accepted": False, "reasons": ["versión conflictiva: remix"]},
            {"path": "C:/tmp/verified_x/Tema 1 (orig).flac", "accepted": True, "format": "flac", "bitrate_kbps": 900.0,
             "length_s": 200.0, "tag_title": "Tema 1", "tag_artists": ["Artista 1"], "isrc": "USRC10000001"}]}]})
    return run, final


def test_downloaded_track_can_be_inspected(tmp_path):
    run, final = track_run(tmp_path)
    write(tmp_path / "cfg" / "settings.json", {"library_root": str(tmp_path / "lib"), "runs_root": str(tmp_path / "runs")})
    app = APP.App(tmp_path / "cfg")
    tracks = by_pos(APP.run_state(run))
    assert tracks[1]["file"] == str(final) and tracks[2]["file"] is None
    detail = app.track_detail(run.name, 1)
    checks = {c["label"]: c for c in detail["checks"]}
    assert checks["Título (origen)"]["ok"] and checks["Artistas (origen)"]["ok"] and checks["ISRC (origen)"]["ok"]
    assert checks["Spotify Track ID (final)"]["ok"] and checks["Duración"]["ok"] is False  # 1,3 s frente a 200 s
    assert detail["source"]["user"] == "peer1" and detail["source"]["rejected"][0]["file"] == "Tema 1 (Remix).flac"
    tags = {t["label"]: t["value"] for t in detail["tags"]}
    assert tags["Título"] == "Tema 1" and tags["ISRC"] == "USRC10000001" and tags["Carátula"].startswith("image/png, 32×24")
    props = {p["label"]: p["value"] for p in detail["properties"]}
    assert props["Formato"] == "MP3" and props["Bitrate"].startswith("128 kbps") and props["Frecuencia"] == "44100 Hz"
    assert APP.embedded_cover(final)[1] == "image/png"
    other = app.track_detail(run.name, 2)  # sin archivo: lo que encontró Soulseek
    assert other["path"] is None and other["search"]["status"] == "available"
    assert other["search"]["candidates"][0] == {"file": "Tema 1 (orig).flac", "folder": "música", "user": "peer1",
                                                "length_s": None, "format": None, "bitrate_kbps": None, "size_mb": None,
                                                "rejection_reason": None}
    write(run / "spotify_reconcile_report.json", {"tracks": [
        *json.loads((run / "spotify_reconcile_report.json").read_text(encoding="utf-8"))["tracks"],
        {"position": 2, "status": "verification_failed",
         "candidates": [{"path": "C:/tmp/x/Tema 2 (Dub Mix).flac", "accepted": False, "reasons": ["versión conflictiva: dub"]}]}]})
    attempts = app.track_detail(run.name, 2)["attempts"]
    assert attempts["summary"].startswith("El archivo de Soulseek no pasó la verificación")
    assert attempts["candidates"] == [{"file": "Tema 2 (Dub Mix).flac", "accepted": False, "reasons": ["versión conflictiva: dub"]}]
    report = json.loads((run / "consolidation_report.json").read_text(encoding="utf-8"))
    report["tracks"][0]["destination"] = str(mp3_with_tags(tmp_path / "otra" / "x.mp3", "id1", "Tema 1"))
    write(run / "consolidation_report.json", report)
    with pytest.raises(ValueError, match="fuera de las carpetas"):
        app.track_detail(run.name, 1)  # el informe no puede servir archivos de cualquier lado


def test_playback_is_blocked_while_a_download_may_record(tmp_path):
    write(tmp_path / "cfg" / "settings.json", {"library_root": str(tmp_path / "lib"), "runs_root": str(tmp_path / "runs")})
    app = APP.App(tmp_path / "cfg")
    assert app.playback_block_reason() is None
    run = make_run(tmp_path / "runs")
    app.job = type("Job", (), {"alive": True, "run_dir": run})()
    assert "grabando" in app.playback_block_reason()
    (run / APP.orch.DEFER_SIGNAL).write_text("1", encoding="utf-8")
    assert app.playback_block_reason() is None  # grabación pospuesta: se puede escuchar


def test_track_audio_is_served_with_ranges(server, tmp_path):
    run, final = track_run(tmp_path)
    query = f"run={run.name}&pos=1"
    assert request(server, f"/api/track/audio?{query}")[0] == 403  # sin token en la URL
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}/api/track/audio?{query}&t={server.token}",
                                 headers={"Range": "bytes=0-99"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 206 and resp.headers["Content-Range"] == f"bytes 0-99/{final.stat().st_size}"
        assert resp.read() == final.read_bytes()[:100]
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/track/cover?{query}&t={server.token}", timeout=10) as resp:
        assert resp.status == 200 and resp.headers["Content-Type"] == "image/png" and resp.read()[:4] == b"\x89PNG"
    server.job = type("Job", (), {"alive": True, "run_dir": None})()  # una descarga que puede grabar
    assert request(server, f"/api/track/audio?{query}&t={server.token}")[0] == 409


@pytest.mark.skipif(os.name != "nt", reason="árbol de procesos de Windows")
def test_process_tree_closes_orphans():
    child = subprocess.Popen([sys.executable, "-c",
                              "import subprocess,sys,time; "
                              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(2)"])
    tree = APP.ProcessTree(child.pid)
    deadline = time.time() + 10
    while time.time() < deadline and not tree.seen:
        time.sleep(0.3)
        tree.scan()
    time.sleep(1)
    tree.scan()
    child.wait(timeout=20)
    orphans = [p for p, (_n, created) in tree.seen.items() if created is not None and APP.process_created(p) == created]
    assert orphans, "el nieto tendría que seguir vivo después de que muere su padre"
    tree.kill_leftovers(APP.keep_spotify)
    time.sleep(0.5)
    assert not [p for p in orphans if APP.process_created(p) is not None]
    assert APP.keep_spotify("Spotify.exe") and not APP.keep_spotify("sockseek.exe")


def test_search_failed_for_every_track(tmp_path):
    run = make_run(tmp_path, status="resumable")
    write(run / "soulseek_availability.json", {"playlist": {"name": "Larga"}, "tracks": [
        {"position": i, "spotify": spotify(i), "status": "error", "error": "timeout del preflight Soulseek batch (600s)"}
        for i in (1, 2, 3)]})
    write(run / "soulseek_failed.json", {"tracks": [{"playlist_position": i} for i in (1, 2, 3)]})
    state = APP.run_state(run)
    assert state["status"] == "failed" and "timeout" in state["preflight_error"]
    assert {t["state"] for t in state["tracks"]} == {"search_error"}


# ---------------------------------------------------------------------------
# Pestaña Colección
# ---------------------------------------------------------------------------

def audio_file(path: Path, size: int = 2048) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def collection_app(tmp_path, monkeypatch):
    monkeypatch.setattr(APP, "organization_library", lambda: None)  # la biblioteca del pipeline: no en el test
    downloads = tmp_path / "descargas"
    audio_file(downloads / "Playlist A" / "a1.mp3")
    audio_file(downloads / "Playlist A" / "sub" / "a2.flac")
    (downloads / "Playlist A" / "cover.jpg").write_bytes(b"x")
    (downloads / "Sin música").mkdir()
    (downloads / "Sin música" / "notas.txt").write_text("x", encoding="utf-8")
    audio_file(downloads / "_copias" / "repetido.mp3")  # sacado de la organización por dedupe.py
    audio_file(downloads / "suelto.wav")
    write(tmp_path / "cfg" / "settings.json", {"library_root": str(downloads), "runs_root": str(tmp_path / "runs")})
    return APP.App(tmp_path / "cfg"), downloads


def test_collection_counts_folders_and_skips_what_is_not_music(tmp_path, monkeypatch):
    app, _ = collection_app(tmp_path, monkeypatch)
    [root] = app.collection()["roots"]
    assert root["kind"] == "downloads" and root["exists"]
    assert [(f["name"], f["n_audio"], f["selected"]) for f in root["folders"]] == [("Playlist A", 2, False)]
    assert root["loose"] == 1  # los sueltos se cuentan, no se eligen


def test_collection_selection_is_saved_and_checked(tmp_path, monkeypatch):
    app, downloads = collection_app(tmp_path, monkeypatch)
    folder = str(downloads / "Playlist A")
    assert app.collection_select([folder], True)["selected"] == [folder]
    assert app.collection()["roots"][0]["folders"][0]["selected"]
    assert json.loads((tmp_path / "cfg" / "collection.json").read_text(encoding="utf-8"))["selected"] == [folder]
    for outside in (str(tmp_path), "relativa", str(downloads / "Playlist A" / "sub"), str(downloads / "no existe")):
        with pytest.raises(ValueError):
            app.collection_select([outside], True)
    if os.name == "nt":  # mayúsculas dan igual en Windows
        assert app.collection_select([folder.upper()], False)["selected"] == []
    app.collection_select([folder], False)
    assert app.collection_select([str(tmp_path / "ya no existe")], False)["selected"] == []  # quitar, siempre


def test_collection_roots_can_be_added_and_removed(tmp_path, monkeypatch):
    app, downloads = collection_app(tmp_path, monkeypatch)
    extra = tmp_path / "Otra música"
    audio_file(extra / "Vinilos" / "v.aiff")
    monkeypatch.setattr(APP, "ask_folder", lambda *a, **k: str(extra))
    assert app.collection_add_root() == {"cancelled": False, "path": str(extra.resolve())}
    with pytest.raises(ValueError):
        app.collection_add_root()                             # ya está
    monkeypatch.setattr(APP, "ask_folder", lambda *a, **k: str(downloads))
    with pytest.raises(ValueError):
        app.collection_add_root()                             # es la de las descargas
    monkeypatch.setattr(APP, "ask_folder", lambda *a, **k: "")
    assert app.collection_add_root() == {"cancelled": True}
    roots = app.collection()["roots"]
    assert [r["kind"] for r in roots] == ["downloads", "custom"]
    assert roots[1]["folders"][0]["name"] == "Vinilos"
    opened = []
    monkeypatch.setattr(APP.os, "startfile", opened.append, raising=False)
    app.open_folder(str(extra / "Vinilos"))                   # las carpetas de la colección se abren
    assert opened == [str(extra / "Vinilos")]
    app.collection_select([str(extra.resolve() / "Vinilos"), str(downloads / "Playlist A")], True)
    with pytest.raises(ValueError):
        app.collection_remove_root(str(downloads))            # la de las descargas no se quita
    app.collection_remove_root(str(extra))
    assert [r["kind"] for r in app.collection()["roots"]] == ["downloads"]
    assert app._collection_saved()["selected"] == [str(downloads / "Playlist A")]  # lo de la quitada se olvida
    assert (extra / "Vinilos" / "v.aiff").is_file()          # no se borra nada


def test_missing_collection_root_keeps_its_selection(tmp_path, monkeypatch):
    app, _ = collection_app(tmp_path, monkeypatch)
    extra = tmp_path / "USB"
    audio_file(extra / "Set" / "s.mp3")
    monkeypatch.setattr(APP, "ask_folder", lambda *a, **k: str(extra))
    app.collection_add_root()
    app.collection_select([str(extra.resolve() / "Set")], True)
    (extra / "Set" / "s.mp3").unlink()
    (extra / "Set").rmdir()
    extra.rmdir()                                             # disco desconectado
    root = app.collection()["roots"][1]
    assert not root["exists"] and root["missing_selected"] == 1


def test_collection_over_http(server, tmp_path, monkeypatch):
    monkeypatch.setattr(APP, "organization_library", lambda: None)
    audio_file(tmp_path / "lib" / "Playlist B" / "b.mp3")
    status, body = request(server, "/api/collection")
    assert status == 200 and json.loads(body)["roots"][0]["folders"][0]["name"] == "Playlist B"
    folder = str(tmp_path / "lib" / "Playlist B")
    assert request(server, "/api/collection/select", body={"paths": [folder], "on": True})[0] == 403  # sin token
    status, body = request(server, "/api/collection/select", body={"paths": [folder], "on": True}, token=server.token)
    assert status == 200 and json.loads(body)["selected"] == [folder]
    assert request(server, "/api/collection/select", body={"paths": folder, "on": True}, token=server.token)[0] == 400
    assert request(server, "/api/collection/remove-root", body={"path": folder}, token=server.token)[0] == 400


# ---------------------------------------------------------------------------
# Pestaña Organización (review_app.py en un proceso propio)
# ---------------------------------------------------------------------------

FAKE_ORGANIZER = """
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        body = json.dumps({"app": "traktor-ml", "busy": True}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
print(f"[INFO] TRAKTOR ML abierto en http://127.0.0.1:{httpd.server_address[1]}/  (cerrá esta ventana para salir)", flush=True)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
sys.stdin.read()  # como --exit-on-stdin-eof
"""


def wait_for(predicate, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "no llegó a tiempo"
        time.sleep(0.05)


def test_organizer_opens_reports_busy_and_closes_with_the_app(tmp_path):
    script = tmp_path / "fake_review_app.py"
    script.write_text(FAKE_ORGANIZER, encoding="utf-8")
    organizer = APP.Organizer([sys.executable, str(script)])
    assert organizer.public()["status"] == "off" and not organizer.busy()
    assert organizer.start()["status"] in {"starting", "ready"}
    wait_for(lambda: organizer.status == "ready")
    assert organizer.url.startswith("http://127.0.0.1:") and organizer.busy()
    assert organizer.start()["url"] == organizer.url        # ya abierta: no lanza otra
    organizer.close()                                         # la app se cierra: se le cierra la entrada
    organizer.proc.wait(timeout=20)
    wait_for(lambda: organizer.status == "error")             # ya no está: la pestaña lo dice
    assert not organizer.busy()


def test_organizer_explains_why_it_did_not_open(tmp_path):
    script = tmp_path / "no_org.py"
    script.write_text("import sys\nsys.exit(\"No existe la organización 'biblioteca' en C:/x/orgs\")\n", encoding="utf-8")
    organizer = APP.Organizer([sys.executable, str(script)])
    organizer.start()
    wait_for(lambda: organizer.status == "error")
    public = organizer.public()
    assert "no está en este ordenador" in public["error"] and public["url"] is None
    assert any("No existe la organización" in line for line in public["details"])


def test_organizer_problem_messages():
    missing = APP.organizer_problem(["ModuleNotFoundError: No module named 'yaml'"], 1)
    assert "«yaml»" in missing and "requirements_v4.txt" in missing
    assert "no está en este ordenador" in APP.organizer_problem(
        ["FileNotFoundError: Audio root for dataset 'musica' not found. Tried:"], 1)
    assert "código 3" in APP.organizer_problem(["algo raro"], 3)


def test_page_has_the_three_tabs_and_the_organizer_waits_for_its_tab(server):
    status, page = request(server, "/")
    assert status == 200
    for tab in ('data-tab="descargas"', 'data-tab="coleccion"', 'data-tab="organizar"'):
        assert tab in page
    status, body = request(server, "/api/state")
    assert json.loads(body)["organizer"]["status"] == "off"  # se abre recién al entrar en la pestaña
