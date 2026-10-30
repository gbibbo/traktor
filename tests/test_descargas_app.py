"""
PURPOSE: Tests de la app de descargas (download_JIJIJI/app/descargas_app.py): estado de cada tema a
         partir de los archivos que escribe el orquestador, historial, comando de la descarga,
         ajustes, seguridad del servidor local y seguimiento de procesos huérfanos. Sin red.
CHANGELOG:
  - 2026-10-30: Creación inicial.
"""
from __future__ import annotations

import importlib.util
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


def test_job_command(tmp_path):
    settings = {"library_root": "D:/Musica", "runs_root": "D:/runs"}
    cmd = APP.build_job_command(tmp_path / "stop", "https://open.spotify.com/playlist/abc", settings, fresh=False)
    assert cmd[2].endswith("run_job.py") and "--fresh" not in cmd
    assert cmd[cmd.index("--library-root") + 1] == "D:/Musica" and cmd[cmd.index("--output-root") + 1] == "D:/runs"
    assert APP.build_job_command(tmp_path / "stop", "u", settings, fresh=True)[-1] == "--fresh"


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
