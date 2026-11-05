"""
PURPOSE: Tests de las herramientas en vivo de Traktor: instalar_api_client.py (parche de D2.qml,
         instalar y desinstalar en una carpeta de prueba), traktor_estado.py (rutas, estado y servidor
         HTTP con texto no ASCII) y sugerencias_explorer.py (playlists .m3u8 y .nml en orden, entradas
         copiadas de la colección, no pisa ni borra archivos ajenos).
CHANGELOG:
  - 2026-11-04: Creación inicial.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import instalar_api_client as inst  # noqa: E402
from tools.traktor_live import sugerencias_explorer as sx  # noqa: E402
from tools.traktor_live import sugerencias_vivo as sv  # noqa: E402
from tools.traktor_live import traktor_estado  # noqa: E402
from tools.traktor_live.sugerencias_itunes import Track  # noqa: E402
from tools.traktor_live.traktor_estado import State, make_handler, parse_endpoint  # noqa: E402

D2_SAMPLE = ("\r\nimport CSI 1.0\r\nimport QtQuick 2.0\r\n\r\nimport \"../../Defines\"\r\nimport \"../Common\"\r\n"
             "import \"../Common/Settings\"\r\n\r\nMapping\r\n{\r\n  // LOOP\r\n  LoopModePads {}\r\n}\r\n")


def test_patch_d2_matches_mod_layout_and_roundtrips():
    patched = inst.patch_d2(D2_SAMPLE)
    lines = patched.split("\r\n")
    assert lines[lines.index('import "../Common/Settings"') + 1] == 'import "./Api"'
    assert lines[lines.index("{") + 1: lines.index("{") + 3] == ["  ApiModule {}", ""]
    assert inst.patch_d2(patched) == patched
    assert inst.unpatch_d2(patched) == D2_SAMPLE


def test_patch_d2_refuses_unknown_layout():
    with pytest.raises(ValueError):
        inst.patch_d2("import CSI 1.0\nItem {}\n")


def test_install_and_uninstall_in_a_test_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "BACKUP_ROOT", tmp_path / "backups")
    monkeypatch.setattr(inst, "LOG_PATH", tmp_path / "backups" / "log.txt")
    d2 = tmp_path / "D2"
    d2.mkdir()
    (d2 / "D2.qml").write_bytes(D2_SAMPLE.encode("utf-8"))
    inst.install(d2)
    assert inst.status(d2)
    assert "ApiModule {}" in (d2 / "D2.qml").read_text(encoding="utf-8")
    backups = list((tmp_path / "backups").glob("qml_CSI_D2_*"))
    assert len(backups) == 1 and (backups[0] / "D2.qml").read_bytes() == D2_SAMPLE.encode("utf-8")
    inst.install(d2)                                    # segunda vez: no cambia nada ni copia de nuevo
    assert len(list((tmp_path / "backups").glob("qml_CSI_D2_*"))) == 1
    inst.uninstall(d2)
    assert (d2 / "D2.qml").read_bytes() == D2_SAMPLE.encode("utf-8")
    assert not (d2 / "Api").exists()


SCREEN_SAMPLE = "import QtQuick 2.0\n\nItem {\n  id: screen\n  function f() {\n    x = 1\n  }\n}\n"


def test_patch_screen_adds_probe_as_last_child_and_roundtrips():
    patched = inst.patch_screen(SCREEN_SAMPLE)
    lines = patched.split("\n")
    assert lines[-3:] == [inst.PROBE_LINE, "}", ""]
    assert inst.patch_screen(patched) == patched
    assert inst.unpatch_screen(patched) == SCREEN_SAMPLE


def test_install_probe_and_uninstall_restore_screen(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "BACKUP_ROOT", tmp_path / "backups")
    monkeypatch.setattr(inst, "LOG_PATH", tmp_path / "backups" / "log.txt")
    qml = tmp_path / "qml"
    d2 = qml / "CSI" / "D2"
    views = qml / "Screens" / "S8" / "Views"
    d2.mkdir(parents=True)
    views.mkdir(parents=True)
    (d2 / "D2.qml").write_bytes(D2_SAMPLE.encode("utf-8"))
    (views / "Screen.qml").write_bytes(SCREEN_SAMPLE.encode("utf-8"))
    inst.install(d2)
    inst.install_probe(d2)
    assert inst.probe_status(d2) and (views / "ApiBrowserProbe.qml").exists()
    inst.uninstall_probe(d2)
    inst.uninstall(d2)
    assert (views / "Screen.qml").read_bytes() == SCREEN_SAMPLE.encode("utf-8")
    assert not (views / "ApiBrowserProbe.qml").exists()
    assert (d2 / "D2.qml").read_bytes() == D2_SAMPLE.encode("utf-8")


def test_probe_lines():
    st = State()
    data = {"reason": "screenLoaded", "path": "Explorer | Music", "count": 12, "currentIndex": 3,
            "isContentList": False}
    assert st.apply("browserProbe", "left", data) == [
        "Pantalla del D2 (left) creada; su navegador está en «Explorer | Music»  [12 elementos, índice 3, nodos]"]
    assert st.apply("browserProbe", "left", {**data, "reason": "heartbeat"}) == []
    assert st.apply("browserProbe", "left", {**data, "reason": "pathChanged", "path": "Playlists"})[0].startswith(
        "Navegador del D2 (left) en «Playlists»")


def test_parse_endpoint():
    assert parse_endpoint("/deckLoaded/A") == ("deckLoaded", "A")
    assert parse_endpoint("/updateMasterClock") == ("updateMasterClock", "")
    assert parse_endpoint("/otra/cosa") is None


def test_state_lines():
    st = State()
    assert st.apply("deckLoaded", "A", {"filePath": "C:\\M\\x.mp3", "title": "Friends", "artist": "Solomun",
                                         "bpm": 120.0, "keyText": "10A"}) == [
        "Deck A cargó: Solomun - Friends  [120.0 BPM, 10A]  C:\\M\\x.mp3"]
    # Traktor 4.4.1 manda keyText vacío y la tonalidad en key (Open Key, "3m" = 10A)
    assert st.apply("deckLoaded", "B", {"filePath": "C:\\M\\y.mp3", "title": "Younan", "artist": "Ali Farahani",
                                         "bpm": 208.0, "key": "3m", "keyText": ""})[0].endswith("[208.0 BPM, 3m]  C:\\M\\y.mp3")
    assert st.apply("updateDeck", "A", {"isPlaying": True, "elapsedTime": 65.2}) == ["Deck A ▶ suena en 1:05"]
    assert st.apply("updateDeck", "A", {"elapsedTime": 66.2}) == []
    assert st.apply("updateChannel", "1", {"isOnAir": True}) == ["Canal 1 AL AIRE"]
    assert st.apply("updateChannel", "1", {"onAirLevel": 0.5}) == []
    assert st.apply("updateMasterClock", "", {"deck": "A", "bpm": 120.0}) == ["Master: deck A, 120.0 BPM"]
    assert st.apply("updateMasterClock", "", {"deck": "A", "bpm": 120.01}) == []


def test_server_receives_non_ascii_json(tmp_path):
    st = State()
    log = tmp_path / "ev.jsonl"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st, log))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"filePath": "C:\\Música\\#1 BIBO\\Electric Tease - Feel the Fun (Каа).mp3",
                           "title": "Feel the Fun (Каа)", "artist": "Electric Tease"}).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/deckLoaded/B", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        assert urllib.request.urlopen(req, timeout=5).status == 200
    finally:
        server.shutdown()
        server.server_close()
    assert st.decks["B"]["title"] == "Feel the Fun (Каа)"
    ev = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert ev["kind"] == "deckLoaded" and ev["target"] == "B" and "Каа" in ev["data"]["filePath"]


def test_probe_command_queue_and_state_over_http():
    st = State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st, None))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=body, method="POST" if body is not None else "GET")
        return json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8") or "{}")
    try:
        assert call("/enqueue", b'{"cmd": "select", "arg": 2}')["id"] == 1
        with pytest.raises(urllib.error.HTTPError):
            call("/enqueue", b'{"cmd": "load"}')            # solo órdenes de navegación
        assert call("/command/left") == {"id": 1, "cmd": "select", "arg": 2}
        assert call("/command/left") == {}
        call("/browserProbe/left", json.dumps({"reason": "done", "cmd": "select", "path": "Browser | Playlists",
                                               "names": ["A", "B"]}).encode("utf-8"))
        assert call("/state")["probe"]["left"]["names"] == ["A", "B"]
    finally:
        server.shutdown()
        server.server_close()


def test_second_receiver_on_the_same_port_is_refused():
    first = traktor_estado.ExclusiveServer(("127.0.0.1", 0), make_handler(State(), None))
    try:
        with pytest.raises(OSError) as exc:
            traktor_estado.ExclusiveServer(("127.0.0.1", first.server_address[1]), make_handler(State(), None))
        assert traktor_estado.port_in_use(exc.value)
    finally:
        first.server_close()


def test_serve_stops_both_servers_without_thread_errors():
    errors = []
    previous = threading.excepthook
    threading.excepthook = lambda a: errors.append(a.exc_value)
    try:
        st = State()
        servers = [ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st, None))]
        try:
            servers.append(traktor_estado._Server6(("::1", 0), make_handler(st, None)))
        except OSError:
            pass
        stop = threading.Event()
        runner = threading.Thread(target=traktor_estado.serve, args=(servers, stop))
        runner.start()
        for s in servers:
            host = "[::1]" if s.address_family != 2 else "127.0.0.1"
            req = urllib.request.Request(f"http://{host}:{s.server_address[1]}/updateChannel/1",
                                         data=b'{"isOnAir": true}', method="POST")
            assert urllib.request.urlopen(req, timeout=5).status == 200
        stop.set()
        runner.join(timeout=5)
        assert not runner.is_alive()
    finally:
        threading.excepthook = previous
    assert errors == [] and st.channels["1"]["isOnAir"] is True


def _entry(path: str, title: str) -> ET.Element:
    p = Path(path)
    e = ET.Element("ENTRY", TITLE=title, ARTIST="A")
    ET.SubElement(e, "LOCATION", DIR="/:" + "/:".join(p.parent.parts[1:]) + "/:", FILE=p.name, VOLUME=p.drive)
    ET.SubElement(e, "CUE_V2", NAME="n.n.", TYPE="0", START="1000.0", HOTCUE="0")
    return e


def test_explorer_playlists_keep_order_and_entries(tmp_path):
    paths = ["C:\\Música\\#1 BIBO\\Lights.mp3", "C:\\TRAKTOR ML\\Musica\\L - House & Techno\\Zimmz.mp3"]
    entries = [_entry(p, f"T{i}") for i, p in enumerate(paths)]
    tracks = [Track(path=Path(p), title=f"T{i}", artist="A", seconds=200.4, bpm="125", key="8A")
              for i, p in enumerate(paths)]
    text = sx.m3u8_text(tracks, 3)
    assert text.split("\r\n")[:4] == ["#EXTM3U", "#TRAKTOR-ML-SUGERENCIAS ronda 3", "#EXTINF:200,A - T0", paths[0]]
    out = tmp_path / "TRAKTOR ML Sugerencias.nml"
    sx.write_atomic(sx.nml_bytes(entries, 3), out)
    root = ET.parse(out).getroot()
    keys = [pk.get("KEY") for pk in root.iter("PRIMARYKEY")]
    assert keys == ["C:/:Música/:#1 BIBO/:Lights.mp3", "C:/:TRAKTOR ML/:Musica/:L - House & Techno/:Zimmz.mp3"]
    assert [e.find("CUE_V2").get("START") for e in root.find("COLLECTION")] == ["1000.0", "1000.0"]
    assert sx.is_ours(out)
    m3u = tmp_path / "TRAKTOR ML Sugerencias.m3u8"
    sx.write_atomic(text.encode("utf-8"), m3u)
    assert sx.is_ours(m3u) and sx.remove_ours(m3u) and not m3u.exists()


def test_explorer_never_overwrites_or_removes_foreign_files(tmp_path):
    foreign = tmp_path / "TRAKTOR ML Sugerencias.nml"
    foreign.write_text('<NML><PLAYLISTS><PLAYLIST UUID="0123"/></PLAYLISTS></NML>', encoding="utf-8")
    with pytest.raises(FileExistsError):
        sx.write_atomic(b"<NML/>", foreign)
    with pytest.raises(FileExistsError):
        sx.remove_ours(foreign)
    assert foreign.exists()


def test_explorer_path_of_the_suggestions_playlist():
    nml = Path("C:\\Users\\bibbo\\Music\\TRAKTOR ML Sugerencias\\TRAKTOR ML Sugerencias NML.nml")
    assert sv.explorer_path(nml) == ["Browser", "Explorer", "C:", "Users", "bibbo", "Music",
                                     "TRAKTOR ML Sugerencias", "TRAKTOR ML Sugerencias NML"]


def test_live_suggestions_live_at_the_drive_root():
    nml = Path("C:\\TRAKTOR ML Sugerencias\\TRAKTOR ML Sugerencias NML.nml")
    assert sv.explorer_path(nml) == ["Browser", "Explorer", "C:", "TRAKTOR ML Sugerencias",
                                     "TRAKTOR ML Sugerencias NML"]
    assert sv.LIVE_DIR.parent == Path(Path.home().anchor)


def test_traktor_note_names_and_default_button():
    assert sv.traktor_name(15, 13) == "Ch15.Note.C#0" and sv.traktor_name(15, 14) == "Ch15.Note.D0"
    assert sv.traktor_name(1, 60) == "Ch01.Note.C4"
    assert sv.traktor_name(**sv.DEFAULT_BUTTON) == "Ch15.Note.D0"


def test_toggle_retries_a_failed_trip_without_saving_the_halfway_place():
    target = ["Browser", "Explorer", "C:", "Sug", "Sug NML"]
    tg = sv.Toggle(target)
    assert tg.decide("Browser | Explorer | C: | Música", 2) == ("sugerencias", target, 0)
    tg.failed("sugerencias", target, 0)                    # se quedó en C: > Users
    assert tg.decide("Browser | Explorer | C: | Users", 0) == ("sugerencias", target, 0)
    back = tg.decide("Browser | Explorer | C: | Sug | Sug NML", 0)
    assert back == ("volver", ["Browser", "Explorer", "C:", "Música"], 2)
    tg.failed(*back)                                       # la vuelta tampoco llegó: se repite
    assert tg.decide("Browser | Explorer | C:", 0) == back


def test_same_name_only_case_and_final_s():
    assert sv.same_name("Playlist", "Playlists") and sv.same_name("música", "Música")
    assert not sv.same_name("A1", "A10") and not sv.same_name("Cluster_0", "Cluster_02")


def test_toggle_goes_to_suggestions_and_back_to_the_saved_place():
    target = ["Browser", "Explorer", "C:", "Sug", "Sug NML"]
    tg = sv.Toggle(target)
    assert tg.decide("Browser | Playlist | Cluster_02", 7) == ("sugerencias", target, 0)
    assert tg.decide("Browser | Explorer | C: | Sug | Sug NML", 3) == (
        "volver", ["Browser", "Playlist", "Cluster_02"], 7)
    # sin lugar guardado (el DJ volvió a mano o se reinició el programa): 10 nuevas, no "volver"
    assert tg.decide("Browser | Explorer | C: | Sug | Sug NML", 0)[0] == "sugerencias"
    # se fue de las sugerencias por su cuenta: guarda el lugar nuevo
    assert tg.decide("Browser | Explorer | C: | Música | #1 BIBO", 2)[0] == "sugerencias"
    assert tg.decide("Browser | Explorer | C: | Sug | Sug NML", 0)[1:] == (
        ["Browser", "Explorer", "C:", "Música", "#1 BIBO"], 2)


def _fake_collection(tmp_path, n=12):
    music = tmp_path / "music"
    music.mkdir()
    root = ET.Element("NML", VERSION="20")
    coll = ET.SubElement(root, "COLLECTION", ENTRIES=str(n))
    for i in range(n):
        f = music / f"Tema {i}.mp3"
        f.write_bytes(b"")
        coll.append(_entry(str(f), f"Tema {i}"))
    out = tmp_path / "collection.nml"
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out


def test_press_goes_to_suggestions_then_back_with_a_fake_probe(tmp_path):
    st = State(show_browser=False)
    sugg = sv.Suggestions(_fake_collection(tmp_path), tmp_path / "sug", 5)
    target = sv.explorer_path(sugg.nml)
    tg = sv.Toggle(target)
    where = {"path": "Browser | Playlist | Cluster_02", "index": 7}
    done = threading.Event()

    def fake_probe():                                   # hace lo que haría ApiBrowserProbe.qml
        while not done.is_set():
            cmd = st.pop_command()
            if cmd is None:
                time.sleep(0.01)
                continue
            if cmd["cmd"] == "goto":
                where["path"] = " | ".join(cmd["arg"]["path"])
                where["index"] = cmd["arg"]["index"]
            reason = "gotoDone" if cmd["cmd"] == "goto" else "done"
            st.apply("browserProbe", "left", {"reason": reason, "id": cmd["id"], "path": where["path"],
                                              "currentIndex": where["index"]})
    threading.Thread(target=fake_probe, daemon=True).start()
    try:
        sv.press(st, tg, sugg)
        assert where["path"].endswith("TRAKTOR ML Sugerencias NML") and where["index"] == 0
        assert len(list(ET.parse(sugg.nml).getroot().iter("PRIMARYKEY"))) == 5
        sv.press(st, tg, sugg)
        assert where == {"path": "Browser | Playlist | Cluster_02", "index": 7}
    finally:
        done.set()


def test_wait_result_returns_the_probe_report_for_an_order():
    st = State(show_browser=False)
    cid = st.enqueue("goto", {"path": ["Browser", "Playlist"], "index": 0})
    assert st.pop_command() == {"id": cid, "cmd": "goto", "arg": {"path": ["Browser", "Playlist"], "index": 0}}
    threading.Timer(0.1, lambda: st.apply("browserProbe", "left", {
        "reason": "gotoDone", "id": cid, "path": "Browser | Playlist", "count": 3})).start()
    res = st.wait_result(cid, 2.0)
    assert res["reason"] == "gotoDone" and st.wait_result(cid, 0.05) is None
    assert st.apply("browserProbe", "left", {"reason": "pathChanged", "path": "Browser"}) == []
