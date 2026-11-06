"""
PURPOSE: Tests de lo que cambia entre Windows y macOS en las sugerencias en vivo: rutas de la colección
         de Traktor (traktor_nml.location_str), camino del árbol hasta la playlist, carpeta por defecto,
         botón (nota o CC), aprender el camino si Traktor lo muestra distinto, ajustes guardados, motor
         MIDI python-rtmidi (con un rtmidi de mentira) y dónde busca Traktor el instalador. Corren en
         cualquier sistema: cada función recibe el sistema que se prueba.
CHANGELOG:
  - 2026-11-05: Creación inicial.
"""
import shlex
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.traktor_nml import location_str  # noqa: E402
from tools.traktor_live import instalar_api_client as inst  # noqa: E402
from tools.traktor_live import midi_io  # noqa: E402
from tools.traktor_live import sugerencias_vivo as sv  # noqa: E402
from tools.traktor_live import traktor_estado as te  # noqa: E402
from tools.traktor_live.midi_monitor import decode  # noqa: E402

NML = "TRAKTOR ML Sugerencias NML"


def test_collection_paths_on_windows_and_mac():
    assert location_str("C:", "/:Música/:#1 BIBO/:", "a.mp3") == "C:\\Música\\#1 BIBO\\a.mp3"
    assert location_str("Macintosh HD", "/:Users/:gabriel/:Music/:", "a.mp3",
                        boot_volume="Macintosh HD") == "/Users/gabriel/Music/a.mp3"
    assert location_str("PENDRIVE", "/:TRAKTOR ML/:Musica/:", "a.mp3",
                        boot_volume="Macintosh HD") == "/Volumes/PENDRIVE/TRAKTOR ML/Musica/a.mp3"
    assert location_str("", "/:tmp/:x/:", "a.mp3") == "/tmp/x/a.mp3"


def test_explorer_path_on_mac():
    assert sv.explorer_path(f"/Users/Shared/TRAKTOR ML Sugerencias/{NML}.nml", "darwin", "Macintosh HD") == [
        "Browser", "Explorer", "Macintosh HD", "Users", "Shared", "TRAKTOR ML Sugerencias", NML]
    assert sv.explorer_path(f"/Volumes/PENDRIVE/Sug/{NML}.nml", "darwin", "Macintosh HD") == [
        "Browser", "Explorer", "PENDRIVE", "Sug", NML]


def test_default_folder_per_system():
    assert sv.default_live_dir("win32", "D:\\Usuarios\\gabriel") == "D:\\TRAKTOR ML Sugerencias"
    assert sv.default_live_dir("darwin", "/Users/gabriel") == "/Users/Shared/TRAKTOR ML Sugerencias"
    assert sv.default_live_dir("linux", "/home/gabriel") == "/home/gabriel/TRAKTOR ML Sugerencias"


def test_button_can_be_a_note_or_a_cc():
    note = sv.Button("note", 15, 14)
    assert note.pressed("note_on", 15, 14, 127) and not note.pressed("note_off", 15, 14, 0)
    assert not note.pressed("note_on", 15, 13, 127) and not note.pressed("note_on", 1, 14, 127)
    cc = sv.Button("cc", 2, 20)
    assert cc.name() == "Ch02.CC.020"
    assert cc.pressed("cc", 2, 20, 127) and not cc.pressed("cc", 2, 20, 0)
    assert sv.Button.from_event(*decode(0x7F0E9E)) == sv.Button("note", 15, 14)    # 9E 0E 7F
    assert sv.Button.from_event(*decode(0x7F14B1)) == sv.Button("cc", 2, 20)       # B1 14 7F
    assert sv.Button.from_event(*decode(0x00148E)) is None                          # note off
    assert sv.Button.from_event(*decode(0xF8)) is None                              # reloj


def test_toggle_learns_the_path_traktor_shows():
    learned = []
    computed = ["Browser", "Explorer", "Macintosh HD", "Users", "Shared", "TRAKTOR ML Sugerencias", NML]
    tg = sv.Toggle(computed, on_learn=learned.append)
    assert tg.decide("Browser | Explorer | Macintosh HD | Música | Techno", 4)[0] == "sugerencias"
    tg.failed("sugerencias", computed, 0)
    # el DJ navega a mano hasta la playlist, que Traktor muestra por otro camino, y aprieta el botón
    shown = "Browser | Explorer | Users | Shared | TRAKTOR ML Sugerencias | " + NML
    assert tg.decide(shown, 0) == ("volver", ["Browser", "Explorer", "Macintosh HD", "Música", "Techno"], 4)
    assert learned == [shown.split(" | ")] and tg.target == shown.split(" | ")
    assert tg.decide("Browser | Playlist | Cluster_02", 1) == ("sugerencias", shown.split(" | "), 0)
    # otra carpeta con el mismo nombre de playlist pero distinta carpeta: no se aprende
    assert not tg.at_target(["Browser", "Explorer", "C:", "Otra", NML])


def test_settings_roundtrip_and_bad_button(tmp_path):
    path = tmp_path / "ajustes.json"
    assert sv.load_settings(path) == {}
    sv.save_settings({"midi": "Kontrol", "button": {"kind": "cc", "channel": 3, "number": 64},
                      "destinos": {"x.nml": ["Browser", "Explorer", "Música"]}}, path)
    data = sv.load_settings(path)
    assert sv.button_from_settings(data) == sv.Button("cc", 3, 64) and data["destinos"]["x.nml"][2] == "Música"
    assert sv.button_from_settings({"button": {"kind": "note"}}) == sv.DEFAULT_BUTTON
    path.write_text("no es json", encoding="utf-8")
    assert sv.load_settings(path) == {}


def _collection_without_cues(tmp_path, n):
    """collection.nml como la de una instalación nueva: temas sin cues propios."""
    import xml.etree.ElementTree as ET
    music = tmp_path / "music"
    music.mkdir()
    root = ET.Element("NML", VERSION="20")
    coll = ET.SubElement(root, "COLLECTION", ENTRIES=str(n))
    for i in range(n):
        f = music / f"Tema {i}.mp3"
        f.write_bytes(b"")
        e = ET.SubElement(coll, "ENTRY", TITLE=f"Tema {i}", ARTIST="A")
        p = f.resolve()
        volume = p.drive if p.drive else ""
        ET.SubElement(e, "LOCATION", VOLUME=volume, FILE=p.name,
                      DIR="/:" + "/:".join(p.parent.parts[1:]) + "/:")
    out = tmp_path / "collection.nml"
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out


def test_a_new_collection_without_cues_still_gives_lists(tmp_path):
    sugg = sv.Suggestions(_collection_without_cues(tmp_path, 12), tmp_path / "sug", 10)
    assert sugg.pool == "temas de la colección" and len(sugg.tracks) == 12
    assert len(sugg.next()) == 10 and sugg.nml.exists()
    small = tmp_path / "chica"
    small.mkdir()
    try:
        sv.Suggestions(_collection_without_cues(small, 3), small / "sug", 10)
    except ValueError as exc:
        assert "3 temas" in str(exc)
    else:
        raise AssertionError("con 3 temas y n=10 tenía que avisar")


class _FakeMidiIn:
    """Lo mínimo de rtmidi.MidiIn para probar RtMidiInput sin un controlador."""
    instances = []

    def __init__(self, name=None):
        self.callback = None
        self.opened = None
        self.deleted = False
        _FakeMidiIn.instances.append(self)

    def get_ports(self):
        return ["XONE:K2 MIDI 0", "DJM-750MK2 1"]

    def ignore_types(self, **kw):
        self.ignored = kw

    def set_callback(self, func, data=None):
        self.callback = func

    def cancel_callback(self):
        self.callback = None

    def open_port(self, port, name=None):
        if port > 1:
            raise ValueError("puerto inválido")
        self.opened = port

    def close_port(self):
        self.opened = None

    def delete(self):
        self.deleted = True


def test_rtmidi_backend_with_a_fake_rtmidi(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtmidi", types.SimpleNamespace(MidiIn=_FakeMidiIn))
    assert midi_io.list_inputs("rtmidi") == ["XONE:K2 MIDI 0", "DJM-750MK2 1"]
    m = midi_io.open_input(0, "rtmidi")
    assert m.open() is None
    fake = _FakeMidiIn.instances[-1]
    assert fake.opened == 0 and fake.ignored == {"sysex": True, "timing": False, "active_sense": False}
    fake.callback(([0x9E, 14, 127], 0.0))
    fake.callback(([0xF0, 1, 2, 3, 0xF7], 0.0))           # sysex: se ignora
    packed, _t = m.events.get(timeout=1)
    assert decode(packed) == ("note_on", 15, 14, 127) and m.events.empty()
    m.close()
    assert fake.deleted and fake.opened is None
    assert "ValueError" in midi_io.open_input(5, "rtmidi").open()


def test_rtmidi_missing_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtmidi", None)       # import rtmidi -> ImportError
    assert "python-rtmidi" in midi_io.open_input(0, "rtmidi").open()
    assert midi_io.default_backend("win32") == "winmm" and midi_io.default_backend("darwin") == "rtmidi"
    assert midi_io.pack([0xB0, 7, 100]) == 0x6407B0 and midi_io.pack([0xF8]) == 0xF8


def _fake_traktor(qml: Path) -> None:
    (qml / "CSI" / "D2").mkdir(parents=True)
    (qml / "CSI" / "D2" / "D2.qml").write_text("Mapping {}", encoding="utf-8")


def test_installer_finds_traktor_on_windows_and_mac(tmp_path):
    win = tmp_path / "win"
    _fake_traktor(win / "Traktor Pro 3" / "Resources64" / "qml")
    _fake_traktor(win / "Traktor Pro 4" / "Resources64" / "qml")
    (win / "Traktor Pro 5" / "Resources64" / "qml").mkdir(parents=True)          # sin D2.qml: no cuenta
    assert inst.default_d2("win32", win) == win / "Traktor Pro 4" / "Resources64" / "qml" / "CSI" / "D2"
    mac = tmp_path / "mac"
    _fake_traktor(mac / "Traktor Pro 4" / "Traktor Pro 4.app" / "Contents" / "Resources" / "qml")
    assert inst.default_d2("darwin", mac) == (mac / "Traktor Pro 4" / "Traktor Pro 4.app" / "Contents"
                                              / "Resources" / "qml" / "CSI" / "D2")
    assert inst.default_d2("darwin", tmp_path / "vacio") is None and inst.qml_dirs("linux") == []


def test_mac_admin_command_quotes_paths_with_spaces():
    cmd = inst.mac_admin_command(["--sonda"], 501, 20)
    assert cmd[:2] == ["osascript", "-e"] and cmd[2].endswith('" with administrator privileges')
    inner = cmd[2][len('do shell script "'):-len('" with administrator privileges')]
    args = shlex.split(inner.replace('\\"', '"').replace("\\\\", "\\"))
    assert args[-3:] == ["--sonda", "--owner", "501:20"] and args[1].endswith("instalar_api_client.py")


def test_open_servers_refuses_a_busy_port():
    first = te.open_servers(0, te.make_handler(te.State(), None))
    port = first[0].server_address[1]
    try:
        try:
            second = te.open_servers(port, te.make_handler(te.State(), None))
        except OSError as exc:
            assert te.port_in_use(exc)
        else:
            for s in second:
                s.server_close()
            raise AssertionError("el segundo receptor abrió el mismo puerto")
    finally:
        for s in first:
            s.server_close()


def test_the_default_button_needs_d_sharp_0_held_and_a_learned_one_does_not():
    assert sv.combo_name(sv.DEFAULT_BUTTON, sv.DEFAULT_MODIFIER) == "Ch15.Note.D#0 + Ch15.Note.D0"
    assert sv.modifier_from_settings({}) == sv.DEFAULT_MODIFIER
    assert sv.modifier_from_settings({"button": {"kind": "cc", "channel": 3, "number": 64}}) is None
    assert sv.modifier_from_settings({"modificador": None}) is None
    assert sv.modifier_from_settings({"modificador": {"kind": "note", "channel": 1, "number": 2}}) == sv.Button("note", 1, 2)
    held = sv.Held()
    b, m = sv.DEFAULT_BUTTON, sv.DEFAULT_MODIFIER
    assert not sv.fires(b, m, held, "note_on", 15, 14, 127)                  # D0 solo: pliega la biblioteca
    held.update("note_on", 15, 15, 127)                                     # mantener D#0
    assert sv.fires(b, m, held, "note_on", 15, 14, 127)
    held.update("note_off", 15, 15, 0)
    assert not sv.fires(b, m, held, "note_on", 15, 14, 127)
    assert sv.fires(b, None, held, "note_on", 15, 14, 127)                  # sin modificador, como antes
    held.update("cc", 1, 7, 127)
    assert held.is_held(sv.Button("cc", 1, 7))
    held.update("cc", 1, 7, 0)
    assert not held.is_held(sv.Button("cc", 1, 7))


def test_learning_takes_the_held_button_as_modifier():
    import queue
    import threading

    def learn(messages):
        q = queue.SimpleQueue()
        for status, d1, d2 in messages:
            q.put((status | (d1 << 8) | (d2 << 16), 0.0))
        return sv.learn_button(q, threading.Event())

    assert learn([(0x9E, 15, 127), (0x9E, 14, 127), (0x8E, 14, 0), (0x8E, 15, 0)]) ==         (sv.Button("note", 15, 14), sv.Button("note", 15, 15))
    assert learn([(0x9E, 14, 127), (0x9E, 14, 0)]) == (sv.Button("note", 15, 14), None)
    assert learn([(0xB2, 20, 127), (0xB2, 20, 0)]) == (sv.Button("cc", 3, 20), None)
