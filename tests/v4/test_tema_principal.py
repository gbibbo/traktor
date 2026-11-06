"""
PURPOSE: Tests del tema principal (tools/traktor_live/tema_principal.py) con eventos como los de
         traktor-api-client: modo simple (deck MASTER), modo difícil (secciones intro/main/outro, fader
         >= 90 %, 3 s de histéresis, el anterior que sale del aire), faders de la DJM o de Traktor según la
         mezcla, temas tocados, el modo de mezcla en Traktor Settings.tsi, el registro de Live, el
         receptor que pasa los eventos y la DJM que falta sin aviso.
CHANGELOG:
  - 2026-11-06: Creación (etapa de recomendación, unidad 1).
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import sugerencias_vivo as sv  # noqa: E402
from tools.traktor_live import tema_principal as tp  # noqa: E402
from tools.traktor_live import traktor_estado as te  # noqa: E402


def load(tr, deck, path, now, length=400.0, elapsed=0.0, playing=False):
    return tr.on_event("deckLoaded", deck, {"filePath": path, "title": path, "artist": "X", "trackLength": length,
                                            "elapsedTime": elapsed, "isPlaying": playing, "tempo": 1, "bpm": 124,
                                            "resultingKey": "12m"}, now)


def tick(tr, deck, now, elapsed):
    return tr.on_event("updateDeck", deck, {"elapsedTime": elapsed}, now)


def master(tr, deck, now):
    return tr.on_event("updateMasterClock", "", {"deck": deck, "bpm": 124}, now)


def mains(changes):
    return [d["path"] for k, d in changes if k == "principal"]


def cc(channel, number, value):
    return (0xB0 | (channel - 1)) | (number << 8) | (value << 16)


def test_master_deck_is_the_main_track_even_stopped_and_a_master_change_recomputes():
    tr = tp.Tracker(mixer="externa")
    master(tr, "A", 0)
    assert mains(load(tr, "A", "a.mp3", 1)) == ["a.mp3"]               # cargado, sin play: igual manda
    assert mains(load(tr, "B", "b.mp3", 2, elapsed=150, playing=True)) == []
    assert mains(master(tr, "B", 3)) == ["b.mp3"]
    assert mains(master(tr, "C", 4)) == [] and tr.principal.path == "b.mp3"   # deck MASTER vacío: sigue
    assert mains(master(tr, "A", 5)) == ["a.mp3"]
    assert mains(master(tr, None, 6)) == ["b.mp3"]                      # master tempo: el único al aire


def test_long_mix_main_beats_intro_and_intro_beats_outro_after_3_seconds():
    tr = tp.Tracker(mixer="externa")
    master(tr, None, 0)
    assert mains(load(tr, "A", "a.mp3", 0, length=400, elapsed=100, playing=True)) == ["a.mp3"]
    load(tr, "B", "b.mp3", 40, length=600, elapsed=0, playing=True)
    switched = []
    for t in range(41, 200):
        changes = tick(tr, "A", t, 100 + t) + tick(tr, "B", t, t - 40)
        if mains(changes):
            switched.append((t, mains(changes)))
    # A pasa al outro (70 % de 400 s) en t = 180 con B en su intro (23 %): B gana 3 s después.
    # Antes, A en main (30-70 %) le ganaba a B en intro, y los dos al 100 % (fader sin dato = arriba).
    assert switched == [(183, ["b.mp3"])]


def test_with_the_djm_only_faders_at_90_percent_or_more_compete():
    tr = tp.Tracker(mixer="externa")
    master(tr, None, 0)
    load(tr, "A", "a.mp3", 0, length=400, elapsed=300, playing=True)      # outro
    tr.on_fader(1, 17, 127, 1)
    tr.on_fader(1, 18, 64, 1)                                             # B al 50 %
    assert mains(load(tr, "B", "b.mp3", 2, elapsed=10, playing=True)) == []
    tr.on_event("updateChannel", "1", {"onAirLevel": 0}, 3)               # Traktor, con mezcla externa: no cuenta
    for t in range(3, 11):
        assert mains(tick(tr, "A", t, 298 + t) + tick(tr, "B", t, 8 + t)) == []
    assert mains(tr.on_fader(1, 18, 127, 11)) == []                       # B sube: tiene que sostenerse
    assert mains(tick(tr, "B", 12, 20) + tick(tr, "B", 13, 21)) == []
    changes = tick(tr, "B", 14, 22)
    assert mains(changes) == ["b.mp3"]
    data = changes[-1][1]
    assert data["motivo"] == "intro gana a outro" and data["al_aire"]["A"] == {"seccion": "outro", "fader": 1.0}


def test_a_dip_resets_the_wait_and_leaving_the_air_switches_at_once():
    tr = tp.Tracker(mixer="externa")
    master(tr, None, 0)
    load(tr, "A", "a.mp3", 0, length=400, elapsed=300, playing=True)
    tr.on_fader(1, 17, 127, 0)
    assert mains(load(tr, "B", "b.mp3", 1, elapsed=10, playing=True)) == []   # B sin dato = arriba: espera
    assert mains(tr.on_fader(1, 18, 100, 2)) == []                            # 79 %: deja de competir
    assert mains(tr.on_fader(1, 18, 127, 3)) == []                            # vuelve: la espera empieza de nuevo
    assert mains(tick(tr, "A", 5, 305)) == []
    changes = tr.on_fader(1, 17, 0, 5.5)                                      # A sale del aire: enseguida
    assert mains(changes) == ["b.mp3"] and changes[-1][1]["motivo"] == "único al aire; el anterior salió"


def test_internal_mixing_uses_traktor_volume_and_ignores_the_djm():
    tr = tp.Tracker(mixer="interna")
    master(tr, None, 0)
    tr.on_event("updateChannel", "1", {"isOnAir": False}, 0)
    tr.on_event("updateChannel", "1", {"onAirLevel": 0}, 0)
    assert mains(load(tr, "A", "a.mp3", 1, elapsed=200, playing=True)) == []
    assert mains(tr.on_fader(1, 17, 127, 2)) == []
    assert mains(tr.on_event("updateChannel", "1", {"onAirLevel": 0.95}, 3)) == ["a.mp3"]


def test_played_needs_30_seconds_on_air_not_just_loading_or_prelistening():
    tr = tp.Tracker(mixer="externa")
    master(tr, "A", 0)
    load(tr, "A", "a.mp3", 0)                                     # principal por ser el MASTER, nunca suena
    tr.on_fader(1, 18, 0, 0)
    load(tr, "B", "b.mp3", 0, playing=True)
    played = []
    for t in range(1, 61):                                        # un minuto preescuchando con el fader abajo
        played += [d["path"] for k, d in tick(tr, "B", t, t) if k == "tocado"]
    tr.on_fader(1, 18, 127, 61)
    for t in range(62, 120):
        played += [d["path"] for k, d in tick(tr, "B", t, t) if k == "tocado"]
    assert played == ["b.mp3"] and tr.played == ["b.mp3"]
    assert 30 <= tr.on_air_s["b.mp3"] < 60 and tr.on_air_s.get("a.mp3", 0) == 0


def test_position_is_estimated_between_updates_with_the_tempo():
    d = tp.Deck(path="x", length=100, elapsed=29, t_elapsed=10, playing=True, tempo=1.05)
    assert d.section(10) == "intro" and d.section(11) == "main"          # 29 + 1,05 s
    assert abs(d.position(30) - (29 + 3 * 1.05) / 100) < 1e-9            # a lo sumo 3 s sin aviso
    assert tp.Deck(path="x").section(0) == "main"                        # largo desconocido


def test_mixer_mode_comes_from_traktor_settings(tmp_path):
    tsi = tmp_path / "Traktor Settings.tsi"

    def write(value, mtime):
        tsi.write_text('<?xml version="1.0" encoding="UTF-8" standalone="no" ?><NIXML><TraktorSettings>'
                       f'<Entry Name="Routing.Mixer.External" Type="1" Value="{value}"></Entry>'
                       '</TraktorSettings></NIXML>', encoding="utf-8")
        os.utime(tsi, (mtime, mtime))

    write(1, 1_000_000)
    assert tp.mixer_mode(tsi) == "externa"
    watch = tp.MixerWatch(tsi)
    assert watch.poll() and watch.mode == "externa"
    assert not watch.poll()                                       # sin cambios en el archivo
    write(0, 1_000_100)
    assert watch.poll() and watch.mode == "interna"
    assert tp.mixer_mode(tmp_path / "no existe.tsi") is None and tp.mixer_mode(None) is None
    assert tp.settings_tsi(tmp_path / "collection.nml") == tsi and tp.settings_tsi(None) is None


def test_faders_from_settings_fall_back_to_the_djm_750mk2():
    assert tp.faders_from(None) == tp.DEFAULT_FADERS == {"A": (1, 17), "B": (1, 18), "C": (1, 19), "D": (1, 20)}
    assert tp.faders_from({"A": [2, 7], "Z": [1, 1]}) == {"A": (2, 7)}
    assert tp.faders_from({"A": "x"}) == tp.DEFAULT_FADERS


def test_live_logs_the_mixer_the_faders_and_the_main_track(tmp_path):
    log = tmp_path / "sesion.jsonl"
    said, seen, now = [], [], [0.0]
    live = tp.Live(tp.Tracker(), log, lambda level, text: said.append(text), clock=lambda: now[0])
    live.on_principal.append(seen.append)
    live.set_mixer("externa")
    live.http("updateMasterClock", "", {"deck": None, "bpm": 124})
    live.midi(cc(1, 17, 0))
    live.midi(0xF8)                                                       # reloj MIDI: se ignora
    live.midi(cc(1, 17, 0))                                               # mismo valor: no se anota
    live.midi(cc(2, 17, 99))                                              # otro canal: no es un fader
    live.http("deckLoaded", "A", {"filePath": "C:\\Música\\a.mp3", "title": "Tema", "artist": "Artista",
                                  "trackLength": 400, "elapsedTime": 150, "isPlaying": True})
    now[0] = 1.0
    live.midi(cc(1, 17, 127))
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["mezcla", "fader", "fader", "principal"]
    assert rows[1]["data"] == {"canal": 1, "cc": 17, "valor": 0} and rows[3]["data"]["path"] == "C:\\Música\\a.mp3"
    assert [d["deck"] for d in seen] == ["A"]
    assert said[-1] == "Tema principal: Artista - Tema (deck A, main 38 %, fader 100 %; único al aire)"


def test_the_receiver_passes_each_event_to_the_listener_and_survives_its_errors():
    got = []

    def listener(kind, target, data):
        got.append((kind, target, data))
        if kind == "updateChannel":
            raise RuntimeError("falla a propósito")

    servers = te.open_servers(0, te.make_handler(te.State(), None, listener=listener))
    port = servers[0].server_address[1]
    import threading
    stop = threading.Event()
    thread = threading.Thread(target=te.serve, args=(servers, stop), daemon=True)
    thread.start()
    try:
        for path, body in (("/updateChannel/2", {"onAirLevel": 0.5}), ("/updateMasterClock", {"deck": "B"})):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
    finally:
        stop.set()
        thread.join(timeout=5)
    assert got == [("updateChannel", "2", {"onAirLevel": 0.5}), ("updateMasterClock", "", {"deck": "B"})]


def test_a_missing_mixer_is_logged_without_a_notification():
    said = []
    djm = sv.MidiWatcher("DJM", "winmm", lambda level, text: said.append((level, text)),
                         lister=lambda backend: ["XONE:K2 MIDI"], label="mezclador", missing_level="info")
    djm.poll()
    k2 = sv.MidiWatcher("K2", "winmm", lambda level, text: said.append((level, text)), lister=lambda backend: [])
    k2.poll()
    assert [level for level, _ in said] == ["info", "aviso"]
    assert said[0][1].startswith("No veo el mezclador «DJM»") and said[1][1].startswith("No veo el controlador «K2»")
