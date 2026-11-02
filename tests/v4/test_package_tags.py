"""
PURPOSE: Tests de package_tags.py (etiquetas y carátula de los temas del paquete portátil): limpieza
         local de artista y título, mezcla en el título, verificación de Deezer, reglas del plan,
         segunda búsqueda (rescue) y escritura con respaldo y vuelta atrás sobre un MP3 mínimo (el
         audio no cambia).
CHANGELOG:
  - 2026-11-02: rescue_pick, B sin confirmar (Genre y Album vacíos), plan con rescue.csv y missing_after.
  - 2026-10-30: Creación inicial.
"""
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.tags import audio_payload_hash, read_release_tags, read_tags  # noqa: E402
from src.v4.pipeline.package_tags import (  # noqa: E402
    cover_options, cover_state, deezer_pick, local_fields, missing_after, plan_row, rescue_pick, run_revert,
    title_with_mix, write_file,
)


@pytest.mark.parametrize("artist,title,filename,expected", [
    ("11. (9A) Ascent & Shogan", "11. (9A) Ascent & Shogan - Weird Circles",
     "11. (9A) Ascent & Shogan - Weird Circles.mp3", ("Ascent & Shogan", "Weird Circles", "")),
    ("Banco de Gaia", "Heliopolis", "Banco de Gaia - Heliopolis (Silinder Remix).mp3",
     ("Banco de Gaia", "Heliopolis", "Silinder Remix")),
    ("Matt Sassari", "Matt Sassari - Put A Record On (Extended Mix)", "Matt Sassari - Put A Record On (Extended Mix).mp3",
     ("Matt Sassari", "Put A Record On", "Extended Mix")),
    ("Sébastien Léger", "Sbastien Lger - Underwater [ADID035]", "Sebastien leger - Underwater.mp3",
     ("Sébastien Léger", "Underwater", "")),
    ("Nu  Jo Ke", "Nu  Jo Ke - Who Loves The Sun (Original Mix)", "Nu Jo Ke - Who Loves The Sun (Original Mix).mp3",
     ("Nu & Jo Ke", "Who Loves The Sun", "Original Mix")),
    ("", "Chica Cascada", "Chica Cascada.mp3", ("", "Chica Cascada", "")),
    ("", "", "Hostox - The Lighthouse.mp3", ("Hostox", "The Lighthouse", "")),
    # un " - " que no es el artista queda en el título
    ("Stavroz", "Love - Hate", "Stavroz - Love - Hate.mp3", ("Stavroz", "Love - Hate", "")),
    # el feat. se conserva; la mezcla que solo está en el nombre de archivo se recupera
    ("Khan, Julee Cruise", "Say Goodbye feat. Julee Cruise",
     "Julee Cruise, Khan - Say Goodbye feat. Julee Cruise (LoSoul 'She's Home' Mix).mp3",
     ("Khan, Julee Cruise", "Say Goodbye feat. Julee Cruise", "LoSoul 'She's Home' Mix")),
    ("Foremost Poets", "Moonraker", "Foremost Poets - Moonraker ( Accapella ).mp3",
     ("Foremost Poets", "Moonraker", "Accapella")),
    # artista de relleno en el tag: manda el del nombre de archivo
    ("various", "Feel the Fun", "Electric Tease - Feel the Fun.mp3", ("Electric Tease", "Feel the Fun", "")),
])
def test_local_fields(artist, title, filename, expected):
    r = local_fields(artist, title, "", filename)
    assert (r["artist"], r["title"], r["mix"]) == expected


def test_title_with_mix():
    assert title_with_mix("Children", "Extended Mix") == "Children (Extended Mix)"
    assert title_with_mix("Eagle", "Original Mix") == "Eagle"
    assert title_with_mix("Heliopolis (Silinder Remix)", "Silinder Remix") == "Heliopolis (Silinder Remix)"
    assert title_with_mix("Sol", "") == "Sol"


def test_deezer_pick():
    c = lambda title, artist, dur, cover="https://x/c.jpg": {  # noqa: E731
        "id": 1, "title": title, "artist": artist, "duration": dur, "album": "Alb", "cover": cover}
    cands = [c("Heliopolis (Eedupolis Dog Mix)", "Banco de Gaia", 524),       # otra mezcla
             c("Heliopolis", "Otro", 445),                                    # otro artista
             c("Heliopolis (Silinder Remix)", "Banco de Gaia", 420, ""),      # sin carátula
             c("Heliopolis (Silinder Remix)", "Banco de Gaia", 410)]
    assert deezer_pick("Banco de Gaia", "Heliopolis", "Silinder Remix", 400.0, cands)["duration"] == 410
    assert deezer_pick("Banco de Gaia", "Heliopolis", "", 999.0, cands) is None
    # la misma duración alcanza aunque la mezcla no esté escrita igual
    assert deezer_pick("Banco de Gaia", "Heliopolis", "", 525.0, cands)["duration"] == 524
    assert deezer_pick("Banco de Gaia, Tim Booth", "Heliopolis", "Silinder Remix", None, cands)["duration"] == 410


def _row(**kw):
    base = {"level": "D", "confirmed": "False", "cover_state": "none",
            "cur_title": "", "cur_artist": "", "cur_remixers": "", "cur_album": "", "cur_label": "",
            "cur_released": "", "cur_genre": "", "loc_artist": "", "loc_title": "", "loc_mix": ""}
    base.update(kw)
    return base


def test_plan_row_confirmed_beatport():
    r = _row(level="B", confirmed="True", cur_title="02. HALDOLIUM - Our Own_Happiness", cur_artist="Haldolium",
             cur_genre="Psy-Trance", loc_artist="Haldolium", loc_title="Our Own_Happiness", loc_mix="",
             bp_title="Our Own Happyness", new_artist="Haldolium", new_remixers="Original Mix",
             new_label="Nutek", new_released="2019-05-01", new_genre="Trance", bp_release_name="Nini",
             bp_image="https://geo-media.beatport.com/image_size/{w}x{h}/abc.jpg")
    p = plan_row(r)
    # título del archivo (no el de Beatport), Artist/Remixers/Label/Released de Beatport, género intacto
    assert p["new_title"] == "Our Own_Happiness" and p["new_artist"] == ""
    assert (p["new_remixers"], p["new_label"], p["new_released"], p["new_album"]) == \
           ("Original Mix", "Nutek", "2019-05-01", "Nini")
    assert p["new_genre"] == "" and p["source"] == "beatport_B"
    assert p["cover_url"].endswith("/image_size/600x600/abc.jpg") and p["cover_source"] == "beatport"
    # con carátula válida no se baja otra
    assert plan_row({**r, "cover_state": "ok"})["cover_url"] == ""


def test_plan_row_local_and_title_mix():
    r = _row(cur_title="Matt Sassari - Put A Record On (Extended", cur_artist="Matt Sassari",
             loc_artist="Matt Sassari", loc_title="Put A Record On", loc_mix="Extended Mix",
             dz_cover="https://cdn/x.jpg", dz_album="Put A Record On")
    p = plan_row(r, "separate")
    assert (p["new_title"], p["new_remixers"], p["new_album"]) == ("Put A Record On", "Extended Mix", "Put A Record On")
    assert p["cover_source"] == "deezer" and p["source"] == "deezer"
    assert plan_row(r)["new_title"] == "Put A Record On (Extended Mix)"   # por defecto, mezcla en el título
    # Remixers ya dice otra cosa: la mezcla se queda en el título
    p = plan_row({**r, "cur_remixers": "Someone"}, "separate")
    assert p["new_title"] == "Put A Record On (Extended Mix)" and p["new_remixers"] == ""
    # B sin confirmar: solo carátula, como tercera opción
    p = plan_row(_row(level="B", confirmed="False", bp_image="https://b/{w}x{h}/i.jpg", loc_title="T", loc_artist="A",
                      cur_title="T", cur_artist="A"))
    assert p["cover_source"] == "beatport_B_sin_confirmar" and p["new_artist"] == "" and p["new_title"] == ""


def test_rescue_pick():
    c = lambda artist, title, dur, mix="": {"artist": artist, "title": title, "mix": mix, "duration": dur,  # noqa: E731
                                            "album": "Alb", "genre": "House", "cover": "https://x/c.jpg"}
    # sin artista: título y duración (la canción con feat. es la misma)
    cands = [c("Otro", "Chica Cascada", 250.0), c("Puna", "Chica Cascada (feat. Mariana Paraway)", 328.887)]
    hit = rescue_pick("", "Chica Cascada", "", 328.89, "Chica Cascada.mp3", cands)
    assert hit["artist"] == "Puna" and hit["why"] == "título y duración"
    assert rescue_pick("", "Chica Cascada", "", 400.0, "Chica Cascada.mp3", cands) is None   # otra duración
    assert rescue_pick("Unknown Artist", "Chica Cascada", "", 328.89, "x.mp3", cands)["artist"] == "Puna"
    # un título genérico no alcanza; si el archivo nombra una mezcla, tiene que ser la misma
    assert rescue_pick("", "Untitled", "", 300.0, "Untitled.mp3", [c("X", "Untitled", 300.0)]) is None
    assert rescue_pick("", "Tonight", "Andre Butano Remix", 300.0, "Tonight (Andre Butano Remix).mp3",
                       [c("X", "Tonight (Y Remix)", 300.0)]) is None
    # artista y título escritos juntos en el nombre de archivo
    hit = rescue_pick("", "Dr. Kucho! Hale Bopp", "Original Mix", 389.0, "Dr. Kucho! Hale Bopp (Original Mix).mp3",
                      [c("Dr. Kucho!", "Hale Bopp", 390.5, "Original Mix")])
    assert hit["why"].startswith("artista y título en el nombre")
    # con artista: otro artista no; misma mezcla sin "Explicit"
    assert rescue_pick("Polo & Pan", "Bakara", "", 420.5, "x.mp3", [c("Otro", "Bakara", 420.0)]) is None
    hit = rescue_pick("Robyn, SG Lewis", "Impact", "Soulwax Remix - Explicit", None, "x.mp3",
                      [c("Robyn", "Impact (Soulwax Remix)", 380.0)])
    assert hit["why"] == "artista, título y mezcla"
    # solo título: duración exacta. Beatport/iTunes +-0.5 s; Deezer da segundos enteros truncados
    assert rescue_pick("", "Magnetic", "", 440.06, "Magnetic.mp3", [c("Molac", "Magnetic", 439.2)]) is None
    assert rescue_pick("", "Oiga Leon", "", 499.96, "Oiga Leon.mp3", [c("F. Allendes", "Oiga Leon", 499)])
    assert rescue_pick("", "Body Tonic", "", 445.73, "Body Tonic.mp3", [c("D. Lewis", "Body Tonic", 443)]) is None
    # el nombre de archivo nombra otra mezcla que los tags: solo con duración exacta
    ext = [c("Stainless", "Losing Your Mind (Extended Mix)", 319)]
    assert rescue_pick("Stainless", "Losing Your Mind", "Extended Mix", 365.35,
                       "Stainless - Losing your mind (Carl Cox Remix).mp3", ext) is None
    assert rescue_pick("Circulation", "Orange", "Mix 2", 483.01, "Circulation - Orange (Mix 2).mp3",
                       [c("Circulation", "Orange (Mix B)", 483.003)])["why"] == "artista, título y duración"


def test_plan_row_b_unconfirmed_and_rescue():
    # B sin confirmar: Genre y Album vacíos se completan; Label y Released no
    r = _row(level="B", confirmed="False", cur_title="Ingonyama", cur_artist="Ace Ventura", loc_artist="Ace Ventura",
             loc_title="Ingonyama", new_genre="Psy-Trance", bp_original_genre="Psy-Trance", new_label="Iboga",
             new_released="2015-01-01", bp_release_name="Paradise Engineering", cover_state="ok")
    p = plan_row(r)
    assert (p["new_genre"], p["new_album"], p["new_label"], p["new_released"]) == \
           ("Psy-Trance", "Paradise Engineering", "", "")
    assert plan_row({**r, "cur_genre": "Trance", "cur_album": "X"})["new_genre"] == ""
    # rescue: completa artista, álbum, género y carátula; no pisa lo que hay
    r = _row(cur_title="Chica Cascada", loc_title="Chica Cascada", rs_src="rescue_itunes", rs_why="título y duración",
             rs_artist="Puna", rs_title="Chica Cascada (feat. Mariana Paraway)", rs_album="Vol. 2 - Single",
             rs_genre="Electronic", rs_cover="https://is1/600x600bb.jpg")
    p = plan_row(r)
    assert (p["new_artist"], p["new_album"], p["new_genre"], p["new_title"]) == \
           ("Puna", "Vol. 2 - Single", "Electronic", "")
    assert (p["cover_url"], p["cover_source"]) == ("https://is1/600x600bb.jpg", "rescue_itunes")
    assert missing_after(r, p) == [] and missing_after(r, plan_row({**r, "rs_src": "", "rs_cover": ""})) == \
           ["cover", "artist", "album", "genre"]
    p = plan_row({**r, "cur_artist": "Puna X", "cur_album": "Mine", "cover_state": "ok"})
    assert (p["new_artist"], p["new_album"], p["cover_url"]) == ("", "", "")
    # si la de Beatport ya no se puede bajar, el plan prueba la siguiente
    opts = cover_options({**r, "level": "B", "confirmed": "True", "bp_image": "https://b/{w}x{h}/i.jpg"})
    assert opts == [("https://b/600x600/i.jpg", "beatport"), ("https://is1/600x600bb.jpg", "rescue_itunes")]
    # artista y título juntos en el nombre: el título sale del candidato
    p = plan_row(_row(cur_title="Dr. Kucho! Hale Bopp (Original Mix)", loc_title="Dr. Kucho! Hale Bopp",
                      loc_mix="Original Mix", rs_src="rescue_beatport", rs_why="artista y título en el nombre, y duración",
                      rs_artist="Dr. Kucho!", rs_title="Hale Bopp"))
    assert (p["new_artist"], p["new_title"]) == ("Dr. Kucho!", "Hale Bopp")


def test_missing_table():
    import pandas as pd
    from src.v4.pipeline.package_tags import missing_table
    row = lambda archivo, **kw: {"archivo": archivo, "level": "D", "cover_state": "none", "cover_file": "",  # noqa: E731
                                 **{f"{p}_{f}": "" for p in ("cur", "new") for f in ("artist", "album", "genre")}, **kw}
    plan = pd.DataFrame([row("a.mp3", cover_file="c.jpg", cur_artist="A", new_album="Alb", cur_genre="House"),
                         row("b.mp3", cover_state="ok", cur_artist="Unknown Artist", new_album="X"),
                         row("c.wav")])
    m = missing_table(plan).set_index("archivo")
    assert list(m.index) == ["a.mp3", "b.mp3"]                      # el WAV no se escribe: va aparte
    assert not m.loc["a.mp3", ["cover", "artist", "album", "genre"]].any()
    assert m.loc["b.mp3", ["cover", "artist", "album", "genre"]].tolist() == [False, True, False, True]


def _tiny_mp3(path: Path, frames: int = 20) -> Path:
    frame = bytes([0xFF, 0xFB, 0x90, 0x64]) + bytes(413)
    path.write_bytes(frame * frames)
    return path


def _jpeg(color=(200, 10, 10)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_write_and_revert_mp3(tmp_path):
    from mutagen.id3 import APIC, ID3, PRIV, TIT2, TPE1
    f = _tiny_mp3(tmp_path / "x.mp3")
    tags = ID3()
    tags.add(TIT2(encoding=3, text=["01. (9A) A - Song"]))
    tags.add(TPE1(encoding=3, text=["A"]))
    tags.add(PRIV(owner="TRAKTOR4", data=b"\x01\x02"))                       # datos de otro programa
    tags.add(APIC(encoding=0, mime="image/jpeg", type=3, desc="", data=b"not an image"))
    tags.save(f, v2_version=4)
    assert cover_state(f) == "bad"
    payload = audio_payload_hash(f)

    backup = write_file(f, {"title": "Song", "remixers": "Extended Mix"}, _jpeg(), False)
    assert audio_payload_hash(f) == payload
    t, rel = read_tags(f), read_release_tags(f)
    assert (t["tag_title"], t["tag_artist"], rel["tag_remixer"]) == ("Song", "A", "Extended Mix")
    assert cover_state(f) == "ok" and ID3(f).version[1] == 4 and ID3(f).getall("PRIV")
    assert backup["fields"] == {"title": ["01. (9A) A - Song"], "remixers": []} and len(backup["pictures"]) == 1

    bj = tmp_path / "backup.json"
    bj.write_text(json.dumps({"u1": {"archivo": "x.mp3", **backup}}), encoding="utf-8")
    run_revert(bj, tmp_path)
    t, rel = read_tags(f), read_release_tags(f)
    assert (t["tag_title"], rel["tag_remixer"]) == ("01. (9A) A - Song", None)
    assert cover_state(f) == "bad" and audio_payload_hash(f) == payload


def test_revert_removes_added_cover_and_fixes_mime(tmp_path):
    from mutagen.id3 import APIC, ID3, TIT2
    f = _tiny_mp3(tmp_path / "y.mp3")
    tags = ID3()
    tags.add(TIT2(encoding=3, text=["T"]))
    tags.save(f, v2_version=3)
    assert cover_state(f) == "none"
    backup = write_file(f, {}, _jpeg(), False)
    assert cover_state(f) == "ok" and backup["pictures"] == []
    bj = tmp_path / "b.json"
    bj.write_text(json.dumps({"u": {"archivo": "y.mp3", **backup}}), encoding="utf-8")
    run_revert(bj, tmp_path)
    assert cover_state(f) == "none"
    # JPEG válido con tipo "image/jpg": se corrige el tipo sin tocar la imagen
    tags = ID3(f)
    data = _jpeg((1, 2, 3))
    tags.add(APIC(encoding=0, mime="image/jpg", type=3, desc="", data=data))
    tags.save(f, v2_version=3)
    assert cover_state(f) == "mime"
    write_file(f, {}, None, True)
    assert cover_state(f) == "ok" and ID3(f).getall("APIC")[0].data == data


def test_confirmed_keeps_valid_values_and_file_mix():
    from src.v4.pipeline.package_tags import is_junk, tidy_mix
    r = _row(level="A", confirmed="True", cur_title="Kofi", cur_artist="Tunnelvisions", cur_label="Atomnation",
             cur_released="2016-05-06", loc_artist="Tunnelvisions", loc_title="Kofi", loc_mix="CIOZ Remix",
             new_artist="Tunnelvisions, X", new_remixers="Original Mix", new_label="BEAT Music Fund",
             new_released="2013-05-14", bp_release_name="Kofi EP")
    p = plan_row(r, "separate")
    # sello, fecha y artista válidos no se pisan; la mezcla del archivo manda sobre la de Beatport
    assert (p["new_artist"], p["new_label"], p["new_released"]) == ("", "", "")
    assert (p["new_title"], p["new_remixers"], p["new_album"]) == ("", "CIOZ Remix", "Kofi EP")
    assert plan_row(r)["new_title"] == "Kofi (CIOZ Remix)"
    # artista con basura: se reemplaza por el de Beatport
    assert plan_row({**r, "cur_artist": "(4A) Tunnelvisions"})["new_artist"] == "Tunnelvisions, X"
    assert is_junk("(4A) Circulation") and is_junk("Kolombo  Malikk") and not is_junk("Effy (UK)")
    assert tidy_mix("original mix") == "Original Mix" and tidy_mix("Cosmjn EDIT") == "Cosmjn EDIT"
    r = local_fields("Joy Wellboy", "Joy Wellboy - The Magic (Maceo Plex  Shall Ocin Remix)", "", "x.mp3")
    assert (r["title"], r["mix"]) == ("The Magic", "Maceo Plex & Shall Ocin Remix")
    assert local_fields("Cerrero", "Pedacito (original mix)", "", "x.mp3")["mix"] == "Original Mix"
    martinez = local_fields("Louie Vega, The Martinez  Brothers", "Let It Go", "", "x.mp3")
    assert martinez["artist"] == "Louie Vega, The Martinez Brothers"


def _tiny_aiff(path: Path, frames: int = 1000) -> Path:
    import struct
    # COMM: 1 canal, frames, 16 bits, 44100 Hz (extended de 80 bits)
    comm = struct.pack(">hIh", 1, frames, 16) + bytes.fromhex("400EAC44000000000000")
    ssnd = struct.pack(">II", 0, 0) + bytes(range(256)) * (frames * 2 // 256) + bytes(frames * 2 % 256)
    body = b"AIFF" + b"COMM" + struct.pack(">I", len(comm)) + comm + b"SSND" + struct.pack(">I", len(ssnd)) + ssnd
    path.write_bytes(b"FORM" + struct.pack(">I", len(body)) + body)
    return path


def test_write_aiff_keeps_sound_chunk(tmp_path):
    from src.v4.pipeline.package_tags import audio_hash
    f = _tiny_aiff(tmp_path / "z.aiff")
    before, payload_before = audio_hash(f), audio_payload_hash(f)
    write_file(f, {"title": "Radionova (Essential Mix)"}, _jpeg(), False)
    assert read_tags(f)["tag_title"] == "Radionova (Essential Mix)" and cover_state(f) == "ok"
    # el sonido no cambia; el hash del repo (track_uid) en AIFF incluye los tags y sí cambia
    assert audio_hash(f) == before and audio_payload_hash(f) != payload_before
