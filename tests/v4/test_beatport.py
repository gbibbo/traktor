"""
PURPOSE: Tests de src/v4/common/beatport.py y de la consulta de beatport_lookup, sin red: parseo de
         títulos y mezclas, núcleo de la mezcla, niveles de match A/B/C/D con candidatos armados a
         mano (casos reales de la playlist "Cumple Seba 2206" y de la colección), convención de tags
         (Artist sin remixers, "Original Mix") y caché del cliente.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.beatport import (  # noqa: E402
    BeatportClient, TrackQuery, clean_text, match, mix_core, parse_search_html, proposed_tags,
    remixers_tag, split_title_mix,
)
from src.v4.pipeline.beatport_lookup import build_queries, lookup  # noqa: E402


def cand(track_id, title, mix, artists, remixers=(), label="L", date="2020-01-01", genre="House",
         isrc=None, length_s=400.0, dj_edit=False):
    return {"track_id": track_id, "title": title, "mix": mix, "artists": list(artists),
            "remixers": list(remixers), "label": label, "release_date": date, "release_name": None,
            "genre": genre, "isrc": isrc, "bpm": 124, "key": "A Minor", "length_s": length_s,
            "is_dj_edit": dj_edit, "is_ugc_remix": False, "url": f"https://www.beatport.com/track/x/{track_id}"}


def test_split_title_mix():
    assert split_title_mix("Roar (Adana Twins Remix)") == ("Roar", "Adana Twins Remix")
    assert split_title_mix("Love’s Theme - Myd Remix") == ("Love’s Theme", "Myd Remix")
    assert split_title_mix("Sinnerman - Prelude Mix") == ("Sinnerman", "Prelude Mix")
    # un paréntesis sin palabras de versión es parte del título
    assert split_title_mix("The Drums (Din Daa Daa)") == ("The Drums (Din Daa Daa)", "")
    assert split_title_mix("Toxic Temptation (feat. Lenny)") == ("Toxic Temptation", "")
    assert split_title_mix("Check Check") == ("Check Check", "")


def test_clean_text_removes_junk():
    assert clean_text("09. (6A) Techyon vs. Soulearth - Pineal Gland") == "Techyon vs. Soulearth - Pineal Gland"
    assert clean_text("Zodiac Talking [DA09]") == "Zodiac Talking"
    assert clean_text("Gimme! Gimme! Gimme!(Dendix Bootleg) - from YouTube") == "Gimme! Gimme! Gimme!(Dendix Bootleg)"


def test_mix_core():
    assert mix_core("Myd Extended Remix") == mix_core("Myd Remix") == frozenset({"myd", "remix"})
    assert mix_core("Original Mix") == mix_core("Extended Mix") == mix_core("Radio Edit") == mix_core("") == frozenset()
    assert mix_core("Xhei Edit") != mix_core("Eduardo De La Calle Edit")


def test_level_a_isrc_exact():
    q = TrackQuery(artists="Malandra Jr.", title="Pam Pam", isrcs=("AUDCB2500370",))
    res = match(q, [cand(1, "Pam Pam", "Original Mix", ["Malandra Jr."], isrc="AUDCB2500370", genre="Tech House"),
                    cand(2, "Pam Pam", "Extended Mix", ["Malandra Jr."], isrc="AUDCB2500371")])
    assert res.level == "A" and res.best["track_id"] == 1 and res.best["genre"] == "Tech House"


def test_level_b_other_cut_prefers_label_and_original_release():
    # Spotify tiene el corte corto; Beatport solo la Extended, en el lanzamiento original y en un compilado
    q = TrackQuery(artists="Patrice Baumel", title="Roar", mix="Adana Twins Remix", label="Watergate Records",
                   date="2018-11-12", duration_s=493.6)
    comp = cand(20, "Roar ", "Adana Twins Remix", ["Adana Twins", "Patrice Baumel"], label="Watergate Records",
                date="2019-10-07", genre="Melodic House & Techno", length_s=493.6)
    orig = cand(10, "Roar", "Adana Twins Remix", ["Patrice Baumel"], remixers=["Adana Twins"],
                label="Watergate Records", date="2018-11-12", genre="Melodic House & Techno", length_s=493.6)
    res = match(q, [comp, orig])
    assert res.level == "B" and res.best["track_id"] == 10
    assert res.label_match and res.date_match and res.same_cut and not res.genre_conflict


def test_level_b_title_with_parenthesis_and_feat():
    q = TrackQuery(artists="Claptone, George Kranz", title="The Drums (Din Daa Daa)")
    res = match(q, [cand(9034128, "The Drums (Din Daa Daa) (feat. George Kranz)", "Original Mix",
                         ["George Kranz", "Claptone"], label="Different"),
                    cand(9508687, "The Drums (Din Daa Daa) (feat. George Kranz)", "Dennis Cruz Remix",
                         ["George Kranz", "Claptone"], remixers=["Dennis Cruz"], genre="Tech House")])
    assert res.level == "B" and res.best["track_id"] == 9034128 and res.best["genre"] == "House"


def test_level_c_only_other_remix_does_not_inherit_genre():
    q = TrackQuery(artists="Degiheugi, Zackarose", title="Favelas")
    res = match(q, [cand(1, "Favelas", "La Fine Equipe Remix", ["Degiheugi"], remixers=["La Fine Equipe"],
                         genre="Electronica")])
    assert res.level == "C" and res.best is None and res.hint_genres == ["Electronica"]


def test_level_d_and_dj_edits_ignored():
    q = TrackQuery(artists="RAYE", title="WHERE IS MY HUSBAND!")
    res = match(q, [cand(1, "RAYE - WHERE IS MY HUSBAND!", "Spryte Edit", ["Spryte"], dj_edit=True),
                    cand(2, "Waves of Luv", "Extended Mix", ["Matt Sassari"])])
    assert res.level == "D" and res.n_pool == 0


def test_proposed_tags_remix_moves_remixer_out_of_artist():
    # Spotify lista al remixer como artista; Beatport lo tipa como Remixer
    q = TrackQuery(artists="Groove Armada, Myd", title="Love’s Theme", mix="Myd Remix", duration_s=240.0)
    res = match(q, [cand(1, "Love’s Theme", "Myd Extended Remix", ["Groove Armada"], remixers=["Myd"],
                         label="Glitterbox Recordings", date="2026-07-10", length_s=390.0)])
    tags = proposed_tags(q, res)
    assert tags == {"artist": "Groove Armada", "title": "Love’s Theme", "remixers": "Myd Remix",
                    "label": "Glitterbox Recordings", "genre": "House", "released": "2026-07-10"}


def test_proposed_tags_remixer_mistyped_as_artist_is_removed():
    q = TrackQuery(artists="Patrice Baumel", title="Roar", mix="Adana Twins Remix")
    res = match(q, [cand(20, "Roar", "Adana Twins Remix", ["Adana Twins", "Patrice Baumel"], length_s=None)])
    tags = proposed_tags(q, res)
    assert tags["artist"] == "Patrice Baumel" and tags["remixers"] == "Adana Twins Remix"


def test_remixers_tag_original_and_extended_rule():
    # el archivo es la Extended de Beatport (mismo corte): se mantiene
    assert remixers_tag("Extended Mix", same_cut=True) == "Extended Mix"
    # Beatport tiene la Extended pero el archivo es otro corte
    assert remixers_tag("Extended Mix", same_cut=False) == "Original Mix"
    # Beatport no confirma el corte, pero el tag del archivo dice Extended
    assert remixers_tag("Original Mix", same_cut=False, local_mix="Extended Mix") == "Extended Mix"
    assert remixers_tag("Original Mix", same_cut=True, local_mix="Extended Mix") == "Original Mix"
    assert remixers_tag("Radio Edit", same_cut=False) == "Original Mix"
    assert remixers_tag("Myd Extended Remix", same_cut=True) == "Myd Extended Remix"
    assert remixers_tag("Myd Extended Remix", same_cut=False) == "Myd Remix"


def test_parse_search_html():
    payload = {"props": {"pageProps": {"dehydratedState": {"queries": [
        {"state": {"data": {"genres": []}}},
        {"state": {"data": {"data": [{"track_id": 5, "track_name": "X", "mix_name": "Original Mix"}]}}}]}}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></html>'
    assert [r["track_id"] for r in parse_search_html(html)] == [5]
    assert parse_search_html("<html></html>") == []


def test_client_cache_and_offline(tmp_path):
    client = BeatportClient(tmp_path, offline=True)
    assert client.search("nada en cache") == []
    client._path("Patrice Baumel Roar").write_text(
        json.dumps({"query": "Patrice Baumel Roar", "results": [cand(1, "Roar", "Original Mix", ["Patrice Baumel"])]}),
        encoding="utf-8")
    assert client.search("  Patrice   Baumel Roar ")[0]["track_id"] == 1
    assert client.n_requests == 0


def test_build_queries_from_tags_and_filename():
    row = pd.Series({"track_uid": "u", "filename": "Mr. G - Lights (G's Out dub).mp3", "tag_artist": "Mr. G",
                     "tag_title": "Lights", "tag_label": "Rekids", "duration_s": 400.0})
    q, fetch = build_queries(row, {"tag_remixer": "G's Out Dub", "tag_isrc": None, "tag_date": "2013-07-22"})[0]
    assert fetch and (q.artists, q.title, q.mix, q.label, q.date) == ("Mr. G", "Lights", "G's Out Dub", "Rekids", "2013-07-22")
    # TPE4 con solo el nombre del remixer
    q, _ = build_queries(row, {"tag_remixer": "Adana Twins", "tag_isrc": "gbx1", "tag_date": None})[0]
    assert q.mix == "Adana Twins Remix" and q.isrcs == ("GBX1",)
    # sin tags: del nombre del archivo
    row = pd.Series({"track_uid": "u", "filename": "09. (6A) Techyon vs. Soulearth - Pineal Gland (Extended Mix).mp3",
                     "duration_s": None})
    q, _ = build_queries(row, {})[0]
    assert (q.artists, q.title, q.mix) == ("Techyon vs. Soulearth", "Pineal Gland", "Extended Mix")


def test_build_queries_youtube_rip():
    # el tag de artista es el canal; el título trae "Artista - Tema (Official Music Video)"
    row = pd.Series({"track_uid": "u", "filename": "nox-vahn-marsh-come-together-official-music-video.mp3",
                     "tag_artist": "Anjunadeep", "tag_title": "Nox Vahn & Marsh - Come Together (Official Music Video)",
                     "duration_s": 300.0})
    variants = build_queries(row, {})
    assert (variants[0][0].artists, variants[0][0].title) == ("Nox Vahn & Marsh", "Come Together")
    # la variante invertida reusa los candidatos (no busca)
    assert variants[-1][1] is False and variants[-1][0].artists == "Come Together"
    row = pd.Series({"track_uid": "u", "filename": "x.mp3", "tag_artist": "Bonjour La France",
                     "tag_title": "Polo & Pan | Mexicali", "duration_s": None})
    assert (build_queries(row, {})[0][0].artists, build_queries(row, {})[0][0].title) == ("Polo & Pan", "Mexicali")


def test_lookup_swapped_variant_uses_fetched_candidates(tmp_path):
    client = BeatportClient(tmp_path, offline=True)
    client._path("Finder Carl Cox").write_text(json.dumps({"query": "Finder Carl Cox", "results": [
        cand(1, "Finder", "Original Mix", ["Carl Cox"], label="Intec", genre="Techno (Peak Time / Driving)")]}),
        encoding="utf-8")
    row = pd.Series({"track_uid": "u", "filename": "Finder - Carl Cox.mp3", "tag_artist": "Finder",
                     "tag_title": "Carl Cox", "tag_label": "Intec", "duration_s": None})
    res, q, used = lookup(client, build_queries(row, {}))
    assert res.level == "B" and q.artists == "Carl Cox" and res.best["track_id"] == 1


def test_label_normalization():
    from src.v4.common.beatport import norm_label
    assert norm_label("Subliminal Records") == norm_label("Subliminal")
    assert norm_label("Full Time Records") == norm_label("Fulltime Production")
    assert norm_label("House Of EFUNK Record") == norm_label("House Of EFUNK Records")
    assert norm_label("Records") == ""


def test_artist_alias_needs_confirmation():
    # Beatport acredita "DARCO 09"; el archivo dice "Darco (IL)"
    bp = cand(1, "76 Cutie", "Original Mix", ["DARCO 09"], label="Diynamic", genre="Indie Dance")
    assert match(TrackQuery(artists="Darco (IL)", title="76 Cutie", label="Diynamic"), [bp]).level == "B"
    assert match(TrackQuery(artists="Darco (IL)", title="76 Cutie"), [bp]).level == "D"


def test_genre_conflict_only_among_same_artists():
    # covers de "Children" que acreditan a Robert Miles no cuentan como conflicto
    q = TrackQuery(artists="Robert Miles, Deborah De Luca", title="Children", mix="Extended Mix", label="L")
    res = match(q, [cand(1, "Children", "Extended Mix", ["Deborah De Luca", "Robert Miles"], genre="Techno (Peak Time / Driving)"),
                    cand(2, "Children", "Original Mix", ["Robert Miles", "Other"], label="X", genre="Trance (Main Floor)")])
    assert res.best["track_id"] == 1 and not res.genre_conflict
    res = match(q, [cand(1, "Children", "Extended Mix", ["Deborah De Luca", "Robert Miles"], genre="Techno (Peak Time / Driving)"),
                    cand(3, "Children", "Extended Mix", ["Robert Miles", "Deborah De Luca"], label="Comp", genre="Dance / Pop")])
    assert res.genre_conflict


def test_rmx_is_remix():
    assert mix_core("Mattia Barcellona Rmx") == mix_core("Mattia Barcellona Remix")


def test_title_is_not_artist_context():
    # "Secondcity, Kydus - The Light" no es del artista "The Light"
    q = TrackQuery(artists="Secondcity, Kydus", title="The Light", mix="Original Mix")
    assert match(q, [cand(1, "The Light", "Original Mix", ["The Light"], genre="Hard Dance / Hardcore / Neo Rave")]).level == "D"


def test_acapella_is_a_version():
    assert split_title_mix("Moonraker ( Accapella )") == ("Moonraker", "Accapella")
    q = TrackQuery(artists="Foremost Poets", title="Moonraker", mix="Accapella")
    assert match(q, [cand(1, "Moonraker", "Original Mix", ["Foremost Poets"])]).level == "C"


def test_original_release_genre():
    q = TrackQuery(artists="Oxia", title="Domino", mix="Original Mix", date="2017-02-24")
    res = match(q, [cand(1, "Domino", "Original Mix", ["Oxia"], label="Armada Music", date="2017-02-24", genre="Melodic House & Techno"),
                    cand(2, "Domino", "Original Mix", ["Oxia"], label="Kompakt Extra", date="2006-05-22", genre="Techno (Peak Time / Driving)")])
    assert res.best["track_id"] == 1 and res.original_genre == "Techno (Peak Time / Driving)" and res.genre_conflict


def test_norm_transliterates_letters_without_decomposition():
    from src.v4.common.beatport import norm
    assert norm("Shlømo") == norm("Shlomo") == "shlomo"
    assert norm("Felix Kröcher") == "felix krocher"


def test_mix_followed_by_year():
    assert split_title_mix("Future (Kenny Larkin Tension mix) 2011") == ("Future", "Kenny Larkin Tension mix")


def test_lookup_reports_candidates_when_unmatched(tmp_path):
    client = BeatportClient(tmp_path, offline=True)
    client._path("Malaa Four Twenty").write_text(json.dumps({"query": "Malaa Four Twenty", "results": [
        cand(1, "Four Twenty", "Original Mix", ["A-Team"])]}), encoding="utf-8")
    row = pd.Series({"track_uid": "u", "filename": "Malaa - Four Twenty.mp3", "tag_artist": "Malaa",
                     "tag_title": "Four Twenty", "duration_s": None})
    res, _, _ = lookup(client, build_queries(row, {}))
    assert res.level == "D" and res.n_candidates == 1
