#!/usr/bin/env python3
"""
spotify_soulseek_download.py
============================

PURPOSE: Reconcile a Spotify playlist against Soulseek and download matches.

CHANGELOG:
- 2026-09-27: Add an optional pre-download gate for external orchestration.

Primero verifica en Soulseek la disponibilidad de TODA una playlist de Spotify
en un único batch Sockseek,
genera una lista y una playlist Spotify privada con los faltantes, y sólo entonces
descarga los disponibles. Prefiere versiones Extended cuando existen y verifica de forma conservadora
la identidad del tema, escribe metadata de Spotify + cover oficial y deja
todo en MP3 320 kbps.

Requisitos:
  - Python 3.10+
  - pip install requests mutagen
  - ffmpeg en el PATH
  - sockseek (o sldl) en el PATH
  - App de Spotify con Redirect URI exactamente:
        http://127.0.0.1:48721/callback

Uso rápido:
  export SPOTIFY_CLIENT_ID=TU_CLIENT_ID
  # PowerShell: $env:SPOTIFY_CLIENT_ID="TU_CLIENT_ID"

  python spotify_soulseek_download_fixed.py \
      "https://open.spotify.com/playlist/XXXX" \
      -o ./descargas

La primera vez se abre el navegador para autorizar Spotify.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import http.server
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

try:
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.id3 import (
        APIC,
        ID3,
        TALB,
        TCOP,
        TDRC,
        TIT2,
        TPE1,
        TPE2,
        TPOS,
        TPUB,
        TRCK,
        TSRC,
        TXXX,
    )
    from mutagen.mp3 import MP3
except ImportError:
    print("Instala mutagen: pip install mutagen", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Instala requests: pip install requests", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Spotify
# ---------------------------------------------------------------------------

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_ROOT = "https://api.spotify.com/v1"
DEFAULT_REDIRECT = "http://127.0.0.1:48721/callback"
SCOPES = "playlist-read-private playlist-read-collaborative playlist-modify-private"
REQUIRED_SCOPES = set(SCOPES.split())


def token_cache_path() -> Path:
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "traktor_spotify_token.json"
    return Path.home() / ".config" / "traktor_spotify_token.json"


@dataclass
class Track:
    spotify_id: str
    uri: str
    url: str
    title: str
    artists: list[str]
    album: str
    album_id: str | None
    album_artists: list[str]
    release_date: str | None
    track_number: int | None
    total_tracks: int | None
    disc_number: int | None
    duration_ms: int
    isrc: str | None
    cover_url: str | None
    explicit: bool = False
    label: str | None = None
    copyrights: list[str] | None = None
    playlist_id: str | None = None
    playlist_name: str | None = None
    added_at: str | None = None

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else "Unknown"

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class LocalFile:
    path: Path
    length_s: float | None
    bitrate_kbps: float | None
    format: str
    tag_title: str | None = None
    tag_artists: list[str] | None = None
    isrc: str | None = None


@dataclass
class AvailabilityCheck:
    status: str
    result_count: int
    candidates: list[dict[str, Any]]
    quality_unknown: bool = False
    error: str | None = None

    @property
    def downloadable(self) -> bool:
        return self.status in {"available", "available_quality_unknown"}


class _Callback(http.server.BaseHTTPRequestHandler):
    code = None
    state = None
    error = None

    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        type(self).code = q.get("code", [None])[0]
        type(self).state = q.get("state", [None])[0]
        type(self).error = q.get("error", [None])[0]
        body = b"OK - puedes cerrar esta pestana."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Spotify:
    def __init__(self, client_id: str, redirect: str = DEFAULT_REDIRECT):
        if not client_id:
            raise ValueError("Spotify Client ID vacío")
        self.client_id = client_id
        self.redirect = redirect
        self.cache = token_cache_path()
        self.session = requests.Session()
        self._token: dict[str, Any] | None = None
        self._album_cache: dict[str, dict[str, Any]] = {}

    def _save(self, token: dict[str, Any]):
        token = dict(token)
        token["expires_at"] = time.time() + max(
            30, float(token.get("expires_in", 3600)) - 60
        )
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text(json.dumps(token, indent=2), encoding="utf-8")
        try:
            os.chmod(self.cache, 0o600)
        except OSError:
            pass
        self._token = token

    def _load(self) -> dict[str, Any] | None:
        if not self.cache.exists():
            return None
        try:
            data = json.loads(self.cache.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _refresh(self, refresh: str) -> dict[str, Any]:
        r = self.session.post(
            TOKEN_URL,
            data={
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        data.setdefault("refresh_token", refresh)
        previous = self._token or self._load() or {}
        if not data.get("scope") and previous.get("scope"):
            data["scope"] = previous["scope"]
        self._save(data)
        return data

    def _login(self) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(self.redirect)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
            raise ValueError(
                "Usa un Redirect URI loopback explícito, por ejemplo "
                f"{DEFAULT_REDIRECT}"
            )

        verifier = secrets.token_urlsafe(64)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        state = secrets.token_urlsafe(24)
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect,
            "scope": SCOPES,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "state": state,
        }
        url = AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)
        port = parsed.port or 80

        _Callback.code = _Callback.state = _Callback.error = None
        server = http.server.HTTPServer(("127.0.0.1", port), _Callback)
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()
        print("Abre esta URL si el navegador no se abre solo:\n", url)
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

        t.join(timeout=180)
        server.server_close()
        if _Callback.error:
            raise RuntimeError(f"Auth Spotify falló: {_Callback.error}")
        if not _Callback.code or _Callback.state != state:
            raise RuntimeError(
                "No se recibió el código de autorización de Spotify o falló state"
            )

        r = self.session.post(
            TOKEN_URL,
            data={
                "client_id": self.client_id,
                "grant_type": "authorization_code",
                "code": _Callback.code,
                "redirect_uri": self.redirect,
                "code_verifier": verifier,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        self._save(data)
        return data

    def _has_required_scopes(self, token: dict[str, Any] | None) -> bool:
        if not token:
            return False
        granted = set(str(token.get("scope") or "").split())
        return REQUIRED_SCOPES.issubset(granted)

    def token(self) -> str:
        tok = self._token or self._load()
        # A refresh token cannot grant scopes that were not authorized originally.
        # If an older cache predates playlist-modify-private, force a fresh PKCE login.
        if tok and not self._has_required_scopes(tok):
            tok = None
            self._token = None
        if (
            tok
            and tok.get("access_token")
            and float(tok.get("expires_at", 0)) > time.time()
        ):
            self._token = tok
            return str(tok["access_token"])
        if tok and tok.get("refresh_token"):
            try:
                self._token = self._refresh(str(tok["refresh_token"]))
                return str(self._token["access_token"])
            except requests.RequestException:
                # El refresh puede haber expirado/revocado. Reautorizar es más seguro
                # que reciclar un access token inválido.
                self._token = None
        self._token = self._login()
        return str(self._token["access_token"])

    def _force_refresh_after_401(self) -> str:
        tok = self._token or self._load() or {}
        refresh = tok.get("refresh_token")
        if refresh:
            try:
                self._token = self._refresh(str(refresh))
                return str(self._token["access_token"])
            except requests.RequestException:
                pass
        self._token = self._login()
        return str(self._token["access_token"])

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        url = path if path.startswith("http") else API_ROOT + path
        headers = {"Authorization": f"Bearer {self.token()}"}

        for attempt in range(4):
            r = self.session.get(url, params=params, headers=headers, timeout=25)

            if r.status_code == 401:
                headers = {
                    "Authorization": f"Bearer {self._force_refresh_after_401()}"
                }
                params = params
                continue

            if r.status_code == 429 and attempt < 3:
                retry_after = r.headers.get("Retry-After", "1")
                try:
                    delay = max(1.0, min(float(retry_after), 30.0))
                except ValueError:
                    delay = 1.0
                time.sleep(delay)
                continue

            if 500 <= r.status_code < 600 and attempt < 3:
                time.sleep(1.0 + attempt)
                continue

            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Respuesta inesperada de Spotify para {url}")
            return data

        raise RuntimeError(f"Spotify no respondió correctamente: {url}")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = path if path.startswith("http") else API_ROOT + path
        headers = {
            "Authorization": f"Bearer {self.token()}",
            "Content-Type": "application/json",
        }

        for attempt in range(4):
            r = self.session.post(url, json=payload, headers=headers, timeout=25)

            if r.status_code == 401:
                headers["Authorization"] = (
                    f"Bearer {self._force_refresh_after_401()}"
                )
                continue

            if r.status_code == 429 and attempt < 3:
                retry_after = r.headers.get("Retry-After", "1")
                try:
                    delay = max(1.0, min(float(retry_after), 30.0))
                except ValueError:
                    delay = 1.0
                time.sleep(delay)
                continue

            if 500 <= r.status_code < 600 and attempt < 3:
                time.sleep(1.0 + attempt)
                continue

            r.raise_for_status()
            if not r.content:
                return {}
            data = r.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Respuesta inesperada de Spotify para {url}")
            return data

        raise RuntimeError(f"Spotify no respondió correctamente: {url}")

    def create_missing_playlist(
        self,
        source_playlist_name: str,
        tracks: list[Track],
        *,
        name: str | None = None,
    ) -> dict[str, str]:
        # Preserve the source playlist exactly, including duplicate occurrences.
        # Spotify accepts repeated URIs in a playlist, so every missing occurrence
        # is appended in the same order in which it appeared in the source.
        uris = [track.uri for track in tracks if track.uri]

        if not uris:
            raise ValueError("No hay tracks para crear la playlist de faltantes")

        playlist_name = name or (
            f"No disponibles en Soulseek - {source_playlist_name} - "
            + time.strftime("%Y-%m-%d %H%M")
        )
        created = self._post(
            "/me/playlists",
            {
                "name": playlist_name,
                "public": False,
                "description": (
                    "Temas no encontrados en el preflight de Soulseek. "
                    f"Fuente: {source_playlist_name}."
                ),
            },
        )
        playlist_id = str(created.get("id") or "")
        if not playlist_id:
            raise RuntimeError("Spotify no devolvió ID para la playlist creada")

        for start in range(0, len(uris), 100):
            self._post(
                f"/playlists/{playlist_id}/items",
                {"uris": uris[start : start + 100]},
            )

        external_urls = created.get("external_urls") or {}
        url = str(
            external_urls.get("spotify")
            or f"https://open.spotify.com/playlist/{playlist_id}"
        )
        return {"id": playlist_id, "name": playlist_name, "url": url}

    def _album_details(self, album_id: str | None) -> dict[str, Any]:
        if not album_id:
            return {}
        if album_id not in self._album_cache:
            self._album_cache[album_id] = self._get(f"/albums/{album_id}")
        return self._album_cache[album_id]

    def playlist_tracks(self, playlist: str) -> tuple[str, list[Track]]:
        pid = playlist.strip()
        if "spotify:playlist:" in pid:
            pid = pid.rsplit(":", 1)[-1]
        elif "open.spotify.com" in pid:
            parts = [p for p in urllib.parse.urlparse(pid).path.split("/") if p]
            if len(parts) >= 2 and parts[0] == "playlist":
                pid = parts[1]

        if not pid or not all(ch.isalnum() for ch in pid):
            raise ValueError(f"No pude interpretar el ID de playlist: {playlist}")

        info = self._get(f"/playlists/{pid}", params={"fields": "id,name"})
        name = str(info.get("name") or pid)
        tracks: list[Track] = []
        url: str | None = f"/playlists/{pid}/items"
        params: dict[str, Any] | None = {
            "limit": 50,
            "offset": 0,
            "additional_types": "track",
        }

        while url:
            page = self._get(url, params=params)
            params = None
            for item in page.get("items") or []:
                if not isinstance(item, dict):
                    continue
                t = item.get("track") or item.get("item")
                if not isinstance(t, dict) or not t.get("id") or t.get("is_local"):
                    continue

                album = t.get("album") or {}
                album_id = album.get("id")
                album_extra = self._album_details(album_id)
                images = album.get("images") or album_extra.get("images") or []
                cover = (
                    images[0].get("url")
                    if images and isinstance(images[0], dict)
                    else None
                )
                ext = t.get("external_ids") or {}
                copyrights = [
                    str(x.get("text"))
                    for x in album_extra.get("copyrights") or []
                    if isinstance(x, dict) and x.get("text")
                ]

                tracks.append(
                    Track(
                        spotify_id=str(t["id"]),
                        uri=str(t.get("uri") or f"spotify:track:{t['id']}"),
                        url=str(
                            (t.get("external_urls") or {}).get("spotify")
                            or f"https://open.spotify.com/track/{t['id']}"
                        ),
                        title=str(t.get("name") or ""),
                        artists=[
                            str(a["name"])
                            for a in t.get("artists") or []
                            if a.get("name")
                        ],
                        album=str(album.get("name") or ""),
                        album_id=str(album_id) if album_id else None,
                        album_artists=[
                            str(a["name"])
                            for a in album.get("artists") or []
                            if a.get("name")
                        ],
                        release_date=album.get("release_date"),
                        track_number=t.get("track_number"),
                        total_tracks=album.get("total_tracks"),
                        disc_number=t.get("disc_number"),
                        duration_ms=int(t.get("duration_ms") or 0),
                        isrc=str(ext.get("isrc")) if ext.get("isrc") else None,
                        cover_url=str(cover) if cover else None,
                        explicit=bool(t.get("explicit")),
                        label=(
                            str(album_extra.get("label"))
                            if album_extra.get("label")
                            else None
                        ),
                        copyrights=copyrights,
                        playlist_id=pid,
                        playlist_name=name,
                        added_at=item.get("added_at"),
                    )
                )
            url = str(page.get("next")) if page.get("next") else None

        return name, tracks


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_VERSION_RE = [
    ("extended", re.compile(r"\bextended(?:\s+(?:mix|version))?\b", re.I)),
    ("radio", re.compile(r"\bradio\s+(?:edit|mix|version)\b", re.I)),
    ("club", re.compile(r"\bclub\s+(?:mix|version)\b", re.I)),
    ("original", re.compile(r"\boriginal\s+mix\b", re.I)),
    ("remix", re.compile(r"\bremix\b", re.I)),
    ("edit", re.compile(r"\bedit\b", re.I)),
    ("live", re.compile(r"\blive\b", re.I)),
    ("acoustic", re.compile(r"\bacoustic\b", re.I)),
    ("instrumental", re.compile(r"\binstrumental\b", re.I)),
    ("bootleg", re.compile(r"\bbootleg\b", re.I)),
    ("vip", re.compile(r"\bvip\b", re.I)),
    ("cover", re.compile(r"\bcover\b", re.I)),
    ("dub", re.compile(r"\bdub\b", re.I)),
    ("sped_up", re.compile(r"\bsped\s*up\b", re.I)),
    ("slowed", re.compile(r"\bslowed\b", re.I)),
]

_CONFLICT = {
    "radio",
    "club",
    "remix",
    "edit",
    "live",
    "acoustic",
    "instrumental",
    "bootleg",
    "vip",
    "cover",
    "dub",
    "sped_up",
    "slowed",
}

# Si Spotify explicita una variante de este grupo, el candidato debe contenerla.
_REQUIRED_TARGET_VARIANTS = _CONFLICT | {"extended"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def variants(s: str) -> set[str]:
    return {name for name, pat in _VERSION_RE if pat.search(s or "")}


def strip_variants(s: str) -> str:
    for _, pat in _VERSION_RE:
        s = pat.sub(" ", s)
    return norm(s)


def strip_extended_only(s: str) -> str:
    out = s
    for name, pat in _VERSION_RE:
        if name == "extended":
            out = pat.sub(" ", out)
    return norm(out)


def sim(a: str, b: str) -> float:
    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # No premiar automáticamente substrings. "Sun" y "Sunrise" no son
    # identidad suficiente para esta tarea.
    return SequenceMatcher(None, a, b).ratio()


def _candidate_stem(f: LocalFile) -> str:
    stem = f.path.stem
    return re.sub(r"^\s*\d{1,3}[\s._-]+", "", stem)


def _candidate_title(track: Track, f: LocalFile) -> str:
    if f.tag_title:
        return f.tag_title
    stem = _candidate_stem(f)
    normalized = norm(stem)
    for artist in sorted(track.artists, key=len, reverse=True):
        a = norm(artist)
        if a:
            normalized = re.sub(rf"\b{re.escape(a)}\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _artist_similarity(track: Track, f: LocalFile) -> float:
    target = [norm(a) for a in track.artists if norm(a)]
    local = [norm(a) for a in (f.tag_artists or []) if norm(a)]
    if local:
        return max((sim(a, b) for a in target for b in local), default=0.0)

    filename = norm(f.path.name)
    exact_presence = [
        1.0
        for a in target
        if a and re.search(rf"\b{re.escape(a)}\b", filename)
    ]
    return max(exact_presence, default=0.0)


def score_candidate(
    track: Track,
    f: LocalFile,
    prefer_extended: bool = True,
) -> tuple[bool, float, bool, list[str]]:
    """Devuelve (aceptado, score, es_extended, razones)."""
    reasons: list[str] = []

    fmt = f.format.casefold()
    if fmt == "mp3":
        if (f.bitrate_kbps or 0) < 315:
            return False, 0.0, False, [
                f"MP3 < 320 kbps ({f.bitrate_kbps or 0:.1f})"
            ]
        quality = 3.0
    elif fmt == "flac":
        quality = 2.0
    elif fmt == "wav":
        quality = 1.0
    else:
        return False, 0.0, False, [f"formato no soportado: {fmt}"]

    title_raw = _candidate_title(track, f)
    base_t = strip_variants(track.title)
    base_c = strip_variants(title_raw)
    title_s = sim(track.title, title_raw)
    base_s = sim(base_t, base_c)
    art_s = _artist_similarity(track, f)

    if base_s < 0.88:
        return False, 0.0, False, [f"título base mismatch ({base_s:.3f})"]
    if art_s < 0.88:
        return False, 0.0, False, [f"artista mismatch ({art_s:.3f})"]

    tv = variants(track.title)
    cv = variants(title_raw + " " + _candidate_stem(f))

    missing_required = sorted((tv & _REQUIRED_TARGET_VARIANTS) - cv)
    if missing_required:
        return False, 0.0, False, [
            "faltan variantes exigidas por Spotify: " + ", ".join(missing_required)
        ]

    # Extended es la única variante adicional permitida porque el objetivo del
    # script es preferirla cuando Spotify contiene la versión corta.
    conflicts = sorted((cv - tv) & _CONFLICT)
    if conflicts:
        return False, 0.0, False, [
            "versión conflictiva: " + ", ".join(conflicts)
        ]

    # Para un remix/club/etc. no basta con compartir la palabra "Remix".
    # Comparamos la identidad completa quitando únicamente "Extended" para
    # permitir, por ejemplo, "DJ Foo Remix" -> "DJ Foo Extended Remix".
    identity_variants = tv & _CONFLICT
    if identity_variants:
        version_identity = sim(
            strip_extended_only(track.title),
            strip_extended_only(title_raw),
        )
        if version_identity < 0.93:
            return False, 0.0, False, [
                f"identidad de versión/remixer mismatch ({version_identity:.3f})"
            ]
        reasons.append(f"version_identity={version_identity:.3f}")

    explicit_extended = "extended" in cv and "extended" not in tv
    inferred_extended = False
    length = f.length_s
    dur = track.duration_s
    dur_s = 0.0

    if length is not None and dur > 0:
        delta = length - dur
        ratio = length / dur

        if "extended" in tv:
            # Spotify ya apunta a una Extended. Debe tener duración cercana.
            if abs(delta) > 15:
                return False, 0.0, False, [
                    f"duración mismatch para Extended de Spotify ({delta:+.1f}s)"
                ]
            dur_s = max(0.0, 1.0 - abs(delta) / 15.0)
        elif explicit_extended:
            if delta < 12 or ratio > 1.9:
                return False, 0.0, False, [
                    f"extended con duración rara ({delta:+.1f}s, {ratio:.2f}x)"
                ]
            dur_s = 1.0
        elif abs(delta) <= 8:
            dur_s = max(0.0, 1.0 - abs(delta) / 8.0)
        elif (
            prefer_extended
            and delta >= 20
            and ratio <= 1.9
            and base_s >= 0.985
            and art_s >= 0.985
            and not (cv - {"original"})
        ):
            inferred_extended = True
            dur_s = 0.9
            reasons.append(f"extended inferido (+{delta:.1f}s)")
        else:
            return False, 0.0, False, [f"duración mismatch ({delta:+.1f}s)"]
    else:
        reasons.append("sin duración local: confianza reducida")

    is_ext = explicit_extended or inferred_extended or "extended" in tv

    isrc_match = False
    if track.isrc and f.isrc:
        isrc_match = norm(track.isrc) == norm(f.isrc)
        if not isrc_match and not is_ext:
            return False, 0.0, False, [
                f"ISRC distinto: local={f.isrc} Spotify={track.isrc}"
            ]
        if isrc_match:
            reasons.append("ISRC exacto")
        else:
            reasons.append(
                "ISRC distinto permitido sólo porque la Extended puede tener ISRC propio"
            )

    score = (
        45.0 * base_s
        + 25.0 * art_s
        + 15.0 * title_s
        + 10.0 * dur_s
        + quality
    )

    if isrc_match and not is_ext:
        score += 40.0
    if is_ext and prefer_extended and "extended" not in tv:
        score += 20.0

    reasons += [
        f"base={base_s:.3f}",
        f"art={art_s:.3f}",
        f"title={title_s:.3f}",
        f"score={score:.3f}",
    ]
    return True, score, is_ext, reasons


# ---------------------------------------------------------------------------
# Inspección local
# ---------------------------------------------------------------------------

def _frame_text(value: Any) -> str | None:
    if value is None:
        return None
    text = getattr(value, "text", value)
    if isinstance(text, (list, tuple)):
        return str(text[0]).strip() if text else None
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace").strip()
    rendered = str(text).strip()
    return rendered or None


def inspect(path: Path) -> LocalFile | None:
    try:
        audio = mutagen.File(path)
        if not audio or not getattr(audio, "info", None):
            return None
        info = audio.info
        br = getattr(info, "bitrate", None)
        tags = getattr(audio, "tags", None)

        def txt(*keys):
            if not tags:
                return None
            for k in keys:
                try:
                    v = tags.get(k)
                except Exception:
                    v = None
                rendered = _frame_text(v)
                if rendered:
                    return rendered
            return None

        artist_raw = txt("TPE1", "artist", "ARTIST")
        artists = None
        if artist_raw:
            artists = [
                part.strip()
                for part in re.split(r"\s*(?:;|\u001f)\s*", artist_raw)
                if part.strip()
            ]

        return LocalFile(
            path=path,
            length_s=float(getattr(info, "length", 0) or 0) or None,
            bitrate_kbps=(float(br) / 1000.0) if br else None,
            format=path.suffix.lstrip(".").lower(),
            tag_title=txt("TIT2", "title", "TITLE"),
            tag_artists=artists,
            isrc=txt("TSRC", "isrc", "ISRC"),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Soulseek
# ---------------------------------------------------------------------------
# Esta sección conserva deliberadamente el mecanismo del script original.

def find_sockseek() -> str:
    for name in ("sockseek", "sldl"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError(
        "No se encontró 'sockseek' ni 'sldl' en el PATH.\n"
        "Descárgalo de https://github.com/fiso64/sockseek/releases"
    )


def _sockseek_query(track: Track) -> str:
    query = f"{track.primary_artist} - {track.title}"
    if track.duration_ms:
        query += f", length={max(1, track.duration_ms // 1000)}"
    return query


def _availability_candidate_summary(item: dict[str, Any]) -> dict[str, Any]:
    user = item.get("User") or {}
    file_info = item.get("File") or {}
    filename = str(file_info.get("Filename") or "")
    suffix = Path(filename.replace("\\", "/")).suffix.casefold()
    return {
        "username": user.get("Username"),
        "upload_speed_mib_s": user.get("UploadSpeed"),
        "has_free_upload_slot": user.get("HasFreeUploadSlot"),
        "filename": filename,
        "format": suffix.lstrip("."),
        "length_s": file_info.get("Length"),
        "bitrate_kbps": file_info.get("Bitrate"),
        "sample_rate": file_info.get("SampleRate"),
        "bit_depth": file_info.get("BitDepth"),
        "size_bytes": file_info.get("Size"),
    }


def _classify_availability_raw(raw: Any) -> AvailabilityCheck:
    """Classify one Sockseek JSON result array without touching the network."""
    if not isinstance(raw, list):
        return AvailabilityCheck(
            status="error",
            result_count=0,
            candidates=[],
            error="sockseek --print json-all no devolvió una lista JSON para el tema",
        )

    summaries = [
        _availability_candidate_summary(item)
        for item in raw
        if isinstance(item, dict)
    ]

    usable: list[dict[str, Any]] = []
    quality_unknown = False
    for candidate in summaries:
        fmt = str(candidate.get("format") or "").casefold()
        bitrate = candidate.get("bitrate_kbps")
        if fmt in {"flac", "wav"}:
            usable.append(candidate)
        elif fmt == "mp3":
            if bitrate is None:
                # Some Soulseek clients do not advertise bitrate. Do not create a false
                # negative; the downloaded file will still be verified locally.
                quality_unknown = True
                usable.append(candidate)
            else:
                try:
                    if float(bitrate) >= 315.0:
                        usable.append(candidate)
                except (TypeError, ValueError):
                    quality_unknown = True
                    usable.append(candidate)

    if usable:
        status = "available_quality_unknown" if quality_unknown else "available"
        return AvailabilityCheck(
            status=status,
            result_count=len(usable),
            candidates=usable[:10],
            quality_unknown=quality_unknown,
        )

    return AvailabilityCheck(
        status="missing",
        result_count=0,
        candidates=summaries[:10],
    )


def _decode_concatenated_json_values(stdout: str) -> list[Any]:
    """Decode consecutive compact JSON values emitted by Sockseek JobList printing.

    Sockseek prints one JSON value per child job for multi-item inputs.  The values
    are compact JSON arrays separated only by whitespace/newlines, so parsing the
    complete stdout with json.loads() is not sufficient.
    """
    text = (stdout or "").lstrip("\ufeff")
    decoder = json.JSONDecoder()
    values: list[Any] = []
    pos = 0
    n = len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        value, end = decoder.raw_decode(text, pos)
        values.append(value)
        pos = end
    return values


def _write_sockseek_preflight_csv(path: Path, tracks: list[Track]) -> None:
    """Write a structured multi-song input preserving Spotify playlist order."""
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Artist", "Title", "Length"])
        writer.writeheader()
        for track in tracks:
            writer.writerow(
                {
                    "Artist": track.primary_artist,
                    "Title": track.title,
                    "Length": max(1, int(round(track.duration_s))) if track.duration_ms else "",
                }
            )


def check_soulseek_availability_batch(
    tracks: list[Track],
    *,
    length_tol: int = 60,
    timeout: float = 600.0,
) -> list[AvailabilityCheck]:
    """Preflight all tracks in one Sockseek process and one Soulseek session.

    No audio is downloaded because --print json-all sets Sockseek to print-only
    mode.  The CSV row order is the Spotify playlist order, and Sockseek's JobList
    renderer emits one compact JSON array per child job in that same order.
    """
    if not tracks:
        return []

    bin_ = find_sockseek()
    with tempfile.TemporaryDirectory(prefix="sockseek_preflight_") as tmp:
        batch_csv = Path(tmp) / "spotify_preflight.csv"
        _write_sockseek_preflight_csv(batch_csv, tracks)

        cmd = [
            bin_,
            str(batch_csv),
            "--strict-title",
            "--strict-artist",
            f"--length-tol={length_tol}",
            f"--pref-length-tol={length_tol}",
            # Required in preflight because the finalizer only supports these formats.
            "--format=flac,wav,mp3",
            "--min-bitrate=315",
            "--pref-format=flac,wav,mp3",
            "--pref-min-bitrate=320",
            "--print",
            "json-all",
        ]

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            message = f"timeout del preflight Soulseek batch ({timeout:.0f}s)"
            return [
                AvailabilityCheck(
                    status="error",
                    result_count=0,
                    candidates=[],
                    error=message,
                )
                for _ in tracks
            ]
        except FileNotFoundError:
            return [
                AvailabilityCheck(
                    status="error",
                    result_count=0,
                    candidates=[],
                    error="sockseek no encontrado",
                )
                for _ in tracks
            ]

        stdout = (completed.stdout or "").lstrip("\ufeff").strip()
        try:
            raw_values = _decode_concatenated_json_values(stdout)
        except json.JSONDecodeError as exc:
            detail = (completed.stderr or stdout or "sin salida").strip()
            message = f"salida JSON batch inválida de sockseek: {exc}; {detail[:500]}"
            return [
                AvailabilityCheck(
                    status="error",
                    result_count=0,
                    candidates=[],
                    error=message,
                )
                for _ in tracks
            ]

        if len(raw_values) != len(tracks):
            detail = (completed.stderr or "").strip()
            message = (
                "sockseek devolvió un número inesperado de resultados de preflight: "
                f"{len(raw_values)} para {len(tracks)} temas"
            )
            if detail:
                message += f"; {detail[:500]}"
            return [
                AvailabilityCheck(
                    status="error",
                    result_count=0,
                    candidates=[],
                    error=message,
                )
                for _ in tracks
            ]

        return [_classify_availability_raw(raw) for raw in raw_values]


def check_soulseek_availability(
    track: Track,
    *,
    length_tol: int = 60,
    timeout: float = 90.0,
) -> AvailabilityCheck:
    """Compatibility helper for a one-track preflight."""
    return check_soulseek_availability_batch(
        [track], length_tol=length_tol, timeout=timeout
    )[0]

def write_missing_list(path: Path, missing: list[tuple[int, Track]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Temas no disponibles en Soulseek durante el preflight",
        "=" * 54,
        "",
    ]
    if not missing:
        lines.append("Ninguno.")
    else:
        for position, track in missing:
            lines.append(
                f"{position:4d}. {track.primary_artist} - {track.title} | {track.url}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _duration_hms(duration_ms: int) -> str:
    total_seconds = max(0, int(round(duration_ms / 1000.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def write_missing_json(
    path: Path,
    *,
    source_playlist_name: str,
    source_playlist: str,
    missing: list[tuple[int, Track]],
    generated_playlist: dict[str, str] | None = None,
) -> None:
    """Write missing tracks in original playlist order with full Spotify metadata."""
    tracks = []
    for position, track in missing:
        metadata = _track_report(track)
        tracks.append(
            {
                "playlist_position": position,
                "duration_ms": track.duration_ms,
                "duration_seconds": round(track.duration_s, 3),
                "duration_hms": _duration_hms(track.duration_ms),
                "spotify": metadata,
            }
        )

    payload = {
        "source_playlist": {
            "name": source_playlist_name,
            "source": source_playlist,
        },
        "generated_missing_playlist": generated_playlist,
        "missing_count": len(tracks),
        "order": "source_playlist_order",
        "tracks": tracks,
    }
    write_report(path, payload)


def download_with_sockseek(
    track: Track,
    out_dir: Path,
    length_tol: int = 60,
) -> list[Path]:
    bin_ = find_sockseek()
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in out_dir.rglob("*") if p.is_file()}

    query = _sockseek_query(track)

    cmd = [
        bin_,
        query,
        "--song",
        "--strict-title",
        "--strict-artist",
        f"--length-tol={length_tol}",
        f"--pref-length-tol={length_tol}",
        "--pref-format=flac,wav,mp3",
        "--pref-min-bitrate=320",
        f"--output-dir={out_dir}",
        "--name-format={artist} - {title}",
    ]
    print(f"    → sockseek: {query}")
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=360,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("    ! timeout sockseek")
    except FileNotFoundError:
        raise RuntimeError("sockseek no encontrado")

    after = [
        p
        for p in out_dir.rglob("*")
        if p.is_file()
        and p.resolve() not in before
        and p.suffix.lower() in {".mp3", ".flac", ".wav", ".m4a"}
    ]
    return sorted(after)


# ---------------------------------------------------------------------------
# Finalización MP3 + metadata + cover
# ---------------------------------------------------------------------------

def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s or "Unknown"


def download_cover(
    url: str | None,
    *,
    retries: int = 3,
    max_bytes: int = 15 * 1024 * 1024,
) -> tuple[bytes, str] | None:
    if not url:
        return None

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "traktor-spotify-import/2.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise RuntimeError("cover demasiado grande")
                ctype = (
                    (r.headers.get("Content-Type") or "image/jpeg")
                    .split(";")[0]
                    .strip()
                )
                if ctype not in ("image/jpeg", "image/png", "image/webp"):
                    raise RuntimeError(f"Content-Type de cover inesperado: {ctype}")
                if not data:
                    raise RuntimeError("cover vacío")
                return data, ctype
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"no se pudo descargar el cover de Spotify: {last_error}")


def _existing_spotify_track_id(path: Path) -> str | None:
    try:
        tags = ID3(path)
        for frame in tags.getall("TXXX"):
            if frame.desc == "Spotify Track ID" and frame.text:
                return str(frame.text[0])
    except Exception:
        return None
    return None


def _validate_existing_output(
    path: Path,
    spotify_id: str,
    *,
    require_cover: bool,
) -> None:
    try:
        audio = MP3(path)
        if audio.info.bitrate < 315_000:
            raise RuntimeError(f"bitrate existente bajo: {audio.info.bitrate}")
        tags = ID3(path)
    except Exception as exc:
        raise RuntimeError(
            f"output existente inválido; usa --overwrite para regenerarlo: {path}: {exc}"
        ) from exc

    stored_id = None
    for frame in tags.getall("TXXX"):
        if frame.desc == "Spotify Track ID" and frame.text:
            stored_id = str(frame.text[0])
            break
    if stored_id != spotify_id:
        raise RuntimeError(
            f"output existente pertenece a otro Spotify Track ID: {path}"
        )
    if require_cover and not tags.getall("APIC"):
        raise RuntimeError(
            f"output existente no contiene cover; usa --overwrite para regenerarlo: {path}"
        )


def _unique_destination(
    out_dir: Path,
    track: Track,
    extended: bool,
    overwrite: bool,
) -> tuple[Path, bool]:
    title = safe_name(track.title)
    if extended and "extended" not in track.title.casefold():
        title += " [Extended]"
    base = out_dir / f"{safe_name(track.primary_artist)} - {title}.mp3"

    if not base.exists():
        return base, False

    existing_id = _existing_spotify_track_id(base)
    if existing_id == track.spotify_id and not overwrite:
        return base, True

    if overwrite:
        return base, False

    alt = out_dir / (
        f"{safe_name(track.primary_artist)} - {title} [{track.spotify_id}].mp3"
    )
    if alt.exists():
        alt_id = _existing_spotify_track_id(alt)
        if alt_id == track.spotify_id:
            return alt, True
        raise RuntimeError(f"colisión de salida no resoluble: {alt}")
    return alt, False


def write_tags(
    mp3_path: Path,
    track: Track,
    source: LocalFile,
    extended: bool,
    cover: tuple[bytes, str] | None,
):
    tags = ID3()
    tags.add(TIT2(encoding=3, text=track.title))
    tags.add(TPE1(encoding=3, text=track.artists or [track.primary_artist]))
    if track.album_artists:
        tags.add(TPE2(encoding=3, text=track.album_artists))
    tags.add(TALB(encoding=3, text=track.album))
    if track.release_date:
        tags.add(TDRC(encoding=3, text=track.release_date))
    if track.track_number:
        tr = str(track.track_number)
        if track.total_tracks:
            tr += f"/{track.total_tracks}"
        tags.add(TRCK(encoding=3, text=tr))
    if track.disc_number:
        tags.add(TPOS(encoding=3, text=str(track.disc_number)))

    # No atribuir a una Extended el ISRC de la versión corta de Spotify.
    tag_isrc: str | None
    if extended:
        tag_isrc = source.isrc
    else:
        tag_isrc = track.isrc or source.isrc
    if tag_isrc:
        tags.add(TSRC(encoding=3, text=tag_isrc))

    if track.label:
        tags.add(TPUB(encoding=3, text=track.label))
    if track.copyrights:
        tags.add(TCOP(encoding=3, text=" | ".join(track.copyrights)))

    custom = {
        "Spotify Track ID": track.spotify_id,
        "Spotify URI": track.uri,
        "Spotify URL": track.url,
        "Spotify Album ID": track.album_id or "",
        "Spotify Playlist ID": track.playlist_id or "",
        "Spotify Playlist Name": track.playlist_name or "",
        "Spotify Added At": track.added_at or "",
        "Spotify Explicit": "1" if track.explicit else "0",
        "Spotify Original Duration ms": str(track.duration_ms),
        "Spotify Cover URL": track.cover_url or "",
        "Spotify ISRC": track.isrc or "",
        "Local Source ISRC": source.isrc or "",
        "Local Source Filename": source.path.name,
        "Version": "Extended" if extended else "Spotify-length",
    }
    for desc, val in custom.items():
        if val:
            tags.add(TXXX(encoding=3, desc=desc, text=val))

    if cover:
        data, mime = cover
        tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=data))

    tags.save(mp3_path, v2_version=3)


def finalize(
    src: Path,
    out_dir: Path,
    track: Track,
    source: LocalFile,
    extended: bool,
    *,
    overwrite: bool = False,
    require_cover: bool = True,
) -> tuple[Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest, already_exists = _unique_destination(
        out_dir,
        track,
        extended,
        overwrite,
    )
    if already_exists:
        _validate_existing_output(
            dest,
            track.spotify_id,
            require_cover=require_cover,
        )
        return dest, True

    if src.resolve() != source.path.resolve():
        raise RuntimeError("el source seleccionado cambió antes de finalizar")
    if not src.is_file():
        raise RuntimeError(f"el source seleccionado no existe: {src}")

    checked = inspect(src)
    if checked is None:
        raise RuntimeError("no se pudo revalidar el source antes de finalizar")
    if source.length_s and checked.length_s and abs(source.length_s - checked.length_s) > 2:
        raise RuntimeError("la duración del source cambió después del matching")
    if src.suffix.lower() == ".mp3" and (checked.bitrate_kbps or 0) < 315:
        raise RuntimeError(
            f"MP3 source cayó por debajo de 320 kbps: {checked.bitrate_kbps or 0:.1f}"
        )

    cover = download_cover(track.cover_url)
    if require_cover and cover is None:
        raise RuntimeError("Spotify no proporcionó cover utilizable")

    with tempfile.TemporaryDirectory(prefix="ssdl_") as tmp:
        tmp_mp3 = Path(tmp) / dest.name

        if src.suffix.lower() == ".mp3":
            # Mantener intactos los frames de audio. Sólo se reemplaza ID3 después.
            shutil.copy2(src, tmp_mp3)
        elif src.suffix.lower() in {".flac", ".wav"}:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-map_metadata",
                "-1",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "320k",
                str(tmp_mp3),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except FileNotFoundError:
                raise RuntimeError("ffmpeg no está en el PATH")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffmpeg falló: {(e.stderr or "sin stderr")[:300]}")
        else:
            raise RuntimeError(f"formato no soportado al finalizar: {src.suffix}")

        try:
            info = MP3(tmp_mp3).info
            if info.bitrate < 315_000:
                raise RuntimeError(f"bitrate final bajo: {info.bitrate}")
        except Exception as e:
            raise RuntimeError(f"no se pudo verificar MP3: {e}")

        write_tags(tmp_mp3, track, source, extended, cover)

        if dest.exists():
            if overwrite:
                dest.unlink()
            else:
                raise RuntimeError(f"la salida apareció durante la finalización: {dest}")
        shutil.move(str(tmp_mp3), dest)

    return dest, False


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _track_report(track: Track) -> dict[str, Any]:
    data = asdict(track)
    data["duration_seconds"] = round(track.duration_s, 3)
    data["duration_hms"] = _duration_hms(track.duration_ms)
    return data


def _candidate_report(
    candidate: LocalFile,
    accepted: bool,
    score: float,
    extended: bool,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "path": str(candidate.path),
        "format": candidate.format,
        "length_s": candidate.length_s,
        "bitrate_kbps": candidate.bitrate_kbps,
        "tag_title": candidate.tag_title,
        "tag_artists": candidate.tag_artists,
        "isrc": candidate.isrc,
        "accepted": accepted,
        "score": round(score, 5),
        "extended": extended,
        "reasons": reasons,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def wait_for_download_gate(path: Path, timeout: float) -> bool:
    """Wait until an orchestrator releases downloads by creating ``path``."""
    print(f"\nDescarga en espera de la señal del orquestador: {path}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            print("Señal recibida. Comenzando las descargas de Soulseek.")
            return True
        time.sleep(0.25)
    print(
        f"Timeout esperando la señal para iniciar descargas: {path}",
        file=sys.stderr,
    )
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Spotify playlist → preflight Soulseek completo → lista de faltantes → "
            "descarga de disponibles → MP3 320 con verificación, metadata + cover"
        )
    )
    ap.add_argument("playlist", help="URL, URI o ID de la playlist de Spotify")
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Carpeta de salida de los MP3 finales",
    )
    ap.add_argument(
        "--client-id",
        default=os.environ.get("SPOTIFY_CLIENT_ID"),
        help="Spotify Client ID (o variable SPOTIFY_CLIENT_ID)",
    )
    ap.add_argument(
        "--staging",
        default=None,
        help="Carpeta temporal de descargas Soulseek",
    )
    ap.add_argument(
        "--length-tol",
        type=int,
        default=60,
        help="Tolerancia de duración para sockseek (default 60s)",
    )
    ap.add_argument(
        "--no-extended",
        action="store_true",
        help="No preferir versiones Extended",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Ejecutar todo el preflight Soulseek y generar reportes, pero no crear "
            "playlist en Spotify ni descargar audio"
        ),
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Ejecutar el preflight completo, crear la playlist Spotify de faltantes "
            "e imprimir su URL, pero no descargar audio"
        ),
    )
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Limitar número de tracks (pruebas)",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Ruta del JSON de auditoría final",
    )
    ap.add_argument(
        "--availability-report",
        default=None,
        help="Ruta del JSON del preflight (default: output/soulseek_availability.json)",
    )
    ap.add_argument(
        "--missing-list",
        default=None,
        help="Ruta TXT de no disponibles (default: output/soulseek_missing.txt)",
    )
    ap.add_argument(
        "--missing-json",
        default=None,
        help=(
            "Ruta JSON de no disponibles con metadata Spotify completa y orden de "
            "playlist (default: output/soulseek_missing.json)"
        ),
    )
    ap.add_argument(
        "--no-missing-playlist",
        action="store_true",
        help="No crear playlist privada de Spotify con los temas faltantes",
    )
    ap.add_argument(
        "--missing-playlist-name",
        default=None,
        help="Nombre personalizado para la playlist Spotify de faltantes",
    )
    ap.add_argument(
        "--preflight-timeout",
        type=float,
        default=600.0,
        help=(
            "Timeout total para el batch único de preflight Soulseek "
            "(default 600s)"
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Permitir reemplazar un output existente con el mismo nombre",
    )
    ap.add_argument(
        "--allow-missing-cover",
        action="store_true",
        help="No fallar si Spotify no entrega/descarga el cover",
    )
    ap.add_argument(
        "--download-gate",
        default=None,
        help=(
            "Esperar a que exista este archivo después del preflight y antes "
            "de comenzar las descargas"
        ),
    )
    ap.add_argument(
        "--download-gate-timeout",
        type=float,
        default=900.0,
        help="Timeout de --download-gate en segundos (default 900)",
    )
    args = ap.parse_args()

    if args.dry_run and args.preflight_only:
        print(
            "Usa sólo uno de --dry-run o --preflight-only.",
            file=sys.stderr,
        )
        return 2

    if not args.client_id:
        print(
            "Falta Client ID. Usa --client-id o exporta SPOTIFY_CLIENT_ID",
            file=sys.stderr,
        )
        return 2

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else out / "spotify_reconcile_report.json"
    )
    availability_report_path = (
        Path(args.availability_report).expanduser().resolve()
        if args.availability_report
        else out / "soulseek_availability.json"
    )
    missing_list_path = (
        Path(args.missing_list).expanduser().resolve()
        if args.missing_list
        else out / "soulseek_missing.txt"
    )
    missing_json_path = (
        Path(args.missing_json).expanduser().resolve()
        if args.missing_json
        else out / "soulseek_missing.json"
    )

    print("Conectando a Spotify…")
    try:
        sp = Spotify(args.client_id)
        pl_name, tracks = sp.playlist_tracks(args.playlist)
    except Exception as exc:
        print(f"Spotify falló: {exc}", file=sys.stderr)
        return 2

    if args.max:
        tracks = tracks[: args.max]
    print(f"Playlist: «{pl_name}» — {len(tracks)} temas")

    try:
        find_sockseek()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # FASE 1: PRE-FLIGHT COMPLETO. No se descarga ningún archivo aquí.
    # ------------------------------------------------------------------
    print("\n=== PRE-FLIGHT SOULSEEK: verificando toda la playlist ===")
    availability_rows: list[dict[str, Any]] = []
    available_tracks: list[tuple[int, Track, AvailabilityCheck]] = []
    missing_tracks: list[tuple[int, Track]] = []
    preflight_errors: list[tuple[int, Track, str]] = []
    print(
        f"Lanzando {len(tracks)} consultas como un único batch de Sockseek..."
    )
    batch_checks = check_soulseek_availability_batch(
        tracks,
        length_tol=args.length_tol,
        timeout=args.preflight_timeout,
    )

    for i, (track, check) in enumerate(zip(tracks, batch_checks), 1):
        print(f"[{i}/{len(tracks)}] {track.primary_artist} - {track.title}")

        if check.status == "available":
            marker = "✓ AVAILABLE"
            available_tracks.append((i, track, check))
        elif check.status == "available_quality_unknown":
            marker = "? AVAILABLE (bitrate no anunciado)"
            available_tracks.append((i, track, check))
        elif check.status == "missing":
            marker = "✗ MISSING"
            missing_tracks.append((i, track))
        else:
            marker = "! ERROR"
            preflight_errors.append((i, track, check.error or "error desconocido"))

        print(f"    {marker} | resultados utilizables: {check.result_count}")
        if check.error:
            print(f"    {check.error}")

        availability_rows.append(
            {
                "position": i,
                "spotify": _track_report(track),
                "status": check.status,
                "result_count": check.result_count,
                "quality_unknown": check.quality_unknown,
                "top_candidates": check.candidates,
                "error": check.error,
            }
        )

    # Escribir el reporte una sola vez porque el batch es atómico: no existe ya
    # un preflight parcial por tema dentro de procesos separados.
    write_report(
        availability_report_path,
        {
            "playlist": {"name": pl_name, "source": args.playlist},
            "total_tracks": len(tracks),
            "checked": len(availability_rows),
            "available": len(available_tracks),
            "missing": len(missing_tracks),
            "errors": len(preflight_errors),
            "preflight_mode": "single_sockseek_batch",
            "tracks": availability_rows,
        },
    )

    write_missing_list(missing_list_path, missing_tracks)
    write_missing_json(
        missing_json_path,
        source_playlist_name=pl_name,
        source_playlist=args.playlist,
        missing=missing_tracks,
        generated_playlist=None,
    )

    preflight_summary = {
        "playlist": {"name": pl_name, "source": args.playlist},
        "total_tracks": len(tracks),
        "checked": len(availability_rows),
        "available": len(available_tracks),
        "missing": len(missing_tracks),
        "errors": len(preflight_errors),
        "missing_list": str(missing_list_path),
        "missing_json": str(missing_json_path),
        "preflight_mode": "single_sockseek_batch",
        "tracks": availability_rows,
    }
    write_report(availability_report_path, preflight_summary)

    print("\n=== RESULTADO PRE-FLIGHT ===")
    print(f"Disponibles:     {len(available_tracks)}/{len(tracks)}")
    print(f"No disponibles:  {len(missing_tracks)}/{len(tracks)}")
    print(f"Errores consulta: {len(preflight_errors)}/{len(tracks)}")
    print(f"Lista faltantes: {missing_list_path}")
    print(f"JSON faltantes:  {missing_json_path}")
    print(f"Reporte:         {availability_report_path}")

    missing_playlist: dict[str, str] | None = None
    if missing_tracks and not args.dry_run and not args.no_missing_playlist:
        print("\nCreando playlist privada de Spotify con los temas no disponibles…")
        try:
            missing_playlist = sp.create_missing_playlist(
                pl_name,
                [track for _, track in missing_tracks],
                name=args.missing_playlist_name,
            )
            print("\n=== PLAYLIST SPOTIFY DE FALTANTES ===")
            print(f"Nombre: {missing_playlist['name']}")
            print(f"URL:    {missing_playlist['url']}")
            print("=======================================")
        except Exception as exc:
            # No impedir la descarga de los disponibles porque Spotify no haya podido
            # crear la playlist auxiliar. El fallo queda auditado.
            print(f"No se pudo crear la playlist de faltantes: {exc}", file=sys.stderr)
            missing_playlist = {"id": "", "name": "", "url": "", "error": str(exc)}

    # Reescribir el JSON de faltantes para incluir la playlist auxiliar creada
    # (o el error de creación), manteniendo el mismo orden de la playlist fuente.
    write_missing_json(
        missing_json_path,
        source_playlist_name=pl_name,
        source_playlist=args.playlist,
        missing=missing_tracks,
        generated_playlist=missing_playlist,
    )

    if args.dry_run:
        if missing_tracks:
            print(
                "\nPlaylist Spotify de faltantes: NO CREADA porque --dry-run "
                "no modifica tu cuenta de Spotify."
            )
            print(
                "Usa --preflight-only para crearla, imprimir su URL y terminar "
                "antes de descargar audio."
            )
        print("\nDry-run: preflight terminado. No se descargó audio.")
        return 0 if not preflight_errors else 4

    if args.preflight_only:
        print("\nPreflight-only: preflight terminado. No se descargó audio.")
        if missing_playlist and missing_playlist.get("url"):
            print(f"Playlist Spotify de faltantes: {missing_playlist['url']}")
        elif missing_tracks and args.no_missing_playlist:
            print("Playlist Spotify de faltantes: deshabilitada por --no-missing-playlist")
        elif missing_tracks:
            print("Playlist Spotify de faltantes: no pudo crearse")
        else:
            print("Playlist Spotify de faltantes: no necesaria; no hubo temas faltantes")
        return 0 if not preflight_errors else 4

    if preflight_errors:
        print(
            "\nHay consultas de disponibilidad con ERROR. No las trato como MISSING. "
            "La descarga continuará sólo con los temas confirmados como disponibles."
        )

    if args.download_gate:
        download_gate = Path(args.download_gate).expanduser().resolve()
        if not wait_for_download_gate(download_gate, args.download_gate_timeout):
            return 5

    if not shutil.which("ffmpeg"):
        print("ffmpeg no está en el PATH", file=sys.stderr)
        return 2

    staging = Path(
        args.staging or tempfile.mkdtemp(prefix="soulseek_staging_")
    ).expanduser().resolve()
    print(f"\nStaging Soulseek: {staging}")
    print(f"Salida final:     {out}\n")

    # ------------------------------------------------------------------
    # FASE 2: DESCARGA. Sólo comienza una vez finalizó TODO el preflight.
    # ------------------------------------------------------------------
    print("=== DESCARGA DE TEMAS CONFIRMADOS COMO DISPONIBLES ===")
    rows: list[dict[str, Any]] = []
    ok = 0
    failed = 0

    for download_index, (source_position, track, availability) in enumerate(
        available_tracks,
        1,
    ):
        print(
            f"[{download_index}/{len(available_tracks)} | Spotify #{source_position}] "
            f"{track.primary_artist} - {track.title}"
        )
        row: dict[str, Any] = {
            "position": source_position,
            "spotify": _track_report(track),
            "preflight": {
                "status": availability.status,
                "result_count": availability.result_count,
                "quality_unknown": availability.quality_unknown,
            },
            "status": "pending",
            "candidates": [],
        }

        try:
            # Conserva la función y flags de descarga Soulseek de la versión anterior.
            files = download_with_sockseek(
                track,
                staging,
                length_tol=args.length_tol,
            )
        except Exception as exc:
            print(f"    ✗ error descarga: {exc}")
            row["status"] = "download_error"
            row["error"] = str(exc)
            rows.append(row)
            failed += 1
            continue

        if not files:
            print("    ✗ preflight tenía resultados, pero la descarga no produjo archivo")
            row["status"] = "availability_changed_or_download_failed"
            rows.append(row)
            failed += 1
            continue

        candidates: list[LocalFile] = []
        for path in files:
            local = inspect(path)
            if local:
                candidates.append(local)

        best: LocalFile | None = None
        best_score = -1.0
        best_ext = False
        best_reasons: list[str] = []

        for candidate in candidates:
            accepted, score, extended, reasons = score_candidate(
                track,
                candidate,
                prefer_extended=not args.no_extended,
            )
            row["candidates"].append(
                _candidate_report(candidate, accepted, score, extended, reasons)
            )
            if accepted and score > best_score:
                best_score = score
                best = candidate
                best_ext = extended
                best_reasons = reasons

        if not best:
            print("    ✗ ningún candidato descargado pasó la verificación")
            for candidate in candidates:
                print(f"       - {candidate.path.name}")
            row["status"] = "verification_failed"
            rows.append(row)
            failed += 1
            continue

        print(
            f"    ✓ elegido: {best.path.name}"
            + ("  [Extended]" if best_ext else "")
        )
        row["selected"] = {
            "path": str(best.path),
            "score": best_score,
            "extended": best_ext,
            "reasons": best_reasons,
        }

        try:
            dest, already_exists = finalize(
                best.path,
                out,
                track,
                best,
                best_ext,
                overwrite=args.overwrite,
                require_cover=not args.allow_missing_cover,
            )
            if already_exists:
                print(f"    → ya existía: {dest.name}")
                row["status"] = "already_finalized"
            else:
                print(f"    → {dest.name}")
                row["status"] = "finalized"
            row["output"] = str(dest)
            ok += 1
        except Exception as exc:
            print(f"    ✗ finalize: {exc}")
            row["status"] = "finalize_error"
            row["error"] = str(exc)
            failed += 1

        rows.append(row)

        write_report(
            report_path,
            {
                "playlist": {"name": pl_name, "source": args.playlist},
                "preflight": {
                    "available": len(available_tracks),
                    "missing": len(missing_tracks),
                    "errors": len(preflight_errors),
                    "availability_report": str(availability_report_path),
                    "missing_list": str(missing_list_path),
                    "missing_json": str(missing_json_path),
                    "missing_playlist": missing_playlist,
                },
                "output": str(out),
                "staging": str(staging),
                "prefer_extended": not args.no_extended,
                "download_total": len(available_tracks),
                "completed": ok,
                "failed": failed,
                "tracks": rows,
            },
        )

    final_report = {
        "playlist": {"name": pl_name, "source": args.playlist},
        "preflight": {
            "available": len(available_tracks),
            "missing": len(missing_tracks),
            "errors": len(preflight_errors),
            "availability_report": str(availability_report_path),
            "missing_list": str(missing_list_path),
            "missing_json": str(missing_json_path),
            "missing_playlist": missing_playlist,
        },
        "output": str(out),
        "staging": str(staging),
        "prefer_extended": not args.no_extended,
        "download_total": len(available_tracks),
        "completed": ok,
        "failed": failed,
        "tracks": rows,
    }
    write_report(report_path, final_report)

    print(f"\nListo. {ok}/{len(available_tracks)} disponibles finalizados en {out}")
    print(f"No disponibles según preflight: {len(missing_tracks)}")
    print(f"JSON de faltantes: {missing_json_path}")
    if missing_playlist and missing_playlist.get("url"):
        print(f"Playlist Spotify de faltantes: {missing_playlist['url']}")
    print(f"Reporte final: {report_path}")

    # Éxito significa: todos los que estaban disponibles fueron finalizados y no hubo
    # errores de consulta. Los MISSING son un resultado válido del preflight.
    return 0 if ok == len(available_tracks) and not preflight_errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
