"""Minimal TuneIn (RadioTime OPML) client — stdlib only, JSON render.

    Search.ashx?query=…    → stations (guide_id, name, logo, bitrate, …)
    Tune.ashx?id=s…        → stream URLs
    Describe.ashx?id=s…    → station details (slogan, genre, location, now playing)

Geo-blocked or unsupported stations tune to a ``notcompatible`` placeholder
clip; those URLs are rejected so the caller can try the next hit.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

BASE = "https://opml.radiotime.com"
TIMEOUT_S = 4.0
USER_AGENT = "pawn-assistant/0.1 (+local voice assistant)"

Fetch = Callable[[str], Any]


class TuneInError(RuntimeError):
    pass


@dataclass
class Station:
    guide_id: str
    name: str
    subtext: str = ""
    logo: str = ""
    bitrate: int = 0
    formats: str = ""
    reliability: int = 0
    now_playing: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StationInfo:
    guide_id: str
    name: str = ""
    slogan: str = ""
    logo: str = ""
    genre: str = ""
    location: str = ""
    language: str = ""
    band: str = ""
    frequency: str = ""
    url: str = ""
    current_song: str = ""
    current_artist: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def https(url: str) -> str:
    return "https://" + url[len("http://") :] if url.startswith("http://") else url


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _http_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310 — fixed https host
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _body(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise TuneInError("bad response")
    head = data.get("head") or {}
    status = str(head.get("status", "200"))
    if status != "200":
        raise TuneInError(head.get("fault") or f"status {status}")
    body = data.get("body") or []
    return [b for b in body if isinstance(b, dict)]


def _flatten(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Search results may nest outlines under ``children``."""
    out: list[dict[str, Any]] = []
    for item in items:
        out.append(item)
        kids = item.get("children")
        if isinstance(kids, list):
            out.extend(_flatten([k for k in kids if isinstance(k, dict)]))
    return out


def parse_search(data: Any) -> list[Station]:
    stations: list[Station] = []
    for item in _flatten(_body(data)):
        if item.get("item") != "station" or item.get("type") != "audio":
            continue
        gid = str(item.get("guide_id") or item.get("preset_id") or "")
        if not gid.startswith("s"):
            continue
        stations.append(
            Station(
                guide_id=gid,
                name=str(item.get("text") or gid),
                subtext=str(item.get("subtext") or ""),
                logo=https(str(item.get("image") or "")),
                bitrate=_int(item.get("bitrate")),
                formats=str(item.get("formats") or ""),
                reliability=_int(item.get("reliability")),
                now_playing=str(item.get("current_track") or ""),
            )
        )
    return stations


def is_placeholder(url: str) -> bool:
    return "notcompatible" in url or "/service/Audio/" in url


def parse_tune(data: Any) -> list[str]:
    """Playable stream URLs, https first, placeholders dropped."""
    urls: list[str] = []
    for item in _body(data):
        url = str(item.get("url") or "")
        if not url or is_placeholder(url) or url in urls:
            continue
        urls.append(url)
    urls.sort(key=lambda u: 0 if u.startswith("https://") else 1)
    return urls


def parse_describe(data: Any, guide_id: str) -> StationInfo:
    for item in _body(data):
        if item.get("element") != "station":
            continue
        return StationInfo(
            guide_id=str(item.get("guide_id") or guide_id),
            name=str(item.get("name") or ""),
            slogan=str(item.get("slogan") or item.get("description") or ""),
            logo=https(str(item.get("logo") or "")),
            genre=str(item.get("genre_name") or ""),
            location=str(item.get("location") or ""),
            language=str(item.get("language") or ""),
            band=str(item.get("band") or ""),
            frequency=str(item.get("frequency") or ""),
            url=str(item.get("url") or ""),
            current_song=str(item.get("current_song") or ""),
            current_artist=str(item.get("current_artist") or ""),
        )
    return StationInfo(guide_id=guide_id)


class TuneIn:
    def __init__(self, fetch: Optional[Fetch] = None, *, cache_size: int = 32) -> None:
        self._fetch = fetch or _http_get_json
        self._cache: "OrderedDict[str, list[Station]]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def _get(self, path: str, **params: Any) -> Any:
        params["render"] = "json"
        url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
        try:
            return self._fetch(url)
        except TuneInError:
            raise
        except Exception as exc:  # noqa: BLE001 — network, JSON, timeout
            raise TuneInError(str(exc)) from exc

    def search(self, query: str) -> list[Station]:
        key = query.strip().lower()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return list(self._cache[key])
        stations = parse_search(self._get("Search.ashx", query=query, types="station"))
        with self._lock:
            self._cache[key] = stations
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return list(stations)

    def tune(self, guide_id: str) -> list[str]:
        return parse_tune(self._get("Tune.ashx", id=guide_id, formats="mp3,aac,ogg,hls"))

    def describe(self, guide_id: str) -> StationInfo:
        return parse_describe(self._get("Describe.ashx", id=guide_id), guide_id)
