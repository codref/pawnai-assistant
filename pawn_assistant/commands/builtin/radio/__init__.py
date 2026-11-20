"""Web radio from TuneIn, played through mpv, with a station card on the HUD.

Config (``commands.radio`` in the YAML):

    presets: {deejay: s114685}   # spoken name → TuneIn guide id
    default_station: deejay      # "accendi la radio" with no last station
    volume: 70                   # 0–100 (mpv scale)
    duck_volume: 12              # while PAWN listens / thinks / speaks
    duck_ramp_ms: 300
    prefer: [Italian, Italy]     # rank hits whose language/location match first

The radio is ducked from AWAKE until the loop is back to IDLE: there is no echo
cancellation, so at full volume it would swamp the utterance. Not on STIR —
the radio itself nudges the wake score, and ducking there would pump.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from pawn_assistant.commands.base import SLOT_STR, Command, CommandContext, intent
from pawn_assistant.commands.builtin.radio.player import MpvPlayer
from pawn_assistant.commands.builtin.radio.tunein import Station, TuneIn, TuneInError
from pawn_assistant.commands.numbers import parse_number
from pawn_assistant.commands.text import FastMatch, normalize

logger = logging.getLogger(__name__)

DUCK_STATES = frozenset({"AWAKE", "LISTEN", "HOLD", "BUSY", "THINK", "SPEAK"})
WIDGET = "radio:main"
MAX_CANDIDATES = 4
COUNTRY_PROBE = 6

_VERBS = (
    r"(?:metti(?:mi)?|accendi|ascolta(?:re)?|ascoltiamo|fammi sentire|fammi ascoltare"
    r"|sintonizza(?:ti)?(?:\s+su)?|vai su|play|put on|turn on|tune in to|tune to|listen to)"
)
_PLAY_RE = re.compile(rf"^{_VERBS}\s+(?:(?:un po di|la|il|lo|una|uno|un|the|some|a)\s+)?(?P<rest>.+)$")
_BARE_RE = re.compile(r"^(?:la\s+)?radio\s+(?P<rest>.+)$")
_GENERIC = {"radio", "web radio", "musica", "music", "un po di musica", "the radio"}
_STATION_HINT = re.compile(r"\b(radio|fm|stazione|station)\b")


# Words that say little about *which* station: matching them counts less.
_GENERIC_WORDS = {"radio", "fm", "am", "web", "the", "la", "il", "lo", "di", "station", "stazione"}
_ARTICLE_NUMBERS = {"un", "una", "uno", "a", "an"}
GENERIC_WEIGHT = 0.3
# Below this a search hit is not the station that was asked for (e.g. "bbc
# radio 2" → "BBC World Service West Africa", which TuneIn ranks first because
# its search hides the BBC national stations).
MIN_NAME_SCORE = 0.8


def _tokens(text: str) -> list[str]:
    out = []
    for tok in normalize(text).split():
        if not tok.isdigit() and tok not in _ARTICLE_NUMBERS:
            num = parse_number(tok)
            if num is not None:
                tok = str(num)  # "two" / "due" → "2"
        out.append(tok)
    return out


def name_score(query: str, name: str) -> float:
    """How well a station ``name`` covers the spoken ``query`` (0–1).

    Every number in the query must appear in the name ("bbc radio 2" never
    matches "BBC Radio 1"); generic words like "radio"/"fm" weigh less.
    """
    q = _tokens(query)
    n = set(_tokens(name))
    if not q:
        return 0.0
    if any(t.isdigit() and t not in n for t in q):
        return 0.0
    total = matched = 0.0
    for tok in q:
        weight = GENERIC_WEIGHT if tok in _GENERIC_WORDS else 1.0
        total += weight
        if tok in n:
            matched += weight
    return matched / total if total else 0.0


def _key(name: str) -> str:
    """Preset/station comparison key: normalised, no leading 'radio'."""
    norm = normalize(name)
    return re.sub(r"^radio\s+", "", norm).strip()


class RadioCommand(Command):
    name = "radio"
    title = "Web radio"
    description = "Live stations from TuneIn, played with mpv; ducks while PAWN listens."
    order = 30
    cancel_words = ("radio", "musica", "music")
    assets = {"js": ("widget.js",), "css": ("widget.css",)}

    def __init__(self) -> None:
        super().__init__()
        self.tunein = TuneIn()
        self.player = MpvPlayer(on_meta=self._on_meta, on_exit=self._on_exit)
        self.presets: dict[str, str] = {}
        self.volume = 70.0
        self.duck_volume = 12.0
        self.duck_ramp_ms = 300.0
        self.default_station = ""
        # TuneIn search is not localised ("virgin radio" → the UK one first);
        # hits whose location/language contain one of these are tried first.
        self.prefer: tuple[str, ...] = ()
        self._ducked = False
        self._station: Optional[Station] = None
        self._urls: list[str] = []
        self._now_playing = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup

    def configure(self, cfg: dict[str, Any]) -> None:
        super().configure(cfg)
        presets = cfg.get("presets") or {}
        self.presets = {_key(str(k)): str(v) for k, v in presets.items() if v}
        self.volume = float(cfg.get("volume", self.volume))
        self.duck_volume = float(cfg.get("duck_volume", self.duck_volume))
        self.duck_ramp_ms = float(cfg.get("duck_ramp_ms", self.duck_ramp_ms))
        self.default_station = str(cfg.get("default_station") or "")
        prefer = cfg.get("prefer") or ()
        if isinstance(prefer, str):
            prefer = [prefer]
        self.prefer = tuple(str(x).strip().lower() for x in prefer if str(x).strip())
        if cfg.get("mpv"):
            self.player.binary = str(cfg["mpv"])

    def setup(self, ctx: CommandContext) -> None:
        saved = self.state_get("volume")
        if isinstance(saved, (int, float)):
            self.volume = float(saved)

    # ------------------------------------------------------------------ matching

    def fast_match(self, norm: str) -> Optional[FastMatch]:
        m = _PLAY_RE.match(norm) or _BARE_RE.match(norm)
        if m is None:
            return None
        rest = m.group("rest").strip()
        if m.re is _BARE_RE:
            rest = f"radio {rest}"
        if rest in _GENERIC or rest in {"la radio", "la musica"}:
            return FastMatch("radio.play", {}, 1.0)
        if _STATION_HINT.search(rest) or _key(rest) in self.presets:
            return FastMatch("radio.play", {"station": rest}, 1.0)
        return None

    # ------------------------------------------------------------------ handlers

    @intent(
        "radio.play",
        "the user wants to listen to a radio station or music (station name if said)",
        slots={"station": SLOT_STR},
        examples=("accendi la radio", "metti la radio", "turn on the radio"),
        fewshot=(
            ("mi metti radio capital?", {"station": "radio capital"}),
            ("voglio ascoltare rtl 102.5", {"station": "rtl 102.5"}),
            ("play bbc radio 1", {"station": "bbc radio 1"}),
        ),
        phrase=False,
    )
    def play(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        if not self.player.available(self.player.binary):
            return ctx.t("Serve mpv per ascoltare la radio.", "I need mpv to play the radio.")
        query = str(slots.get("station") or "").strip()
        try:
            found = self._resolve(query)
        except TuneInError as exc:
            logger.warning("tunein: %s", exc)
            return ctx.t("TuneIn non risponde.", "TuneIn is not responding.")
        if found is None:
            if not query:
                return ctx.t("Quale stazione?", "Which station?")
            return ctx.t("Non trovo quella stazione.", "I can't find that station.")
        station, urls = found
        self._start(station, urls)
        return ctx.t(f"Metto {station.name}.", f"Playing {station.name}.")

    @intent(
        "radio.stop",
        "the user wants to turn off the radio or the music",
        examples=(
            "spegni la radio",
            "ferma la radio",
            "spegni la musica",
            "ferma la musica",
            "basta musica",
            "stop radio",
            "turn off the radio",
            "stop the music",
        ),
        phrase=False,
    )
    def stop(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        if not self.player.running:
            ctx.ui.close(WIDGET)
            return ctx.t("La radio è già spenta.", "The radio is already off.")
        self._stop()
        return ctx.t("Radio spenta.", "Radio off.")

    @intent(
        "radio.info",
        "the user asks what is playing on the radio or which station it is",
        examples=(
            "cosa sta suonando",
            "che canzone e",
            "che canzone e questa",
            "che radio e",
            "che radio e questa",
            "cosa stiamo ascoltando",
            "what s playing",
            "what song is this",
        ),
        phrase=False,
    )
    def info(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        station = self._station
        if station is None or not self.player.running:
            return ctx.t("La radio è spenta.", "The radio is off.")
        if self._now_playing:
            return ctx.t(
                f"Su {station.name}: {self._now_playing}.",
                f"On {station.name}: {self._now_playing}.",
            )
        return ctx.t(f"Stai ascoltando {station.name}.", f"You're listening to {station.name}.")

    # ------------------------------------------------------------------ resolve

    def _resolve(self, query: str) -> Optional[tuple[Station, list[str]]]:
        if not query:
            last = self.state_get("last")
            if isinstance(last, dict) and last.get("guide_id"):
                st = Station(**{k: v for k, v in last.items() if k in Station.__dataclass_fields__})
                urls = self.tunein.tune(st.guide_id)
                if urls:
                    return st, urls
            if not self.default_station:
                return None
            query = self.default_station

        gid = self.presets.get(_key(query))
        if gid is None and re.fullmatch(r"s\d+", query.strip()):
            gid = query.strip()
        if gid is not None:
            urls = self.tunein.tune(gid)
            if not urls:
                return None
            name = query.strip().title()
            try:
                info = self.tunein.describe(gid)
                if info.name:
                    name = info.name
                return Station(guide_id=gid, name=name, logo=info.logo, subtext=info.slogan), urls
            except TuneInError:
                return Station(guide_id=gid, name=name), urls

        hits = self._matching(query, self.tunein.search(query))
        if not hits:
            # "radio pippo" → "pippo"; "pippo" → "radio pippo".
            bare = _key(query)
            alt = bare if bare != normalize(query) else f"radio {bare}"
            hits = self._matching(query, self.tunein.search(alt))
        if not hits:
            logger.info("radio: no TuneIn hit named like %r — add a preset if it exists", query)
            return None
        hits = self._rank_preferred(hits)
        for st in hits[:MAX_CANDIDATES]:
            urls = self.tunein.tune(st.guide_id)
            if urls:
                return st, urls
            logger.info("radio: %s (%s) not playable here, trying next", st.name, st.guide_id)
        return None

    @staticmethod
    def _matching(query: str, hits: list[Station]) -> list[Station]:
        """Hits whose name matches the query well enough, best first (stable)."""
        scored = [(name_score(query, st.name), i, st) for i, st in enumerate(hits)]
        good = [x for x in scored if x[0] >= MIN_NAME_SCORE]
        good.sort(key=lambda x: (-x[0], x[1]))
        return [st for _, _, st in good]

    def _rank_preferred(self, hits: list[Station]) -> list[Station]:
        if not self.prefer or len(hits) < 2:
            return hits
        top = hits[:COUNTRY_PROBE]

        def _located(st: Station) -> bool:
            try:
                info = self.tunein.describe(st.guide_id)
            except TuneInError:
                return False
            where = f"{info.location} | {info.language}".lower()
            return any(p in where for p in self.prefer)

        with ThreadPoolExecutor(max_workers=len(top)) as pool:
            local = list(pool.map(_located, top))
        first = [st for st, ok in zip(top, local, strict=True) if ok]
        return first + [st for st in hits if st not in first]

    # ------------------------------------------------------------------ playback

    def _target_volume(self) -> float:
        return self.duck_volume if self._ducked else self.volume

    def _start(self, station: Station, urls: list[str]) -> None:
        with self._lock:
            self._station = station
            self._urls = list(urls[1:])
            self._now_playing = station.now_playing
        self.player.play(urls[0], volume=self._target_volume())
        self.mark_active()
        self.state_set("last", station.as_dict())
        if self.ctx is not None:
            self.ctx.ui.open(
                WIDGET,
                "radio",
                command=self.name,
                title="RADIO",
                anchor="tl",
                data={
                    "guide_id": station.guide_id,
                    "name": station.name,
                    "slogan": station.subtext,
                    "logo": station.logo,
                    "bitrate": station.bitrate,
                    "format": station.formats.upper(),
                    "now_playing": station.now_playing,
                    "volume": round(self.volume),
                    "ducked": self._ducked,
                    "status": "playing",
                    "error": "",
                },
            )
        threading.Thread(
            target=self._describe, args=(station.guide_id,), name="pawn-radio-describe", daemon=True
        ).start()

    def _describe(self, guide_id: str) -> None:
        try:
            info = self.tunein.describe(guide_id)
        except TuneInError as exc:
            logger.info("tunein describe failed: %s", exc)
            return
        if self.ctx is None or self._station is None or self._station.guide_id != guide_id:
            return
        data: dict[str, Any] = {
            "genre": info.genre,
            "location": info.location,
            "language": info.language,
            "band": info.band,
            "frequency": info.frequency,
            "url": info.url,
        }
        if info.slogan:
            data["slogan"] = info.slogan
        if info.logo:
            data["logo"] = info.logo
        if not self._now_playing and (info.current_song or info.current_artist):
            self._now_playing = " — ".join(x for x in (info.current_artist, info.current_song) if x)
            data["now_playing"] = self._now_playing
        self.ctx.ui.update(WIDGET, data)

    def _stop(self) -> None:
        self.player.stop()
        with self._lock:
            self._urls = []
        if self.ctx is not None:
            self.ctx.ui.close(WIDGET)

    def _on_meta(self, title: str) -> None:
        self._now_playing = title
        if self.ctx is not None:
            self.ctx.ui.update(WIDGET, {"now_playing": title, "status": "playing"})

    def _on_exit(self, reason: str) -> None:
        with self._lock:
            nxt = self._urls.pop(0) if self._urls else None
        if nxt is not None:
            logger.info("radio: stream failed (%s), trying next url", reason)
            self.player.play(nxt, volume=self._target_volume())
            return
        if self.ctx is not None:
            self.ctx.ui.update(WIDGET, {"status": "error", "error": reason})

    # ------------------------------------------------------------------ hooks

    def is_active(self) -> bool:
        return self.player.running

    def on_cancel(self, ctx: CommandContext) -> Optional[str]:
        if not self.player.running:
            return None
        self._stop()
        return ctx.t("Radio spenta.", "Radio off.")

    def on_state(self, state: str) -> None:
        ducked = state in DUCK_STATES
        if ducked == self._ducked:
            return
        self._ducked = ducked
        if not self.player.running:
            return
        # Duck fast, restore gently.
        self.player.ramp_to(self._target_volume(), ms=80 if ducked else self.duck_ramp_ms)
        if self.ctx is not None:
            self.ctx.ui.update(WIDGET, {"ducked": ducked})

    def on_ui_action(self, widget_id: str, action: str, data: dict[str, Any]) -> None:
        if action in ("stop", "close"):
            self._stop()
        elif action == "set_volume":
            try:
                value = max(0.0, min(100.0, float(data.get("value"))))
            except (TypeError, ValueError):
                return
            self.volume = value
            self.state_set("volume", value)
            if not self._ducked:
                self.player.set_volume(value)
            if self.ctx is not None:
                self.ctx.ui.update(WIDGET, {"volume": round(value)})

    def on_disable(self) -> None:
        self._stop()

    def shutdown(self) -> None:
        self.player.stop()
