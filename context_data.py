"""
context_data.py
===============
The context panels on the props page: game-day weather, each team's target share (with red-zone
splits), and how the opposing defense has treated a player's position. All of it is shown as
information next to a prop, not fed into the simulator's projection -- game_factors.py and
team_ratings.py found no measurable effect of wind, cold or roof on results beyond what the market
already prices, so the page says so rather than implying otherwise.

WEATHER
-------
Forecast from Open-Meteo (free, no key), for the hours of the game at the stadium. The venue comes
from the schedule's `stadium` name FIRST, because "home" is not always where it's played: this
season's Jaguars "home" game is in London. Domes and closed roofs are reported as indoors (no
forecast fetched). A forecast can only look ~16 days ahead; if the call fails the game simply has
no weather and the page omits the card.

TARGET SHARE
------------
From play-by-play: for each team and week, every target (a pass thrown to a receiver) split by
player, plus the red-zone (inside the opponent's 20) and inside-the-10 subsets. Stored per team-week
so the page can recompute the L3 / L6 / L10 / season windows itself, over the *team's* games (a bye
week doesn't count as a game). "Inside the 10" is the closest honest stand-in for end-zone targets
that nflverse supports; first-read targets need charting data that isn't free.

DEFENSE VS POSITION
-------------------
How many yards (or catches, or TDs) a defense has allowed per game to ALL players at a position,
ranked 1-32 with 1 = allows the fewest (toughest). Uses this season's games once a defense has
`MIN_DVP_GAMES` of them, otherwise this and last season together.
"""

import datetime
import json
import pathlib
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

MIN_DVP_GAMES = 3
MIN_TARGETS = 3                       # a player needs this many targets all season to be listed
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Home stadium (lat, lon) by team abbreviation, as used elsewhere in this repo (LA = Rams, WAS, JAX, LV).
TEAM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4010), "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167), "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LV": (36.0909, -115.1833), "LAC": (33.9535, -118.3387), "LA": (33.9535, -118.3387), "MIA": (25.9580, -80.2389),
    "MIN": (44.9740, -93.2575), "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158), "SF": (37.4030, -121.9700),
    "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033), "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}
# International / neutral venues, matched by a fragment of the schedule's stadium name.
VENUE_COORDS = {
    "tottenham": (51.6043, -0.0664), "wembley": (51.5560, -0.2796), "allianz": (48.2188, 11.6247),
    "deutsche bank": (50.0686, 8.6455), "azteca": (19.3029, -99.1505), "banorte": (19.3029, -99.1505),
    "bernab": (40.4531, -3.6883), "maracan": (-22.9122, -43.2302), "croke": (53.3606, -6.2512),
    "melbourne": (-37.8200, 144.9834), "mcg": (-37.8200, 144.9834),
}
INDOOR_ROOFS = ("dome", "closed")
# Retractable-roof homes. The schedule only fills in `roof` at game time, and these roofs are closed for
# most games (Arizona, Atlanta, Houston, Indianapolis and Dallas were closed for nearly all of 2025), so
# an unknown roof here must not be read as "outdoors": the card says so instead of raising wind/rain flags.
RETRACTABLE_HOMES = ("ARI", "ATL", "HOU", "IND", "DAL")

WEATHER_TEXT = {0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Fog",
                51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
                61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
                71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Rain showers",
                81: "Rain showers", 82: "Heavy showers", 85: "Snow showers", 86: "Snow showers",
                95: "Thunderstorms", 96: "Thunderstorms, hail", 99: "Thunderstorms, hail"}


# ---- weather -----------------------------------------------------------------
def resolve_site(home_team: str, stadium: str = None):
    """(lat, lon) for a game: the named stadium first (international games), else the home team's."""
    name = (stadium or "").lower()
    for frag, xy in VENUE_COORDS.items():
        if frag in name:
            return xy
    return TEAM_COORDS.get(home_team)


def is_indoors(roof) -> bool:
    return str(roof).lower() in INDOOR_ROOFS


def parse_forecast(js: dict, kickoff_utc: str, hours: int = 3):
    """
    The conditions over the `hours` from kickoff, from an Open-Meteo hourly response:
    mean temperature, worst wind/gust and rain chance, and the typical sky. None if kickoff is
    outside the forecast window.
    """
    h = js.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        return None
    kick = pd.to_datetime(kickoff_utc, utc=True).tz_convert("UTC").tz_localize(None).floor("h")
    idx = [i for i, t in enumerate(times) if kick <= pd.Timestamp(t) < kick + pd.Timedelta(hours=hours)]
    if not idx:
        return None
    pick = lambda key: [h[key][i] for i in idx if h.get(key) and h[key][i] is not None]
    temps, wind, gust, rain, code = pick("temperature_2m"), pick("wind_speed_10m"), pick("wind_gusts_10m"), \
        pick("precipitation_probability"), pick("weather_code")
    if not temps:
        return None
    worst = max(code) if code else None
    return {"temp_f": round(float(np.mean(temps))), "wind_mph": round(float(max(wind))) if wind else None,
            "gust_mph": round(float(max(gust))) if gust else None,
            "precip_pct": int(max(rain)) if rain else None,
            "sky": WEATHER_TEXT.get(int(worst), "") if worst is not None else ""}


def fetch_forecast(lat: float, lon: float, timeout: float = 12):
    """One forecast call (hourly, 16 days, US units, UTC times). None on any failure."""
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "UTC", "forecast_days": 16,
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m,weather_code"})
    try:
        with urllib.request.urlopen(f"{FORECAST_URL}?{q}", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def weather_for_game(home_team, commence, roof, stadium, fetch=fetch_forecast) -> dict | None:
    """The weather card for one game: {indoors, stadium, ...conditions} or None if it can't be had."""
    if is_indoors(roof):
        return {"indoors": True, "roof": str(roof), "stadium": stadium}
    xy = resolve_site(home_team, stadium)
    if xy is None:
        return None
    js = fetch(*xy)
    wx = parse_forecast(js, commence) if js else None
    if wx is None:
        return None
    unknown_roof = str(roof).lower() not in ("outdoors", "open")
    retractable = bool(home_team in RETRACTABLE_HOMES and unknown_roof and xy == TEAM_COORDS.get(home_team))
    return dict(wx, indoors=False, retractable=retractable, roof=str(roof), stadium=stadium)


def schedule_row(sched: pd.DataFrame, home: str, away: str, commence):
    """The schedule row for this pairing nearest `commence` (rematches exist), or None."""
    if sched is None or sched.empty:
        return None
    m = sched[(sched["home_team"] == home) & (sched["away_team"] == away)]
    if m.empty:
        return None
    when = pd.to_datetime(commence, utc=True, errors="coerce")
    if pd.isna(when):
        return m.iloc[0]
    days = (pd.to_datetime(m["gameday"], utc=True) - when).abs()
    return m.loc[days.idxmin()]


# ---- target share ------------------------------------------------------------------
def load_pbp_targets(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    cols = ["season", "week", "season_type", "posteam", "play_type", "receiver_player_id", "yardline_100"]
    p = nfl.load_pbp([season]).to_pandas()
    p = p[[c for c in cols if c in p.columns]]
    return p[p["season_type"] == "REG"]


def target_tables(pbp: pd.DataFrame, history: pd.DataFrame, season: int, teams) -> dict:
    """
    {team: {"weeks": [[week, team targets, team RZ targets, team inside-10 targets], ...],
            "players": [{"id", "name", "pos", "g": [[week, targets, rz, in10], ...]}]}}
    for this season. A player's `g` lists every game he appeared in (zeros included), so games
    played is right even when he saw no targets. RZ = inside the opponent's 20, in10 = inside the 10.
    """
    t = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna() & pbp["posteam"].notna()].copy()
    t["rz"] = (t["yardline_100"] <= 20).astype(int)
    t["in10"] = (t["yardline_100"] <= 10).astype(int)
    t["tgt"] = 1
    per = (t.groupby(["posteam", "week", "receiver_player_id"])[["tgt", "rz", "in10"]].sum().reset_index())
    hs = history[history["season"] == season]
    out = {}
    for team in teams:
        tp = per[per["posteam"] == team]
        weeks = (tp.groupby("week")[["tgt", "rz", "in10"]].sum().reset_index().sort_values("week"))
        roster = hs[hs["team"] == team]
        if weeks.empty or roster.empty:
            continue
        players = []
        for pid, rows in roster.groupby("player_id"):
            mine = tp[tp["receiver_player_id"] == pid].set_index("week")
            games = []
            for wk in sorted(rows["week"].unique()):
                r = mine.loc[wk] if wk in mine.index else None
                games.append([int(wk)] + ([int(r["tgt"]), int(r["rz"]), int(r["in10"])] if r is not None else [0, 0, 0]))
            if sum(g[1] for g in games) < MIN_TARGETS:
                continue
            players.append({"id": pid, "name": rows["player_display_name"].iloc[-1], "pos": rows["position"].iloc[-1], "g": games})
        players.sort(key=lambda p: -sum(g[1] for g in p["g"]))
        out[team] = {"weeks": [[int(r.week), int(r.tgt), int(r.rz), int(r.in10)] for r in weeks.itertuples()],
                     "players": players}
    return out


# ---- defense vs position ---------------------------------------------------------
def defense_vs_position(history: pd.DataFrame, season: int, markets: dict, min_games=MIN_DVP_GAMES) -> dict:
    """
    {defense: {position: {market: {"pg", "rank", "lg", "n"}}}}: stat allowed per game to ALL players
    at `position`, rank 1 = allows the fewest. `markets` maps market key -> a Spec with `.cols`.
    """
    from props_model import stat_series

    h = history[history["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    out = {}
    for market, spec in markets.items():
        h["_v"] = stat_series(h, spec)
        g = h.groupby(["opponent_team", "position", "season", "week"])["_v"].sum().reset_index()
        for pos, gp in g.groupby("position"):
            rows = {}
            for opp, d in gp.groupby("opponent_team"):
                cur = d[d["season"] == season]
                use = cur if len(cur) >= min_games else d[d["season"] >= season - 1]   # one row per game (season, week)
                if len(use) == 0:
                    continue
                rows[opp] = (float(use["_v"].mean()), int(len(use)))
            if not rows:
                continue
            lg = float(np.mean([v for v, _ in rows.values()]))
            order = sorted(rows, key=lambda o: rows[o][0])
            for rank, opp in enumerate(order, start=1):
                out.setdefault(opp, {}).setdefault(pos, {})[market] = {
                    "pg": round(rows[opp][0], 1), "rank": rank, "lg": round(lg, 1), "n": rows[opp][1]}
    return out


# ---- weather backlog ---------------------------------------------------------------
WEATHER_HISTORY_FILE = pathlib.Path(__file__).parent / "tracking" / "weather_history.csv"
WEATHER_COLUMNS = ["pulled_at", "season", "week", "game", "commence", "stadium", "roof", "indoors", "temp_f",
                   "wind_mph", "gust_mph", "precip_pct", "sky"]


def weather_rows(weather_by_game: dict, meta_by_game: dict, season: int, pulled_at: str) -> list:
    """One backlog row per game from {game: weather card} and {game: {week, commence}}."""
    rows = []
    for game, w in weather_by_game.items():
        m = meta_by_game.get(game, {})
        rows.append({"pulled_at": pulled_at, "season": season, "week": m.get("week"), "game": game,
                     "commence": m.get("commence"), "stadium": w.get("stadium"), "roof": w.get("roof"),
                     "indoors": bool(w.get("indoors")), "temp_f": w.get("temp_f"), "wind_mph": w.get("wind_mph"),
                     "gust_mph": w.get("gust_mph"), "precip_pct": w.get("precip_pct"), "sky": w.get("sky")})
    return rows


def append_weather_history(rows: list, path=None) -> int:
    """Append-only, like line_history: the forecast for a game changes day to day and that path is the data."""
    import csv

    path = pathlib.Path(path or WEATHER_HISTORY_FILE)
    if not rows:
        return 0
    path.parent.mkdir(exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=WEATHER_COLUMNS, extrasaction="ignore")
        if fresh:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def collect_weather(games, sched: pd.DataFrame, fetch=fetch_forecast):
    """
    Weather for a list of (game_label, home, away, commence) -> ({game: card}, {game: {week, commence}}).
    Games the schedule doesn't list, or whose forecast can't be had, are left out.
    """
    cards, meta = {}, {}
    for label, home, away, commence in games:
        row = schedule_row(sched, home, away, commence)
        if row is None:
            continue
        card = weather_for_game(home, commence, row["roof"], row["stadium"], fetch=fetch)
        if card is not None:
            cards[label] = card
            meta[label] = {"week": int(row["week"]), "commence": commence}
    return cards, meta
