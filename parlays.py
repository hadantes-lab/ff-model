"""
parlays.py
==========
Parlay recommendations: player-prop parlays (4-5 legs, plus-money), moneyline parlays, and Wong teasers.

THE HONEST FRAMING (read this first)
------------------------------------
A parlay multiplies the sportsbook's margin on every leg. Our testing (market_trust.py, 14,777 past props) found the
model carries no measurable information beyond the market's price in any prop market, and a straight-up winner's
probability here IS the market's. So the expected value of almost any parlay built from these legs is NEGATIVE, and
this module reports it as it is: every parlay shows its joint win probability, the probability its payout needs to
break even, and the edge (EV per $1) between them. "Highest edge" means the least-negative or, where one exists,
the positive. The one place real edge has been measured is the underdog side of Wong teasers (see below).

PLAYER-PROP PARLAYS
-------------------
Legs are props the model and the market agree on (the model's raw lean is the same side as the market's favored
side), ranked by

    leg score = (adjusted probability of the side - 0.5) x market quality + a small game-script bonus

where *market quality* is the shrunk hit rate of the model's high-confidence picks in that market this season (from
tracking/weekly_summary.csv), so legs come from the markets the predictions have been doing best in, and a market
that has been losing is down-weighted. Parlays of 4-5 legs from the top legs are searched, and a parlay is scored on
its JOINT probability:

* Correlation. Props in the same game move together, and the size of that was measured on 14,777 past props
  (parlay_correlations.json, built by `python parlays.py --estimate`): a QB's passing yards and his teammates'
  receiving yards correlate +0.31, his yards and an RB's rushing yards -0.07, opposing RBs' rushing yards -0.12.
  The joint probability uses a Gaussian copula with those correlations (sign-flipped for an under), and a parlay
  never pairs legs that are negatively correlated in the same game.
* Game script. Each leg is tagged with whether it fits the game's expected script (a favorite's RB over, a trailing
  QB's attempts over, ...); aligned legs get a small bonus. Cross-game legs are treated as independent.

Same-game legs: sportsbooks price same-game parlays with the correlation already built in, so the naive product of
prices used here overstates what a book will pay for those pairs. Cross-game legs have no such issue, which is why the
search allows at most two legs from one game and the report says which parlays have them.

MONEYLINE PARLAYS
-----------------
Favorites with the best available moneyline prices across books, 2-4 legs from different games. A leg's probability is
the market's vig-free consensus, so a parlay's edge here is only what line shopping earns (best price vs consensus).

WONG TEASERS
------------
A 6-point NFL teaser that moves a line across both 3 and 7: favorites of -7.5 or -8.5 down to -1.5 or -2.5, and
underdogs of +1.5 or +2.5 up to +7.5 or +8.5. Each bucket's cover rate comes from history (2014-2026, recomputed from
nflverse every run), and the report uses the last five seasons with shrinkage, because the edge has not held evenly:
underdog legs have covered ~76% in recent years, while -7.5 favorites fell to 57% in 2022-25. A 2-team teaser at -120
needs 73.9% per leg (72.4% at -110); EV is computed for both prices, which are assumptions (the Odds API doesn't
supply teaser prices). Legs from different games are independent, so a pair's chance is the product.
Play them as PAIRS (the best two legs) or as a ROUND ROBIN (every pair among 4-6 legs).
"""

import hashlib
import itertools
import json
import math
import pathlib
from statistics import NormalDist

import numpy as np
import pandas as pd

CORR_FILE = pathlib.Path(__file__).parent / "parlay_correlations.json"
ND = NormalDist()
MC_SAMPLES = 300_000

# ---- knobs ---------------------------------------------------------------------------------
PROP_POOL = 16                 # top legs searched
MIN_LEG_PROB = 0.51            # floor only: legs are RANKED by score (prop lines sit near coin flips, so 58%+ legs are rare)
MAX_LEG_PRICE = -300           # nothing heavier than this (a -500 leg adds almost no payout)
MIN_PARLAY_DECIMAL = 3.0       # "underdog odds": at least +200
MAX_PER_GAME = 2
MAX_SHARED_LEGS = 2            # two recommended parlays may share at most this many legs
NEG_CORR_BLOCK = -0.03         # same-game legs with effective correlation below this are never combined
SHRINK_PICKS = 20              # prior picks (at 50%) behind a market's quality estimate
TEASER_PRICES = {"2-team at -120": 100 / 120 + 1, "2-team at -110": 100 / 110 + 1}
TEASER_POINTS = 6.0
WONG_BUCKETS = {-7.5: "fav -7.5 to -1.5", -8.5: "fav -8.5 to -2.5", 1.5: "dog +1.5 to +7.5", 2.5: "dog +2.5 to +8.5"}
WONG_RECENT_FROM = 2021
WONG_SHRINK = 40


# ---- odds math -----------------------------------------------------------------------------
def dec(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / -american)


def to_american(decimal: float) -> int:
    return int(round((decimal - 1) * 100)) if decimal >= 2 else int(round(-100 / (decimal - 1)))


def fmt_american(a: float) -> str:
    return f"+{int(a)}" if a > 0 else f"{int(a)}"


def ncdf(x):
    """Standard normal CDF, vectorized (Abramowitz-Stegun 7.1.26, error ~1e-7); no scipy needed."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + 0.3275911 * z)
    erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * erf)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(160)


def bvn_cdf(h: float, k: float, rho: float) -> float:
    """P(X <= h, Y <= k) for standard bivariate normal with correlation rho (Gauss-Legendre over x)."""
    if abs(rho) < 1e-9:
        return float(ncdf(h) * ncdf(k))
    lo = -9.0
    hi = min(h, 9.0)
    if hi <= lo:
        return 0.0
    x = 0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * _GL_W
    s = math.sqrt(1 - rho * rho)
    dens = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return float(np.sum(w * dens * ncdf((k - rho * x) / s)))


def pair_lift(p_i: float, p_j: float, rho: float) -> float:
    """P(both hit) / (p_i * p_j) for two legs with latent normal correlation rho."""
    if abs(rho) < 1e-9:
        return 1.0
    both = bvn_cdf(ND.inv_cdf(p_i), ND.inv_cdf(p_j), rho)
    return both / (p_i * p_j)


def joint_probability(ps, R, n=MC_SAMPLES, seed=0) -> float:
    """P(every leg hits) under a Gaussian copula with correlation matrix R (Monte Carlo, fixed seed)."""
    ps = np.asarray(ps, float)
    R = np.asarray(R, float)
    k = len(ps)
    if k == 1:
        return float(ps[0])
    w, v = np.linalg.eigh(R)                                   # nearest valid correlation matrix
    w = np.clip(w, 1e-6, None)
    R = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    L = np.linalg.cholesky(R)
    z = np.random.default_rng(seed).standard_normal((n, k)) @ L.T
    thresholds = np.array([ND.inv_cdf(min(max(p, 1e-6), 1 - 1e-6)) for p in ps])
    return float(np.mean(np.all(z < thresholds, axis=1)))


# ---- correlations -------------------------------------------------------------------------
ROLE = {"WR": "WRTE", "TE": "WRTE", "RB": "RB", "QB": "QB", "FB": "RB"}


def _key(market, pos):
    return (market.replace("player_", ""), ROLE.get(pos, pos))


def estimate_correlations(pairs: pd.DataFrame, min_n: int = 150) -> dict:
    """
    From within-game pairs of standardized misses (columns rel, ka, kb, za, zb; see correlation_pairs) -> the
    parlay_correlations.json structure. Each correlation is shrunk by t^2/(1+t^2) so a noisy estimate counts for less.
    """
    out = {}
    d = pairs.dropna(subset=["za", "zb"]).copy()
    d = d[np.isfinite(d["za"]) & np.isfinite(d["zb"])]
    d["ka"] = d["ka"].map(lambda k: _key(*k))
    d["kb"] = d["kb"].map(lambda k: _key(*k))
    d["pair"] = [tuple(sorted([a, b])) for a, b in zip(d["ka"], d["kb"])]
    for (rel, pair), g in d.groupby(["rel", "pair"]):
        if len(g) < min_n:
            continue
        first = np.array([a == pair[0] for a in g["ka"]])
        za = np.where(first, g["za"], g["zb"])
        zb = np.where(first, g["zb"], g["za"])
        r = float(np.corrcoef(za, zb)[0, 1])
        if not math.isfinite(r):
            continue
        n = len(g)
        t = r * math.sqrt(max(n - 3, 1))
        shrunk = r * (t * t) / (1 + t * t)
        out[f"{rel}|{pair[0][0]}/{pair[0][1]}|{pair[1][0]}/{pair[1][1]}"] = {"rho": round(shrunk, 4), "raw": round(r, 4), "n": int(n)}
    return out


def correlation_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Within-game pairs of standardized misses from the backtest frame joined to team/opponent/position."""
    rows = []
    for _, d in frame.groupby(["season", "week", "game"]):
        d = d.reset_index(drop=True)
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                a, b = d.loc[i], d.loc[j]
                rel = "same_player" if a["pid"] == b["pid"] else ("teammates" if a["team"] == b["team"] else "opponents")
                rows.append((rel, (a["market"], a["position"]), (b["market"], b["position"]), a["z"], b["z"]))
    return pd.DataFrame(rows, columns=["rel", "ka", "kb", "za", "zb"])


class CorrModel:
    """Looks up the latent correlation between two prop legs, adjusted for which side (over/under) each is on."""

    def __init__(self, pairs: dict = None):
        self.pairs = pairs or {}

    @classmethod
    def load(cls, path=None):
        path = pathlib.Path(path or CORR_FILE)
        if not path.exists():
            return cls({})
        return cls(json.loads(path.read_text(encoding="utf-8")).get("pairs", {}))

    def rho(self, a: dict, b: dict) -> float:
        if a.get("game") != b.get("game"):
            return 0.0
        if a["player_id"] == b["player_id"]:
            rel = "same_player"
        elif a["team"] == b["team"]:
            rel = "teammates"
        else:
            rel = "opponents"
        ka, kb = sorted([_key(a["market"], a["pos"]), _key(b["market"], b["pos"])])
        hit = self.pairs.get(f"{rel}|{ka[0]}/{ka[1]}|{kb[0]}/{kb[1]}")
        if hit is None:
            return 0.0
        sa = 1 if a["side"] == "over" else -1
        sb = 1 if b["side"] == "over" else -1
        return float(hit["rho"]) * sa * sb


# ---- legs ---------------------------------------------------------------------------------------
def market_quality(summary: pd.DataFrame, season: int = None) -> dict:
    """
    {market: shrunk hit rate of the model's high-confidence picks this season}, from weekly_summary.csv rows named
    '<market>:high'. A market with no record yet sits at 0.5.
    """
    out = {}
    if summary is None or summary.empty:
        return out
    h = summary if season is None else summary[summary["season"] == season]
    for g, d in h[h["group"].str.endswith(":high")].groupby("group"):
        n, w = float(d["n"].sum()), float(d["wins"].sum())
        out[g[:-5]] = (w + 0.5 * SHRINK_PICKS) / (n + SHRINK_PICKS)
    return out


def quality_factor(rate: float) -> float:
    return float(np.clip(rate / 0.5, 0.5, 1.3))


RUN_MARKETS = {"player_rush_yds", "player_rush_attempts", "player_rush_longest"}
PASS_MARKETS = {"player_pass_yds", "player_pass_attempts", "player_pass_completions", "player_pass_tds", "player_reception_yds",
                "player_receptions", "player_reception_longest", "player_rush_reception_yds"}


def script_alignment(market: str, side: str, team_margin: float, total: float = None) -> tuple:
    """
    (+1 / 0 / -1, short text): does this leg fit the game's expected script? team_margin > 0 = the leg's team is favored.
    Rushing volume follows leading teams, passing volume follows trailing teams and high totals; the effects were measured
    (volume_model.py): strong for rushing, weaker for passing, and the market prices them, so this is a tie-breaker.
    """
    s, why = 0, ""
    if market in RUN_MARKETS:
        if team_margin >= 3:
            s, why = (1 if side == "over" else -1), f"favored by {team_margin:g}: leans on the run"
        elif team_margin <= -3:
            s, why = (-1 if side == "over" else 1), f"underdog by {-team_margin:g}: runs less"
    elif market in PASS_MARKETS:
        if team_margin <= -3:
            s, why = (1 if side == "over" else -1), f"underdog by {-team_margin:g}: throws more"
        elif team_margin >= 7:
            s, why = (-1 if side == "over" else 1), f"favored by {team_margin:g}: throws less late"
        if total is not None and total >= 48 and side == "over" and s >= 0:
            s, why = 1, (why + "; " if why else "") + f"high total {total:g}"
        elif total is not None and total <= 41 and side == "under" and s >= 0:
            s, why = 1, (why + "; " if why else "") + f"low total {total:g}"
    return s, why


def prop_legs(props: list, games: list, quality: dict = None) -> list:
    """
    One leg per qualifying prop: the side with the higher adjusted probability that the model's raw lean also favors,
    at the best available price. Skips thin samples, injured players, anytime TDs, short or very heavy prices.
    """
    quality = quality or {}
    gm = {g["game"]: g for g in games}
    legs = []
    for p in props:
        if p.get("kind") == "td" or p.get("thin") or p.get("injury") or "player_id" not in p:
            continue
        sides = []
        for side in ("over", "under"):
            best = p.get("best_" + side)
            if best and p.get(side) is not None:
                sides.append((float(p[side]), side, best))
        if not sides:
            continue
        prob, side, best = max(sides)
        raw_side = "over" if p.get("model_over", 0.5) > 0.5 else "under"
        if raw_side != side or prob < MIN_LEG_PROB or best["price"] < MAX_LEG_PRICE:
            continue
        g = gm.get(p["game"], {})
        team_margin = None
        if g.get("home_spread") is not None:
            team_margin = -g["home_spread"] if p["team"] == g.get("home") else g["home_spread"]
        align, why = script_alignment(p["market"], side, team_margin, g.get("total")) if team_margin is not None else (0, "")
        q = quality.get(p["market"], 0.5)
        qf = quality_factor(q)
        edge = prob * dec(best["price"]) - 1
        legs.append({"id": f"{p['player_id']}|{p['market']}|{side}", "kind": "prop", "player_id": p["player_id"], "player": p["player"],
                     "team": p["team"], "opp": p["opp"], "game": p["game"], "pos": p["pos"], "market": p["market"], "label": p["label"],
                     "side": side, "line": p["line"], "price": best["price"], "book": best.get("book"), "prob": prob,
                     "market_prob": (p.get("market_over") if side == "over" else (None if p.get("market_over") is None else 1 - p["market_over"])),
                     "edge": edge, "quality": q, "script": align, "script_note": why,
                     "score": (prob - 0.5) * qf + 0.01 * align + 0.25 * max(edge, 0)})
    legs.sort(key=lambda l: -l["score"])
    return legs


# ---- parlays -------------------------------------------------------------------------------------
def _parlay_id(legs) -> str:
    return hashlib.sha1("|".join(sorted(l["id"] for l in legs)).encode()).hexdigest()[:10]


def _summary(legs, joint, naive, kind) -> dict:
    d = float(np.prod([dec(l["price"]) for l in legs]))
    same_game = [g for g in {l["game"] for l in legs} if sum(1 for l in legs if l["game"] == g) > 1]
    return {"id": _parlay_id(legs), "kind": kind, "n_legs": len(legs), "price": to_american(d), "decimal": d,
            "joint_prob": joint, "naive_prob": naive, "correlation_lift": joint / naive if naive else 1.0,
            "breakeven_prob": 1 / d, "ev": joint * d - 1, "ev_if_independent": naive * d - 1,
            "avg_leg_prob": float(np.mean([l["prob"] for l in legs])), "script_aligned": sum(1 for l in legs if l.get("script", 0) > 0),
            "same_game": same_game, "legs": legs}


def _combo_ok(legs, corr: CorrModel):
    games = {}
    for l in legs:
        games[l["game"]] = games.get(l["game"], 0) + 1
    if max(games.values()) > MAX_PER_GAME:
        return False
    if len({l["player_id"] for l in legs}) < len(legs):
        return False
    for a, b in itertools.combinations(legs, 2):
        if a["game"] == b["game"] and corr.rho(a, b) < NEG_CORR_BLOCK:
            return False
    return True


def _approx_joint(legs, corr: CorrModel) -> float:
    j = float(np.prod([l["prob"] for l in legs]))
    for a, b in itertools.combinations(legs, 2):
        r = corr.rho(a, b)
        if r:
            j *= pair_lift(a["prob"], b["prob"], r)
    return min(j, min(l["prob"] for l in legs))


def build_prop_parlays(legs: list, corr: CorrModel, n_recommend: int = 4, sizes=(4, 5), pool: int = PROP_POOL) -> dict:
    """Search 4-5 leg parlays among the top legs, rank by edge (EV per $1), keep 3-4 that don't overlap much."""
    top = []
    per_game = {}
    for l in legs:                                               # keep the pool varied: at most 3 legs from any one game
        if per_game.get(l["game"], 0) >= 3:
            continue
        per_game[l["game"]] = per_game.get(l["game"], 0) + 1
        top.append(l)
        if len(top) == pool:
            break
    cands = []
    for k in sizes:
        for combo in itertools.combinations(top, k):
            if not _combo_ok(combo, corr):
                continue
            d = float(np.prod([dec(l["price"]) for l in combo]))
            if d < MIN_PARLAY_DECIMAL:
                continue
            j = _approx_joint(combo, corr)
            cands.append((j * d - 1, j, combo))
    cands.sort(key=lambda c: (-c[0], -c[1]))
    chosen = []
    for ev, j, combo in cands[:60]:
        ids = {l["id"] for l in combo}
        if any(len(ids & {l["id"] for l in c["legs"]}) > MAX_SHARED_LEGS for c in chosen):
            continue
        n = len(combo)
        R = np.eye(n)
        for a in range(n):
            for b in range(a + 1, n):
                R[a, b] = R[b, a] = corr.rho(combo[a], combo[b])
        ps = [l["prob"] for l in combo]
        chosen.append(_summary(list(combo), joint_probability(ps, R), float(np.prod(ps)), "props"))
        if len(chosen) == n_recommend:
            break
    chosen.sort(key=lambda c: -c["ev"])
    return {"parlays": chosen, "candidate_legs": len(legs), "searched": len(cands)}


# ---- moneyline parlays ---------------------------------------------------------------------------
def moneyline_legs(games: list) -> list:
    """Favorites with a live moneyline: probability = the market's vig-free consensus, price = the best across books."""
    legs = []
    for g in games:
        if g.get("fair_home_win") is None or not g.get("best_home_ml") or not g.get("best_away_ml"):
            continue
        home_fav = g["fair_home_win"] >= 0.5
        side = "home" if home_fav else "away"
        p = g["fair_home_win"] if home_fav else 1 - g["fair_home_win"]
        best = g["best_home_ml"] if home_fav else g["best_away_ml"]
        team, opp = (g["home"], g["away"]) if home_fav else (g["away"], g["home"])
        legs.append({"id": f"{g['game']}|ml|{team}", "kind": "ml", "game": g["game"], "team": team, "opp": opp, "side": side,
                     "price": best["price"], "book": best.get("book"), "prob": p, "edge": p * dec(best["price"]) - 1,
                     "label": "Moneyline", "player": team, "line": None, "script": 0, "script_note": "", "week": g.get("week")})
    return legs


def build_ml_parlays(legs: list, n_recommend: int = 3, min_prob: float = 0.55) -> dict:
    """
    Moneyline parlays from different games, ranked by edge (the gain from the best prices); legs are independent. One of each
    size (3, 4, then 2 legs), best edge first, sharing at most one leg, so the set is varied.
    """
    pool = sorted([l for l in legs if l["prob"] >= min_prob], key=lambda l: -l["edge"])[:12]
    by_size = {}
    for k in (2, 3, 4):
        c = []
        for combo in itertools.combinations(pool, k):
            d = float(np.prod([dec(l["price"]) for l in combo]))
            if d < 2.0:
                continue
            j = float(np.prod([l["prob"] for l in combo]))
            c.append((j * d - 1, j, combo))
        c.sort(key=lambda x: (-x[0], -x[1]))
        by_size[k] = c
    chosen = []
    for k in (3, 4, 2, 3):
        for ev, j, combo in by_size[k]:
            ids = {l["id"] for l in combo}
            if any(c["id"] == _parlay_id(combo) for c in chosen) or any(len(ids & {l["id"] for l in c["legs"]}) > 1 for c in chosen):
                continue
            chosen.append(_summary(list(combo), j, j, "ml"))
            break
        if len(chosen) == n_recommend:
            break
    chosen.sort(key=lambda c: -c["ev"])
    return {"parlays": chosen, "candidate_legs": len(legs)}


# ---- Wong teasers ----------------------------------------------------------------------------------
def wong_history(sched: pd.DataFrame, recent_from: int = WONG_RECENT_FROM, first: int = 2014) -> dict:
    """
    {bucket spread: {name, n_all, rate_all, n_recent, rate_recent, rate}} -- how often a team at that spread covered the
    6-point-teased line. `rate` is the recent rate shrunk toward the pooled recent Wong rate (WONG_SHRINK games), which is
    the number used for EV. Regular season only; margins are the team's final margin.
    """
    d = sched[(sched["game_type"] == "REG") & (sched["season"] >= first) & sched["home_score"].notna() & sched["spread_line"].notna()]
    recs = []
    for r in d.itertuples(index=False):
        m = r.home_score - r.away_score
        recs.append((r.season, -r.spread_line, m))
        recs.append((r.season, r.spread_line, -m))
    t = pd.DataFrame(recs, columns=["season", "spread", "margin"])
    t["cover"] = t["margin"] + t["spread"] + TEASER_POINTS > 0
    w = t[t["spread"].isin(WONG_BUCKETS)]
    recent = w[w["season"] >= recent_from]
    pooled = float(recent["cover"].mean()) if len(recent) else float(w["cover"].mean()) if len(w) else 0.74
    out = {}
    for sp, name in WONG_BUCKETS.items():
        a, r = w[w["spread"] == sp], recent[recent["spread"] == sp]
        k, n = float(r["cover"].sum()), len(r)
        out[sp] = {"name": name, "n_all": int(len(a)), "rate_all": float(a["cover"].mean()) if len(a) else None, "n_recent": int(n),
                   "rate_recent": float(r["cover"].mean()) if n else None,
                   "rate": (k + WONG_SHRINK * pooled) / (n + WONG_SHRINK)}
    out["_pooled_recent"] = pooled
    return out


def wong_legs(games: list, hist: dict) -> list:
    """Every upcoming game's side whose consensus spread sits exactly on a Wong bucket, with its teased line and cover rate."""
    legs = []
    for g in games:
        hs = g.get("home_spread")
        if hs is None:
            continue
        for team, opp, spread, side in ((g["home"], g["away"], hs, "home"), (g["away"], g["home"], -hs, "away")):
            sp = float(spread)
            if sp not in WONG_BUCKETS or sp not in hist:
                continue
            h = hist[sp]
            legs.append({"id": f"{g['game']}|wong|{team}", "game": g["game"], "team": team, "opp": opp, "side": side, "spread": sp,
                         "teased": sp + TEASER_POINTS, "bucket": h["name"], "prob": h["rate"], "n_recent": h["n_recent"],
                         "rate_recent": h["rate_recent"], "rate_all": h["rate_all"], "n_all": h["n_all"], "kickoff": g.get("commence"),
                         "kind": "wong", "week": g.get("week")})
    legs.sort(key=lambda l: -l["prob"])
    be = math.sqrt(1 / TEASER_PRICES["2-team at -120"])
    for l in legs:
        l["playable"] = l["prob"] >= be                             # covers more often than a 2-team teaser at -120 needs per leg
    return legs


def teaser_pair(a: dict, b: dict) -> dict:
    p = a["prob"] * b["prob"]
    return {"legs": [a, b], "prob": p, "ev": {k: p * d - 1 for k, d in TEASER_PRICES.items()}}


def build_teasers(legs: list, max_round_robin: int = 5, n_pairs: int = 3) -> dict:
    """Best pairs, and a round robin (every pair among the top legs), with the EV of each at the assumed teaser prices."""
    out = {"legs": legs, "pairs": [], "round_robin": None, "assumptions": {k: fmt_american(to_american(d)) for k, d in TEASER_PRICES.items()},
           "breakeven_leg": {k: math.sqrt(1 / d) for k, d in TEASER_PRICES.items()}}
    distinct = []
    seen = set()
    for l in legs:
        if l["game"] in seen:
            continue                                                # one side per game
        seen.add(l["game"])
        distinct.append(l)
    pairs = [teaser_pair(a, b) for a, b in itertools.combinations(distinct, 2)]
    pairs.sort(key=lambda p: -p["ev"]["2-team at -120"])
    out["pairs"] = pairs[:n_pairs]
    out["playable_legs"] = sum(1 for l in distinct if l.get("playable"))
    rr = [l for l in distinct if l.get("playable")][:max_round_robin]       # only legs that clear the break-even
    if len(rr) >= 3:
        rp = [teaser_pair(a, b) for a, b in itertools.combinations(rr, 2)]
        out["round_robin"] = {"legs": rr, "pairs": rp, "n_pairs": len(rp),
                              "total_ev": {k: sum(p["ev"][k] for p in rp) for k in TEASER_PRICES},
                              "avg_ev": {k: float(np.mean([p["ev"][k] for p in rp])) for k in TEASER_PRICES}}
    return out


# ---- everything ------------------------------------------------------------------------------------
def build_all(props: list, games: list, summary: pd.DataFrame = None, corr: CorrModel = None, sched: pd.DataFrame = None,
              season: int = None, week: int = None) -> dict:
    """The whole parlay block for the snapshot. Each part is independent and degrades to an empty result with a note."""
    corr = corr or CorrModel.load()
    quality = market_quality(summary, season)
    upcoming = [g for g in games if g.get("in_window", True)]
    out = {"quality": {m: round(v, 3) for m, v in quality.items()}, "notes": []}
    legs = prop_legs(props, upcoming, quality)
    for l in legs:
        l["week"] = week
    out["props"] = build_prop_parlays(legs, corr) if legs else {"parlays": [], "candidate_legs": 0, "searched": 0}
    if not out["props"]["parlays"]:
        n_games = len({l["game"] for l in legs})
        out["notes"].append(
            f"Only {len(legs)} legs from {n_games} game(s) qualified, and no combination of 4-5 of them passed the rules (one leg per "
            f"player, at most {MAX_PER_GAME} per game, no negatively correlated same-game pairs, a payout of at least +200). "
            f"On a light slate (Monday or Thursday night) that is expected.")

    out["ml"] = build_ml_parlays(moneyline_legs(games))
    hist = wong_history(sched) if sched is not None and len(sched) else {}
    out["teasers"] = build_teasers(wong_legs(games, hist), ) if hist else {"legs": [], "pairs": [], "round_robin": None, "assumptions": {}}
    out["wong_history"] = {str(k): v for k, v in hist.items() if k != "_pooled_recent"}
    out["wong_pooled_recent"] = hist.get("_pooled_recent")
    return out
