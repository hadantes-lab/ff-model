"""
auction_engine.py
=================
Live auction draft engine. Pulls real player data, computes auction dollar
values from projected production, classifies every player into three strategy
archetypes, and recommends a MAX BID that updates as the draft unfolds.

THE THREE ARCHETYPES (your strategy framework):
  1. Reliable Stud  — high floor. Anchor. Pay up, protect budget for 1-2.
  2. Upside Gamble  — high ceiling / low floor. Hard-cap, collect several cheap.
  3. Value Pick     — safe production. Buy BELOW market. Where auctions are won.

HOW AUCTION VALUE IS COMPUTED (no paid feed needed):
  - Pull projected season points per player (from nflreadpy-derived data).
  - Compute "value above replacement" (VOR): points above the last startable
    player at each position. Replacement-level players are worth $1.
  - Scale total VOR across the league's total auction money so the dollar
    values sum correctly to (teams x budget) minus $1-per-roster-slot.
  This is the standard method real auction-value tools use.

LIVE DATA:
  pip install nflreadpy numpy pandas
  Player pool + stats come live. League settings you enter once at the top.

RUN:  python auction_engine.py
  Then follow the prompts: it shows max bids, you type who was bought for how
  much, and the board re-ranks with updated budgets.
"""

import sys
import numpy as np
import pandas as pd

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None


# ---- LEAGUE SETTINGS (edit these to match your league) ------------------
TEAMS = 12
BUDGET = 200                 # per team
ROSTER_SLOTS = 16            # total players per team
# starting lineup used to find "replacement level" per position
STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}  # FLEX = RB/WR/TE
SEASONS = [2024, 2025]
# simple PPR scoring for projecting points from stats
SCORING = {"pass_yd": 0.04, "pass_td": 4, "int": -1,
           "rush_yd": 0.1, "rush_td": 6,
           "rec": 1.0, "rec_yd": 0.1, "rec_td": 6}


# ---- DATA: pull real stats, project season points -----------------------
def load_player_points():
    """
    Pull play-by-play, aggregate to per-player season production, project
    points using the scoring settings. Returns a DataFrame:
      name, pos, team, proj_points, floor, ceiling
    floor/ceiling are derived from game-to-game consistency (for archetypes).
    """
    if nfl is None:
        raise RuntimeError("Install first:  pip install nflreadpy numpy pandas")

    pbp = nfl.load_pbp(SEASONS).to_pandas()
    pbp = pbp[(pbp["pass_attempt"] == 1) | (pbp["rush_attempt"] == 1)].copy()

    rows = []

    # rushing per player-game
    rush = pbp[pbp["rush_attempt"] == 1]
    for (pid, name, gid), g in rush.groupby(["rusher_player_id", "rusher_player_name", "game_id"]):
        rows.append(dict(pid=pid, name=name, game=gid,
                         rush_yd=g["rushing_yards"].sum(),
                         rush_td=g["rush_touchdown"].sum()))
    # receiving per player-game
    rec = pbp[pbp["pass_attempt"] == 1]
    for (pid, name, gid), g in rec.groupby(["receiver_player_id", "receiver_player_name", "game_id"]):
        if pid is None or (isinstance(pid, float) and np.isnan(pid)):
            continue
        rows.append(dict(pid=pid, name=name, game=gid,
                         rec=g["complete_pass"].sum(),
                         rec_yd=g["receiving_yards"].sum(),
                         rec_td=g["pass_touchdown"].sum()))
    # passing per player-game
    pas = pbp[pbp["pass_attempt"] == 1]
    for (pid, name, gid), g in pas.groupby(["passer_player_id", "passer_player_name", "game_id"]):
        if pid is None or (isinstance(pid, float) and np.isnan(pid)):
            continue
        rows.append(dict(pid=pid, name=name, game=gid,
                         pass_yd=g["passing_yards"].sum(),
                         pass_td=g["pass_touchdown"].sum(),
                         int=g["interception"].sum()))

    df = pd.DataFrame(rows).fillna(0)
    gm = df.groupby(["pid", "name", "game"], as_index=False).sum(numeric_only=True)

    # points per game
    def pts(r):
        return (r.get("pass_yd", 0) * SCORING["pass_yd"] + r.get("pass_td", 0) * SCORING["pass_td"]
                + r.get("int", 0) * SCORING["int"] + r.get("rush_yd", 0) * SCORING["rush_yd"]
                + r.get("rush_td", 0) * SCORING["rush_td"] + r.get("rec", 0) * SCORING["rec"]
                + r.get("rec_yd", 0) * SCORING["rec_yd"] + r.get("rec_td", 0) * SCORING["rec_td"])
    gm["ppg_pts"] = gm.apply(pts, axis=1)

    # per-player: mean ppg (proj), std (consistency), games
    agg = gm.groupby(["pid", "name"]).agg(
        ppg=("ppg_pts", "mean"), sd=("ppg_pts", "std"), games=("ppg_pts", "count")
    ).reset_index()
    agg = agg[agg["games"] >= 3].copy()
    agg["sd"] = agg["sd"].fillna(agg["ppg"] * 0.3)

    # position lookup from roster data, joined by gsis_id (pbp's player ids use
    # this same format) rather than name -- pbp names are abbreviated
    # ("J.Conner") while roster full_name is not ("James Conner"), so a
    # name-based join silently matches almost nothing.
    try:
        rosters = nfl.load_rosters([SEASONS[-1]]).to_pandas()
        pos_map = dict(zip(rosters["gsis_id"], rosters["position"]))
        team_map = dict(zip(rosters["gsis_id"], rosters["team"]))
        name_map = dict(zip(rosters["gsis_id"], rosters["full_name"]))
    except Exception:
        pos_map, team_map, name_map = {}, {}, {}

    agg["pos"] = agg["pid"].map(pos_map).fillna("NA")
    agg["team"] = agg["pid"].map(team_map).fillna("")
    agg["name"] = agg["pid"].map(name_map).fillna(agg["name"])
    agg = agg[agg["pos"].isin(["QB", "RB", "WR", "TE"])].copy()

    # project season points (17 games), and floor/ceiling on 1-10 scale for archetypes
    agg["proj_points"] = (agg["ppg"] * 17).round(1)
    # normalize floor/ceiling within position
    agg["floor"] = 0.0
    agg["ceiling"] = 0.0
    for pos, grp in agg.groupby("pos"):
        # floor: higher ppg + lower sd = higher floor
        consistency = grp["ppg"] / (grp["sd"] + 1)
        agg.loc[grp.index, "floor"] = _scale_1_10(consistency)
        # ceiling: ppg + upside (sd contributes to ceiling)
        upside = grp["ppg"] + grp["sd"] * 0.5
        agg.loc[grp.index, "ceiling"] = _scale_1_10(upside)
    return agg


def _scale_1_10(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([5] * len(series), index=series.index)
    return (1 + 9 * (series - lo) / (hi - lo)).round()


# ---- AUCTION VALUES: points -> dollars via VOR --------------------------
def compute_auction_values(players):
    """
    Value Above Replacement -> dollars.
    Replacement level = the (TEAMS * starters_at_pos)-th best player at that pos.
    Everyone above replacement shares the league's discretionary money.
    """
    players = players.copy()
    # number of starters drafted leaguewide per position (FLEX spread across RB/WR/TE)
    flex_share = STARTERS.get("FLEX", 0)
    start_counts = {
        "QB": STARTERS.get("QB", 0) * TEAMS,
        "RB": (STARTERS.get("RB", 0) + flex_share * 0.4) * TEAMS,
        "WR": (STARTERS.get("WR", 0) + flex_share * 0.4) * TEAMS,
        "TE": (STARTERS.get("TE", 0) + flex_share * 0.2) * TEAMS,
    }

    players["vor"] = 0.0
    for pos, grp in players.groupby("pos"):
        grp = grp.sort_values("proj_points", ascending=False)
        n_start = int(round(start_counts.get(pos, TEAMS)))
        if n_start >= len(grp):
            n_start = max(1, len(grp) - 1)
        replacement = grp["proj_points"].iloc[n_start]
        players.loc[grp.index, "vor"] = (grp["proj_points"] - replacement).clip(lower=0)

    # total money available beyond the $1 minimum per drafted slot
    total_money = TEAMS * BUDGET
    total_slots = TEAMS * ROSTER_SLOTS
    discretionary = total_money - total_slots  # $1 reserved per slot
    total_vor = players["vor"].sum()
    if total_vor <= 0:
        players["value"] = 1
        return players
    players["value"] = (1 + players["vor"] / total_vor * discretionary).round().astype(int)
    players = players.sort_values("value", ascending=False)
    return players


# ---- ARCHETYPES ---------------------------------------------------------
def archetype(row):
    if row["floor"] >= 7:
        return "stud"
    if row["ceiling"] >= 8 and row["floor"] <= 5:
        return "gamble"
    return "value"

ARCHE_RULES = {
    # bid multiplier vs base value + one-line guidance
    "stud":   (1.05, "High floor anchor. Pay up; protect budget for 1-2."),
    "gamble": (0.75, "High ceiling, unknown. Hard cap; collect several cheap."),
    "value":  (0.90, "Safe production. Buy BELOW value. Win the auction here."),
}


# ---- LIVE STATE ---------------------------------------------------------
class Draft:
    def __init__(self, players):
        self.players = players.reset_index(drop=True)
        self.owned = []   # (name, price)
        self.gone = set() # names bought by others
        self.my_budget = BUDGET
        self.my_slots = ROSTER_SLOTS

    def spent(self):
        return sum(p for _, p in self.owned)

    def remaining(self):
        return self.my_budget - self.spent()

    def slots_left(self):
        return self.my_slots - len(self.owned)

    def max_bid_now(self):
        # must keep $1 for every other empty slot
        return max(0, self.remaining() - (self.slots_left() - 1))

    def max_bid_for(self, row):
        arch = archetype(row)
        mult, _ = ARCHE_RULES[arch]
        raw = row["value"] * mult
        # scale by how much budget you have vs an even pace
        pace = self.remaining() / max(1, self.my_budget * self.slots_left() / self.my_slots)
        scaled = raw * min(1.15, pace)
        return int(max(1, min(self.max_bid_now(), round(scaled))))

    def available(self):
        taken = set(n for n, _ in self.owned) | self.gone
        av = self.players[~self.players["name"].isin(taken)].copy()
        av["arch"] = av.apply(archetype, axis=1)
        av["max_bid"] = av.apply(self.max_bid_for, axis=1)
        return av

    def show_top(self, n=20, pos=None):
        av = self.available()
        if pos:
            av = av[av["pos"] == pos]
        av = av.sort_values("value", ascending=False).head(n)
        print(f"\n  Budget ${self.remaining()}  |  max bid now ${self.max_bid_now()}  |  slots left {self.slots_left()}")
        print(f"  {'PLAYER':22s} {'POS':4s} {'TYPE':8s} {'VALUE':>6s} {'MAXBID':>7s}")
        print("  " + "-" * 55)
        for _, r in av.iterrows():
            tag = {"stud": "STUD", "gamble": "GAMBLE", "value": "VALUE"}[r["arch"]]
            print(f"  {r['name'][:22]:22s} {r['pos']:4s} {tag:8s} ${int(r['value']):>4d} ${int(r['max_bid']):>5d}")

    def show_tiers(self, pos=None, gap_factor=0.18, max_players=40):
        """
        Group available players into TIERS by detecting value cliffs.

        A "cliff" is where the dollar drop to the next player is large relative
        to the current tier's top value. Everything above a cliff is one tier;
        the cliff starts a new one. This shows you WHEN a position is about to
        fall off — i.e. pay up now, or wait because there's more of the same.

        gap_factor: a drop bigger than this fraction of the tier-top value
                    triggers a new tier (0.18 = an ~18% drop = a cliff).
        """
        av = self.available()
        if pos:
            av = av[av["pos"] == pos]
        av = av.sort_values("value", ascending=False).head(max_players).reset_index(drop=True)
        if av.empty:
            print("    no players available")
            return

        # walk down the value list; start a new tier when the drop is a cliff
        tiers = []
        current = [av.iloc[0]]
        tier_top = av.iloc[0]["value"]
        for i in range(1, len(av)):
            v = av.iloc[i]["value"]
            drop = tier_top - v
            # cliff if the drop exceeds gap_factor of the tier's top value
            if tier_top > 0 and drop >= max(2, tier_top * gap_factor):
                tiers.append(current)
                current = [av.iloc[i]]
                tier_top = v
            else:
                current.append(av.iloc[i])
        tiers.append(current)

        header = f"TIERS{' · ' + pos if pos else ''}"
        print(f"\n  {header}   (budget ${self.remaining()} · max bid ${self.max_bid_now()})")
        print("  " + "=" * 58)
        for ti, tier in enumerate(tiers, 1):
            vals = [int(p["value"]) for p in tier]
            hi, lo = max(vals), min(vals)
            span = f"${hi}" if hi == lo else f"${lo}–${hi}"
            # how many players left in this tier = how much you can wait
            n = len(tier)
            urgency = "LAST ONE — pay up" if n == 1 else ("thin — 2 left" if n == 2 else f"{n} available")
            print(f"\n  ── Tier {ti}  ({span})  ·  {urgency}")
            for p in tier:
                tag = {"stud": "STUD", "gamble": "GAMBLE", "value": "VALUE"}[p["arch"]]
                print(f"      {p['name'][:22]:22s} {p['pos']:4s} {tag:8s} ${int(p['value']):>4d}  max ${int(p['max_bid']):>4d}")
        # tell the user where the NEXT cliff is
        if len(tiers) >= 2 and len(tiers[0]) <= 3:
            print(f"\n  ⚠ Only {len(tiers[0])} player(s) in the top tier — cliff after them.")


# ---- INTERACTIVE LOOP ---------------------------------------------------
def run_interactive(draft):
    print("\n" + "=" * 60)
    print("  AUCTION ENGINE — live max-bid recommendations")
    print("=" * 60)
    print("""
  Commands:
    top [POS]           show top available (optional: QB/RB/WR/TE)
    tiers [POS]         group available players into value tiers / cliffs
    me NAME PRICE       record a player YOU won for $PRICE
    out NAME            record a player someone ELSE won
    bid NAME            show your recommended max bid for one player
    roster              show your team + spend
    undo                undo last action
    quit                exit
""")
    while True:
        try:
            cmd = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        parts = cmd.split()
        op = parts[0].lower()

        if op == "quit":
            break
        elif op == "top":
            draft.show_top(pos=parts[1].upper() if len(parts) > 1 else None)
        elif op == "tiers":
            draft.show_tiers(pos=parts[1].upper() if len(parts) > 1 else None)
        elif op == "me" and len(parts) >= 3:
            name = " ".join(parts[1:-1]); price = int(parts[-1])
            draft.owned.append((name, price))
            print(f"    ✓ You won {name} for ${price}. Budget left ${draft.remaining()}.")
        elif op == "out" and len(parts) >= 2:
            name = " ".join(parts[1:])
            draft.gone.add(name)
            print(f"    · {name} off the board.")
        elif op == "bid" and len(parts) >= 2:
            name = " ".join(parts[1:])
            row = draft.players[draft.players["name"].str.lower() == name.lower()]
            if row.empty:
                print("    ? player not found"); continue
            r = row.iloc[0]; arch = archetype(r)
            _, advice = ARCHE_RULES[arch]
            print(f"    {r['name']} ({r['pos']}) — {arch.upper()}")
            print(f"    value ${int(r['value'])}  |  YOUR MAX BID ${draft.max_bid_for(r)}")
            print(f"    {advice}")
        elif op == "roster":
            print(f"\n    Your team ({len(draft.owned)}/{draft.my_slots}), spent ${draft.spent()}:")
            for n, p in draft.owned:
                print(f"      ${p:>3d}  {n}")
            print(f"    Budget left: ${draft.remaining()}")
        elif op == "undo":
            if draft.owned:
                n, p = draft.owned.pop(); print(f"    undid: you won {n} (${p})")
            elif draft.gone:
                draft.gone.pop(); print("    undid last 'out'")
        else:
            print("    ? unknown command")


if __name__ == "__main__":
    if nfl is None:
        print("\n[!] Install dependencies first:")
        print("    pip install nflreadpy numpy pandas\n")
        sys.exit(1)
    print("Pulling live player data (first run downloads, then caches)...")
    players = load_player_points()
    players = compute_auction_values(players)
    print(f"Loaded {len(players)} players with computed auction values.")
    draft = Draft(players)
    draft.show_top(15)
    run_interactive(draft)
