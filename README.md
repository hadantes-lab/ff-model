# NFL Betting Probability Model

A personal-use model that puts probabilities on player props, spreads, and
totals — then compares them to the market to flag value. Built on real nflverse
data. Includes a backtest so you can check whether it actually works.

## What it is (and isn't)

**Is:** a real, working simulation model. Pulls live play-by-play, builds
per-player stat distributions, runs Monte Carlo, converts to probabilities,
and finds edges vs. betting lines. Backtested against historical results.

**Isn't:** a guaranteed money-maker. No honest model claims that. The market
is sharp. This gives you defensible probabilities and a way to VALIDATE them —
not certainty. Anyone selling certainty is lying.

## The three files

| File | Job |
|------|-----|
| `projections.py` | Pulls nflverse data, builds per-player mean+std for each stat |
| `simulate.py` | Monte Carlo engine: props, spreads, totals, edge/EV math |
| `backtest.py` | Walk-forward out-of-sample test with hit rate, ROI, and CLV |

The pipeline: **projections feed the simulator; the simulator feeds edges;
the backtest proves it out.** Spreads are built by summing player projections
into team scores — the same engine does everything.

## Setup (one time)

1. Install Python 3.10+ from python.org if you don't have it.
   Check: `python --version`
2. Install the libraries:
   ```
   pip install nflreadpy numpy pandas
   ```

## Running it

**See it work immediately (no data needed):** each file has a demo built in.
   ```
   python simulate.py      # prop + game simulation on sample numbers
   python backtest.py      # backtest summary on synthetic data
   ```

**Run it for real (pulls live data):**
   ```
   python projections.py   # first run downloads nflverse data, then caches
   ```
   This prints real projections for current players. From there, feed those
   projections into `simulate.py`'s functions with live odds.

**Run the tests:**
   ```
   python -m unittest discover -s tests -v
   ```

## How to read the output

- **model_prob** — your model's probability the bet hits
- **implied_prob** — the market's probability (from the odds, vig removed)
- **edge_pts** — model minus market, in percentage points. Positive = value.
- **ev_per_dollar** — expected profit per $1 staked
- **CLV** (backtest) — did you beat the closing line? The #1 sign of real skill.

## The honest caveats (read these)

1. **Injuries at lock time** are the model's biggest blind spot. If a starter is
   ruled out after you project, everything shifts. Don't bet props with unresolved
   injury news.
2. **Small samples lie.** The backtest shouts at you below 200 bets. Respect it.
3. **Beating the closing line (CLV) matters more than early results.** Hit rate is
   noisy; CLV signals edge faster.
4. **Correlation is simplified.** The game-script factor is a first-order fix, not
   a full covariance model. Good enough to start, not the last word.

## Shareable draft board (HTML)

A point-and-click version of `auction_engine.py`, with an Auction and a
Snake mode. Mark players "Won"/"Gone" (auction) or "Draft" (snake) and
watch max bids, budget, and pick order update live. No Python needed to
use it, just a browser.

Tiers are built from current expert consensus draft rank (ADP) — FantasyPros
overall redraft rankings pulled via `nflreadpy`'s `load_ff_rankings`, merged
onto this tool's own per-player projections by name. A new tier starts
whenever the gap to the next player exceeds the experts' own disagreement
(std. dev.) at that range. Players outside the ADP consensus land in an
"Unranked" group at the bottom, sorted by this tool's own value model.

**Regenerate it after pulling fresh data:**
```
python export_snapshot.py > web/snapshot.json
python web/build_page.py
```
This writes `web/auction_board.html`, a single self-contained file (real
player values baked in) you can open directly or upload anywhere to share.

## Player props simulator (HTML)

Simulates every offensive player prop on this week's slate 20,000 times and
compares the result to the sportsbook lines: model probability vs. the market's
no-vig probability, best price across books, and EV per $1. Click any row for
the simulated distribution, last-10-games hit rate, and a calculator where you
can type in *your* book's line and price.

**One-time setup** (real lines need a free key from https://the-odds-api.com):
```
# create a file named .env in this folder containing one line (it is git-ignored):
ODDS_API_KEY=your_key_here
```

**Each time you want fresh lines:**
```
python export_props.py            # pulls lines + simulates -> web/props_snapshot.json
python web/build_props_page.py    # -> web/props.html (one self-contained file)
```
`python export_props.py --sample` builds the page from *demo* lines made up from
the model, with no key and no credits, just to see the layout.

Credits: listing games is free; each game costs one credit per market. The default
run (6 core markets) is at most ~90 credits for a 15-game week, and responses are
cached for 30 minutes so re-running is free. `--all-markets` adds attempts,
completions, interceptions and rush+rec yards; `--max-events N` limits games.
Whether player props are included on a free plan is up to the Odds API, not this code.

**Scheduled refresh.** `.github/workflows/refresh-site.yml` rebuilds the site every Sunday,
Monday and Thursday morning (8 AM Central) and publishes it to GitHub Pages: the draft board
at `/draft/` (always) and the props page at `/props/` (when the key below exists), using only
games starting in the next 24 hours. Add the key as a repository secret:
```
gh secret set ODDS_API_KEY
```
(it prompts for the value; it is never written to the repo). Until the secret exists the props
part is skipped without failing. Run it any time from the repo's Actions tab.

**Testing and refining the model.** Nothing in the projection engine is taken on faith; it is
checked against history:
```
python backtest_props.py            # walk-forward: project every 2024-25 game from earlier games only,
                                    # compare to what happened (accuracy, bias, interval calibration)
python tune_props.py --write        # re-fit the bias calibration and game-environment effects
                                    # -> model_calibration.json (read by export_props.py)
```
What that found so far: projections for high-usage players run ~5% hot (fixed by calibration, which
improved every market); a defense-strength adjustment helps slightly; the game total/spread genuinely
moves volume and yardage out of sample (higher totals lift receiving and passing-TD projections,
favorites pass less); and man/zone coverage matchups are a very weak signal (`matchups.py`).
Two limits: there are no historical *prop lines* to score edges against, only outcomes, so the
model's agreement with the market can't be validated yet; and anytime TD is opt-in
(`--markets player_anytime_td`) because long shots are mostly noise.

**Game sides and totals.** `game_odds.py` rolls the same per-player projections up into team point
totals, to price spreads and totals the identical bottom-up way: simulate the players (with a
shared per-team "game script" factor so a team's players move together, not independently), sum
to a score, compare to the market's posted line. Team points are a real fitted formula, not a
guess -- regressed on 2021-25 team-games:
```
points = 2.09 + 0.0297*rush_yds + 0.0156*pass_yds + 5.60*off_td + noise
```
(R² = 0.81, MAE 3.35 points; `noise` matches the real residual's shape). What it does not
model -- defense/special-teams scores, kicking, 2-point tries -- is the unexplained ~19%.
Early in a season, summing many small-sample player TD rates can overstate a hot team's total
(caught by comparing a real roster's simulated total to the actual league-average score before
this shipped), so the simulated probability is shrunk toward the market's own price-implied
probability, more so with less history behind the roster (`GAME_LAMBDA_MAX` in `game_odds.py`
-- a judgment call, like the player-prop blend, pending real tracked results). This needs no
extra Odds API credits: it reuses the same bulk game-lines call the environment adjustment
above already makes. `export_props.py` adds a `games` list to the snapshot automatically; the
page's "Game Lines" tab shows it, and `props_tracker.py` logs and grades these picks the exact
same way it does player props (against final scores, not player stats).

**Tracking real results.** That second limit is closed over time by `props_tracker.py`: every real
(non-sample) run logs each prop's line and probabilities, then once a game is a few hours old,
grades it against the actual stat:
```
python props_tracker.py grade              # fills in results for finished games
python props_tracker.py review             # audit the most recently graded week
python props_tracker.py review --week 4     # a specific week
python props_tracker.py report              # pick win rate, ROI, and calibration
python props_tracker.py report --json        # + writes web/track_record.json and web/player_history.json
```
`tracking/props_log.csv` is the record; the scheduled workflow runs `grade` and `report --json`
before each week's pull and commits it back to the repo, since Actions runs don't persist state
otherwise. Only our own numbers are stored -- never a sportsbook odds board -- consistent with
the Odds API's terms. The props page shows the aggregate as a "Track record" panel, and a
"Player History" tab lets you pick any tracked player and see a chart of the line recorded each
week against what actually happened, for the current season. Early on this will be a small,
noisy sample; give it a few weeks before reading much into it.

**Prediction lines: the model's own opinion, independent of the market.** Every prop and game
also gets a "prediction line" (`predicted_line` for props, `predicted_spread`/`predicted_total`
for games) computed from the simulated distribution alone, before comparing to anything the
Odds API returns -- the half-point line where the model's own simulation is closest to a coin
flip (`props_model.fair_line`). Once the real line is pulled, the page shows both side by side
with the gap between them. This is a second, more direct way to see how far the model's own
view sits from the market's, separate from the probability-level edge.

**Backup players and role changes.** `depth_chart.py` catches the case a game log alone misses:
a backup who is about to play a starter's snaps because the real starter is out. It compares a
player's own recent usage (pass attempts, carries, targets) to his team's, and if a more-used
teammate at his position is Out/Doubtful/IR this week, scales his volume projection up toward
what the *offense* -- not him personally -- has recently done there (never his per-touch
efficiency, which still comes from his own history). Verified against a real slate before
shipping (Chicago's Tyson Bagent correctly promoted 2.4x with Caleb Williams out); a first version
also mis-flagged a healthy RB committee's lead back and starting WRs as "promoted" because it
used a fixed share-of-team-usage threshold, which works for a QB's near-monopoly on attempts
but not for positions that split touches more evenly -- fixed by requiring an actual injured,
more-used teammate to exist, no threshold needed.

**Refining the market-blend weight from real results, not just history.** `tune_props.py`
validates the projection engine against 2023-25 *stats*; there's no way to check its agreement
with the *market* that way, since no historical prop lines exist. `props_tracker.py refine`
closes that gap using props_tracker's own tracked outcomes: once a market has 50+ graded, priced
picks, it grid-searches the trust weight (see `blend_toward_market` above) that would have
minimized Brier score against what actually happened, and only overrides the static default if
that clearly beats it (`REFINE_MIN_N`, `REFINE_MIN_IMPROVEMENT` in `props_tracker.py` --
deliberately conservative, so a mediocre early sample can't swing the weight around). The
scheduled workflow runs this before each week's pull, writing `live_calibration.json` fresh each
time (not committed -- cheap to regenerate from the already-persisted tracking log).

**Line movement.** Every logged prop/game keeps its `opening_line` (set once, never overwritten)
alongside the latest `line`, so a later pull that differs from an earlier one is visible as
`line_move`, not silently overwritten. `python props_tracker.py moves` lists moves past a
per-kind threshold (`BIG_MOVE` in `props_tracker.py`); the biggest ones also show in the page's
Track Record panel. In practice this mostly populates for games: player props are pulled once
under the `--days 1` schedule, but game lines are pulled every run for the whole week's slate at
a flat ~2-credit cost regardless of how many games it covers (see `game_odds.py` below), so the
same game genuinely does get re-priced Sunday, Monday, and Thursday.

**Team stats and what they say about the over/under.** `team_stats.py` builds one row per team per
game from play-by-play: plays, yards, yards per play (offense, and the same figure *allowed* by the
defense), first downs, third-down rate, neutral-state pass rate and pass rate over expected
(tied or leading, quarters 1-3), and time of possession from the drive clocks. Each Game Lines card
shows both teams' recency-weighted profile plus each side's expected plays, yards per play and yards
for that matchup (offense yards per play adjusted by how the opposing defense differs from league
average). `game_factors.py` tests every stat against the posted lines on 2021-26 games (1,374 with
lines), using only each team's *prior* games so a result never leaks into its own features:
```
python game_factors.py            # correlations + walk-forward test
python game_factors.py --json     # + writes web/game_factors.json
```
What it found, and it is a null result: the stats track the posted total closely (expected yards
r = 0.68, yards per play r = 0.63, first downs r = 0.63) because the market already prices them in,
and none of the 22 stats predicts how far a game lands from the line (strongest |r| = 0.06, about what
chance gives across 22 tries). A walk-forward ridge model fit on earlier seasons and scored on the
next had out-of-sample R^2 of -0.006 for totals and 0.000 for sides, and leaning over/under on it was
right 51.1% of the time (95% CI 47.8-54.4%, break-even 52.4%). So the stats are shown as context and
are deliberately *not* an input to the model's picks. Time of possession does move with plays
(r = 0.73 within a game, 0.71 holding yards per play and third-down rate fixed), but seconds per
play does not (r = -0.25 with plays), meaning possession mostly tracks how many snaps a team runs,
not how fast it runs them.

How the model works, and what it still can't see: `props_model.py`. It does not know about
weather or in-game injuries, and it simulates players independently of each other (aside from
the shared game-script factor `game_odds.py` uses for team totals). Treat large model-vs-market
gaps as "the model is missing something" until proven otherwise.

## What to add next (natural extensions)

- Live odds via The Odds API (feeds real lines into `find_edge`)
- Injury/inactive scraping to auto-adjust volume
- Weather for totals (wind/precip move scoring)
- Proper correlation matrix (QB-WR stacks) for sharper spreads
- Kelly bet sizing off the EV numbers
