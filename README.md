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

How the model works, and what it can't see: `props_model.py`. It does not know
about injuries, role changes, weather, or game script, and it simulates players
independently. Treat large model-vs-market gaps as "the model is missing
something" until proven otherwise.

## What to add next (natural extensions)

- Live odds via The Odds API (feeds real lines into `find_edge`)
- Injury/inactive scraping to auto-adjust volume
- Weather for totals (wind/precip move scoring)
- Proper correlation matrix (QB-WR stacks) for sharper spreads
- Kelly bet sizing off the EV numbers
