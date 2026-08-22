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

## What to add next (natural extensions)

- Live odds via The Odds API (feeds real lines into `find_edge`)
- Injury/inactive scraping to auto-adjust volume
- Weather for totals (wind/precip move scoring)
- Proper correlation matrix (QB-WR stacks) for sharper spreads
- Kelly bet sizing off the EV numbers
