# NBA Odds Predictor

NBA game scraper, Elo tracker, and home-win predictor built from Basketball-Reference history and current ESPN scoreboard results.

## What it does

- Scrapes regular-season NBA game results across full season month pages
- Cleans and normalizes team names into NBA abbreviations
- Runs an Elo model with home-court advantage, margin-of-victory weighting, and season regression
- Trains a home-win predictor with engineered pre-game features
- Renders everything in a browser app with leaderboard, predictor, scraper, about, and inner-workings pages

## Pages

- `Home` explains the project and current workflow
- `Leaderboard` shows the latest Elo ratings
- `Predictor` shows today’s model win probabilities and the season’s completed games in History, newest first
- `Scrape` refreshes game data and injury reports
- `About` lists contact information and lets visitors send questions by email
- `Inner-Workings` explains how the scraper, predictor, and Elo model work

## Predictor features

The predictor uses pre-game context only:

- team strength from season win percentage and scoring margin
- recent form from rolling five-game performance
- rest days between games
- injury counts and injury impact when official reports are available

## Controls

- `Refresh leaderboard` checks for new final scores and reruns the Elo standings
- `Reset data` clears generated exports, caches, and results while keeping the bundled datasets

## Automatic game and injury updates

GitHub Actions checks every hour at :17 from noon through 3 a.m. Eastern, with a
6:17 a.m. reconciliation. It saves ESPN finals and score corrections to
`data/raw/nba_game_updates.csv` and official NBA team injury features to
`data/raw/official_nba_injuries_by_team_live.csv`. The leaderboard rates the latest
season. The predictor trains on the full game history and joins injury features
by team and game date using reports published before tipoff. Reports for upcoming
games can feed matchup predictions before the final score is available.

The workflow saves the schedule and picks in `data/raw/nba_daily_predictions.csv`.
It adds pregame checks between hourly runs when a game tips within 60 minutes.
Those checks pull new finals and official injury reports before fitting the random
forest, including completed games already observed that day. Picks start in the hour
before tipoff, update as inputs change, and freeze at tipoff. History prefers each
saved pregame pick and labels replays separately. Replays train only on games from
earlier dates and live accuracy counts only saved pregame picks.

Render deploys changes to the snapshots. If injury retrieval fails, the workflow
can still publish finals and retry injuries on the next run. A verification commit
after 45 idle days keeps the offseason schedule active without redeploying Render.

Run `python game_updates.py`, `python injury_updates.py`, `python daily_predictions.py`,
and `python prediction_replay.py` in order to refresh a local checkout, or run
**Refresh NBA games** from the repository’s Actions tab. Use
`python injury_updates.py --since YYYY-MM-DD` to backfill older reports.

## Run it

1. Create and activate a virtual environment
2. Install dependencies from `requirements.txt`
3. Start the app with `python main.py`
4. Open the local browser URL shown in the terminal

Example setup:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Deploy with Render and Cloudflare

Render runs the Flask, scikit-learn, and Java backend. Cloudflare manages the
`brianbzeng.com` DNS, proxy, and HTTPS connection.

### 1. Deploy the Render Blueprint

1. Push this repository, including `render.yaml`, to GitHub.
2. Sign in to Render and connect the GitHub account that owns the repository.
3. Select `New > Blueprint`, choose this repository, and apply the Blueprint.
4. Wait for the Docker build and `/health` check to pass.
5. Open the generated `nbamodel.onrender.com` URL and test the app.

The Blueprint starts on Render's free web-service plan. It automatically
deploys new commits and generates the Flask secret key.

### 2. Add the custom domain in Render

1. Open the `nbamodel` service in Render.
2. Go to `Settings > Custom Domains`.
3. Add `brianbzeng.com`. Render also adds or redirects `www.brianbzeng.com`.
4. Keep this page open while configuring Cloudflare DNS.

### 3. Configure Cloudflare DNS

1. Open `brianbzeng.com` in Cloudflare.
2. Under `SSL/TLS > Overview`, choose `Full`.
3. Under `DNS > Records`, remove conflicting `A`, `AAAA`, or `CNAME` records
   for `@` and `www`. Render does not support IPv6, so remove `AAAA` records.
4. Add a CNAME named `@` targeting the service's exact
   `nbamodel.onrender.com` hostname.
5. Add a second CNAME named `www` with the same target.
6. Initially set both records to `DNS only` so Render can verify the domain and
   issue its certificate.
7. Return to Render and select `Verify`.
8. After Render reports a valid certificate, optionally switch both Cloudflare
   records to `Proxied`.

### Deployment behavior

- Free Render services sleep after 15 minutes without traffic. The first visit
  after that can take about a minute.
- The bundled game and injury CSVs make prediction available after each start.
- Files created by browser scrapes are temporary on the free plan and disappear
  after a restart or redeploy.
- Persistent CSV changes require a paid Render service with a disk mounted at
  `/app/data`.
- Cloudflare's normal proxy timeout is 120 seconds. Full historical scrapes
  should eventually run as background jobs instead of one browser request.
- Use one Gunicorn worker because generated state is stored in CSV files.
- Render limits the scraper to four concurrent downloads and two parsing
  threads so refresh work does not starve the `/health` endpoint.

The earlier `compose.yaml` setup remains available for local Docker testing or
for running the app behind a Cloudflare Tunnel later.

## Data outputs

- `data/raw/bref_games_2016_2025.csv`
- `data/raw/nba_game_updates.csv`
- `data/processed/games_2016_2025_normalized.csv`
- `data/raw/nba_daily_predictions.csv`
- `data/raw/nba_prediction_replay.csv`
- scrape exports under `data/exports/scrapes/`
- evaluation summaries under `data/results/`

## Key files

- [app.py](/Users/brianzeng/Documents/Visual%20Studio%20Code/nbamodel/app.py): Flask web app
- [predictor.py](/Users/brianzeng/Documents/Visual%20Studio%20Code/nbamodel/predictor.py): feature engineering and prediction workflow
- [model.py](/Users/brianzeng/Documents/Visual%20Studio%20Code/nbamodel/model.py): Elo model
- [scraper.py](/Users/brianzeng/Documents/Visual%20Studio%20Code/nbamodel/scraper.py): Basketball-Reference scraper
- [cleaner.py](/Users/brianzeng/Documents/Visual%20Studio%20Code/nbamodel/cleaner.py): cleaning and normalization helpers

## Notes

- `main.py` is the browser-app launcher.
- The Basketball-Reference scraper now walks monthly season pages so full seasons are captured instead of only the first month.
- Injury refresh uses `nbainjuries` when official reports are available, and stored CSVs can support the predictor workflow.
- `monthly_scraper.py` and `updater.py` are legacy/manual helpers and are not the main entrypoint.
- The duplicate `data/raw/* 2.csv` files are not part of the intended pipeline.
