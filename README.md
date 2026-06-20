# NBA Odds Predictor

NBA game scraper and Elo rating browser app built from Basketball-Reference data.

## What it does

- Scrapes regular-season game results
- Cleans and normalizes team names into NBA abbreviations
- Runs an Elo model with home-court advantage, margin-of-victory weighting, and season regression
- Renders the project in a browser with a shared hotbar and multiple pages

## Pages

- `Home` explains the project, the pipeline, and the current-season behavior
- `Leaderboard` shows the latest team ratings and can be refreshed from the processed data
- `Predictor` scores a custom matchup directly in the app using team form, Elo, blended strength, and injury context, with the tree model using an Elo-enhanced feature set
  - It also keeps a latest-season evaluation summary and exports predicted-vs-true results as a CSV
- `Scrape` imports NBA games for a chosen season range and refreshes the processed dataset
- `NBA Elo Model` walks through the update rule, logistic win probability, and season regression in math notation

## Controls

- `Refresh leaderboard` reloads the latest processed data and reruns the standings view
- `Refresh predictor data` rebuilds the predictor training range, refreshes injury context when available, and updates the latest-season comparison export
- `Reset data` wipes generated raw, processed, and result files so you can start clean

## Run it

1. Install dependencies from `requirements.txt`
2. Start the web app with `python main.py`
3. Open the local browser URL shown in the terminal

## Data outputs

- `data/raw/bref_games_2016_2025.csv`
- `data/processed/games_2016_2025_normalized.csv`
- Generated results under `data/results/`
- Latest-season predictor exports under `data/exports/predictor/`

## Notes

- `main.py` is now the browser-app launcher.
- The main scraper now walks Basketball-Reference month pages so full seasons are captured instead of relying on a single season hub page.
- If live injury refresh does not return reports, the predictor can fall back to a bundled CSV in `data/reference/`.
- `monthly_scraper.py` and `updater.py` are legacy/manual tools and are not the primary entrypoint anymore.
- A local post-commit hook is configured in `.githooks/post-commit` to push to `origin` automatically once a remote is added.
- Scraping the current season only imports games that have already been played, so the leaderboard reflects the current Elo state of the latest processed data.
