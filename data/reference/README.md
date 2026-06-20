Place a bundled injury CSV in this folder when you want the predictor to have a stable
backend fallback even if the live official NBA injury refresh returns no reports.

Recommended filename:
- official_nba_injuries_by_team_2022_2025.csv

The predictor checks `data/raw` first for freshly scraped injury files, then falls back
to matching files in `data/reference`.
