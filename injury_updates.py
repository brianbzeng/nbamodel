"""Refresh the official pregame injury features used by the NBA predictor."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from filelock import FileLock

from cleaner import TEAM_MAP_BREF
from game_updates import FIRST_UPDATE_DATE, NBA_TIMEZONE, fetch_scoreboard, nba_competition, write_snapshot
from injury_reports import (
    QUARTER_HOUR_REPORTS_START,
    ReportUnavailable,
    fetch_report,
    fetch_report_index,
    prepare_report_parser,
)
from predictor import LIVE_INJURIES_FILE, STATUS_IMPACT_MULTIPLIERS, aggregate_team_injuries

INJURY_KEY = ["game_date", "team"]
EXCLUDED_GAMES = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SUSPENDED"}


def scheduled_teams(events: list[dict], start: date, end: date) -> dict:
    teams = {}
    for event in events:
        competition = nba_competition(event)
        if competition is None or competition["status"]["type"]["name"] in EXCLUDED_GAMES:
            continue
        tipoff = datetime.fromisoformat(competition["date"]).astimezone(NBA_TIMEZONE).replace(tzinfo=None)
        if not start <= tipoff.date() <= end:
            continue
        codes = [TEAM_MAP_BREF.get(team["team"]["displayName"]) for team in competition["competitors"]]
        if len(codes) == 2 and all(codes):
            for code in codes:
                teams[(pd.Timestamp(tipoff.date()), code)] = (tipoff, int(event["season"]["year"]))
    return teams


def report_candidates(tipoff: datetime, now: datetime, index: list[datetime]) -> list[datetime]:
    cutoff = min(now, tipoff - timedelta(microseconds=1))
    first = tipoff - timedelta(hours=36)
    last_index = bisect_right(index, cutoff)
    indexed = [stamp for stamp in index[max(0, last_index - 4):last_index] if stamp >= first]

    # Older reports can remain on the league CDN after their links leave the index.
    interval = 15 if cutoff >= QUARTER_HOUR_REPORTS_START else 60
    last = cutoff.replace(minute=cutoff.minute // interval * interval, second=0, microsecond=0)
    guesses = [last - timedelta(minutes=offset * interval) for offset in range(5)]
    guesses.extend(cutoff.replace(hour=hour, minute=0, second=0, microsecond=0) for hour in (17, 14, 12, 9))
    if indexed and indexed[-1].date() == cutoff.date():
        guesses = [stamp for stamp in guesses if stamp > indexed[-1]]
    return sorted({stamp for stamp in [*indexed, *guesses] if first <= stamp <= cutoff}, reverse=True)


def _fetch_report_or_empty(timestamp: datetime) -> pd.DataFrame:
    try:
        return fetch_report(timestamp)
    except ReportUnavailable:
        return pd.DataFrame()
    except (RuntimeError, ValueError) as exc:
        print(f"Trying an earlier injury report. {exc}")
        return pd.DataFrame()


def _report_features(report: pd.DataFrame) -> dict:
    if report.empty:
        return {}

    player_rows = report[report["status"].str.upper().isin(STATUS_IMPACT_MULTIPLIERS)]
    features = aggregate_team_injuries(player_rows)
    empty_reports = report[report["reason"].str.upper().isin({"NO INJURIES TO REPORT", "NO INJURIES REPORTED"})]
    if not empty_reports.empty:
        healthy = empty_reports[[*INJURY_KEY, "report_datetime"]].drop_duplicates().rename(
            columns={"report_datetime": "latest_report_datetime"}
        )
        metrics = features.columns.difference(healthy.columns)
        healthy = healthy.assign(**dict.fromkeys(metrics, 0)).reindex(columns=features.columns)
        features = pd.concat([healthy, features], ignore_index=True).drop_duplicates(INJURY_KEY, keep="last")
    return {key: row for key, row in features.set_index(INJURY_KEY).iterrows()}


def collect_injuries(teams: dict, existing: pd.DataFrame, now: datetime, indexes: dict) -> pd.DataFrame:
    saved = existing.set_index(INJURY_KEY)["latest_report_datetime"].to_dict()
    pending = {
        key: iter(report_candidates(tipoff, now, indexes[season]))
        for key, (tipoff, season) in teams.items()
        if key not in saved or key[0].date() >= now.date() - timedelta(days=1)
    }
    if not pending:
        return existing.iloc[:0]

    prepare_report_parser()
    selected = {}
    cache = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending:
            batches = defaultdict(list)
            for key, candidates in list(pending.items()):
                stamp = next(candidates, None)
                if stamp is None or (key in saved and stamp <= saved[key]):
                    del pending[key]
                    continue
                batches[stamp].append(key)

            new_stamps = [stamp for stamp in batches if stamp not in cache]
            for stamp, report in zip(new_stamps, pool.map(_fetch_report_or_empty, new_stamps)):
                cache[stamp] = _report_features(report)

            for stamp, keys in batches.items():
                for key in keys:
                    row = cache[stamp].get(key)
                    if row is None:
                        continue
                    published = row["latest_report_datetime"]
                    if saved.get(key, datetime.min) < published < teams[key][0] and published <= now:
                        selected[key] = row
                        del pending[key]

    if not selected:
        return existing.iloc[:0]
    fresh = pd.DataFrame.from_dict(selected, orient="index")
    fresh.index = pd.MultiIndex.from_tuples(fresh.index, names=INJURY_KEY)
    return fresh.reset_index().reindex(columns=existing.columns)


def refresh_injury_updates(path: Path = LIVE_INJURIES_FILE, since: date | None = None) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock"), timeout=120):
        existing = (
            pd.read_csv(path, parse_dates=["game_date", "latest_report_datetime"])
            if path.exists() else aggregate_team_injuries(pd.DataFrame())
        )
        now = datetime.now(NBA_TIMEZONE).replace(tzinfo=None)
        today = now.date()
        start = today - timedelta(days=2)
        if not existing.empty:
            start = min(start, existing["game_date"].max().date() - timedelta(days=1))
        start = max(FIRST_UPDATE_DATE, since or start)
        end = today + timedelta(days=1)
        if start > end:
            raise ValueError("The injury refresh start date must be on or before tomorrow.")
        periods = list(pd.period_range(start=start, end=end, freq="M").strftime("%Y%m"))
        periods.extend(day.strftime("%Y%m%d") for day in (today - timedelta(days=1), today, end))
        with ThreadPoolExecutor(max_workers=3) as pool:
            boards = list(pool.map(fetch_scoreboard, periods))
        teams = scheduled_teams([event for board in boards for event in board], start, end)
        print(f"Checking NBA injury reports for {len(teams):,} scheduled team-games.")
        indexes = {}
        for season in {season for _, season in teams.values()}:
            try:
                indexes[season] = fetch_report_index(season)
            except (requests.RequestException, ValueError) as exc:
                print(f"Official report index unavailable for {season}. Trying the NBA archive. {exc}")
                indexes[season] = []
        fresh = collect_injuries(teams, existing, now, indexes)

        if not fresh.empty and not existing.empty:
            # A reissued PDF with the same features does not trigger a redeploy.
            metrics = existing.columns.difference([*INJURY_KEY, "latest_report_datetime"])
            previous = existing.set_index(INJURY_KEY).reindex(pd.MultiIndex.from_frame(fresh[INJURY_KEY]))
            changed = fresh[metrics].ne(previous[metrics].reset_index(drop=True)).any(axis=1)
            fresh = fresh.loc[changed]
        updated = existing
        if not fresh.empty:
            combined = fresh if existing.empty else pd.concat([existing, fresh], ignore_index=True)
            updated = combined.drop_duplicates(INJURY_KEY, keep="last").sort_values(INJURY_KEY).reset_index(drop=True)
            write_snapshot(updated, path)

        saved = set(updated[INJURY_KEY].itertuples(index=False, name=None))
        yesterday = today - timedelta(days=1)
        yesterday_teams = {key for key in teams if key[0].date() == yesterday}
        if yesterday_teams and not (yesterday_teams & saved):
            raise RuntimeError(f"No official pregame injury reports were saved for {yesterday}.")
        completed_tipoffs = {key for key, (tipoff, _) in teams.items() if tipoff <= now}
        state = "Updated" if not fresh.empty else "Checked"
        return {
            "status": "refreshed" if not fresh.empty else "cached",
            "message": (
                f"{state} official NBA injury reports. Saved {len(updated):,} team-date records. "
                f"Pregame coverage in checked window: {len(completed_tipoffs & saved):,}/{len(completed_tipoffs):,} team-games."
            ),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=date.fromisoformat, help="Backfill reports starting on YYYY-MM-DD.")
    print(refresh_injury_updates(since=parser.parse_args().since)["message"])
