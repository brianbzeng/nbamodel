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
from player_injuries import (
    LIVE_PLAYER_INJURIES_FILE,
    PLAYER_INJURY_COLUMNS,
    PLAYER_INJURY_KEY,
    read_live_player_injury_reports,
    submitted_injury_rows,
)
from predictor import LIVE_INJURIES_FILE, aggregate_team_injuries

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

    submitted = submitted_injury_rows(report)
    player_rows = submitted[submitted["status"].ne("")]
    features = aggregate_team_injuries(player_rows)
    empty_reports = submitted[submitted["status"].eq("")]
    if not empty_reports.empty:
        healthy = empty_reports[[*INJURY_KEY, "report_datetime"]].drop_duplicates().rename(
            columns={"report_datetime": "latest_report_datetime"}
        )
        metrics = features.columns.difference(healthy.columns)
        healthy = healthy.assign(**dict.fromkeys(metrics, 0)).reindex(columns=features.columns)
        features = pd.concat([healthy, features], ignore_index=True).drop_duplicates(INJURY_KEY, keep="last")
    player_reports = {key: rows for key, rows in submitted[PLAYER_INJURY_COLUMNS].groupby(INJURY_KEY)}
    return {key: (row, player_reports[key]) for key, row in features.set_index(INJURY_KEY).iterrows()}


def collect_injuries(
    teams: dict,
    existing: pd.DataFrame,
    player_reports: pd.DataFrame,
    now: datetime,
    indexes: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    saved = existing.set_index(INJURY_KEY)["latest_report_datetime"].to_dict()
    detail_keys = set(player_reports[PLAYER_INJURY_KEY].itertuples(index=False, name=None))
    recover = {key for key, stamp in saved.items() if (*key, stamp) not in detail_keys}
    pending = {}
    for key, (tipoff, season) in teams.items():
        if key in saved and key not in recover and key[0].date() < now.date() - timedelta(days=1):
            continue
        candidates = set(report_candidates(tipoff, now, indexes[season]))
        if key in recover and saved[key] < tipoff and saved[key] <= now:
            candidates.add(saved[key])
        pending[key] = iter(sorted(candidates, reverse=True))

    if not pending:
        return existing.iloc[:0], player_reports.iloc[:0]

    prepare_report_parser()
    selected = {}
    player_details = []
    cache = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending:
            batches = defaultdict(list)
            for key, candidates in list(pending.items()):
                stamp = next(candidates, None)
                previous = saved.get(key, datetime.min)
                if stamp is None or stamp < previous or (stamp == previous and key not in recover):
                    del pending[key]
                    continue
                batches[stamp].append(key)

            new_stamps = [stamp for stamp in batches if stamp not in cache]
            for stamp, report in zip(new_stamps, pool.map(_fetch_report_or_empty, new_stamps)):
                cache[stamp] = _report_features(report)

            for stamp, keys in batches.items():
                for key in keys:
                    report = cache[stamp].get(key)
                    if report is None:
                        continue
                    row, details = report
                    published = row["latest_report_datetime"]
                    previous = saved.get(key, datetime.min)
                    new_publication = previous < published or (previous == published and key in recover)
                    if new_publication and published < teams[key][0] and published <= now:
                        selected[key] = row
                        player_details.append(details)
                        del pending[key]

    if not selected:
        return existing.iloc[:0], player_reports.iloc[:0]
    fresh = pd.DataFrame.from_dict(selected, orient="index")
    fresh.index = pd.MultiIndex.from_tuples(fresh.index, names=INJURY_KEY)
    return fresh.reset_index().reindex(columns=existing.columns), pd.concat(player_details, ignore_index=True)


def _player_lists(reports: pd.DataFrame) -> dict:
    lists = defaultdict(set)
    for game_date, team, player_name, status, stamp in reports[PLAYER_INJURY_COLUMNS].itertuples(index=False, name=None):
        lists[(game_date, team, stamp)].add((player_name, status))
    return lists


def _changed_team_reports(
    fresh: pd.DataFrame,
    existing: pd.DataFrame,
    fresh_players: pd.DataFrame,
    saved_players: pd.DataFrame,
) -> pd.DataFrame:
    if fresh.empty or existing.empty:
        return fresh

    metrics = existing.columns.difference([*INJURY_KEY, "latest_report_datetime"])
    previous = existing.set_index(INJURY_KEY).reindex(pd.MultiIndex.from_frame(fresh[INJURY_KEY]))
    changed_metrics = fresh[metrics].ne(previous[metrics].reset_index(drop=True)).any(axis=1)
    player_lists = _player_lists(fresh_players)
    previous_lists = _player_lists(saved_players)
    changed_players = [
        player_lists[(game_date, team, stamp)] != previous_lists.get((game_date, team, earlier_stamp))
        for (game_date, team, stamp), earlier_stamp in zip(
            fresh[[*INJURY_KEY, "latest_report_datetime"]].itertuples(index=False, name=None),
            previous["latest_report_datetime"],
        )
    ]
    return fresh.loc[changed_metrics | pd.Series(changed_players, index=fresh.index)]


def refresh_injury_updates(path: Path = LIVE_INJURIES_FILE, since: date | None = None) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    player_path = path.with_name(LIVE_PLAYER_INJURIES_FILE.name)
    with FileLock(path.with_suffix(".lock"), timeout=120):
        existing = (
            pd.read_csv(path, parse_dates=["game_date", "latest_report_datetime"])
            if path.exists() else aggregate_team_injuries(pd.DataFrame())
        )
        player_reports = read_live_player_injury_reports(player_path)
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
        fresh, fresh_players = collect_injuries(teams, existing, player_reports, now, indexes)
        fresh = _changed_team_reports(fresh, existing, fresh_players, player_reports)
        updated = existing
        if not fresh.empty:
            combined = fresh if existing.empty else pd.concat([existing, fresh], ignore_index=True)
            updated = combined.drop_duplicates(INJURY_KEY, keep="last").sort_values(INJURY_KEY).reset_index(drop=True)

        saved = set(updated[INJURY_KEY].itertuples(index=False, name=None))
        yesterday = today - timedelta(days=1)
        yesterday_teams = {key for key in teams if key[0].date() == yesterday}
        if yesterday_teams and not (yesterday_teams & saved):
            raise RuntimeError(f"No official pregame injury reports were saved for {yesterday}.")

        if not fresh.empty:
            new_players = fresh_players.merge(fresh[INJURY_KEY], on=INJURY_KEY, how="inner")
            details = new_players if player_reports.empty else pd.concat([player_reports, new_players], ignore_index=True)
            detail_key = [*PLAYER_INJURY_KEY, "player_name"]
            details = details.drop_duplicates(detail_key, keep="last").sort_values(detail_key)
            # Publish detail first so the previous aggregate timestamp stays readable if interrupted.
            write_snapshot(details, player_path)
            write_snapshot(updated, path)

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
