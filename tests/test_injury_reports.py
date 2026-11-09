from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import injury_reports


def test_index_orders_reports_and_filters_the_requested_season(monkeypatch):
    html = """
        <a href="/referee/injury/Injury-Report_2026-11-08_07_15PM.pdf">latest</a>
        <a href="/referee/injury/Injury-Report_2025-10-21_06PM.pdf">previous season</a>
        <a href="/referee/injury/Injury-Report_2026-11-08_12_00AM.pdf">midnight</a>
        <a href="/referee/injury/Injury-Report_2026-11-08_07_15PM.pdf">duplicate</a>
    """
    response = SimpleNamespace(text=html, raise_for_status=lambda: None)
    monkeypatch.setattr(injury_reports, "_get", lambda url: response)

    assert injury_reports.fetch_report_index(2027) == [
        datetime(2026, 11, 8),
        datetime(2026, 11, 8, 19, 15),
    ]
    assert injury_reports.fetch_report_index(2026) == [datetime(2025, 10, 21, 18)]


def test_legacy_pdf_uses_header_time_and_preserves_unsubmitted_row(monkeypatch):
    timestamp = datetime(2025, 10, 21, 18)
    response = SimpleNamespace(status_code=200, content=b"%PDF-", raise_for_status=lambda: None)
    monkeypatch.setattr(injury_reports, "_get", lambda url: response)
    page = SimpleNamespace(extract_text=lambda: "Injury\nReport:\n10/21/25\n06:30\nPM\n")
    monkeypatch.setattr(injury_reports, "PdfReader", lambda data: SimpleNamespace(pages=[page]))
    pdf_table = pd.DataFrame(
        [["10/21/2025", "07:30 (ET)", "LAC@GSW", "LA Clippers", None, None, "NOT YET SUBMITTED"]],
        columns=injury_reports.REPORT_FIELDS,
    )
    monkeypatch.setattr(injury_reports, "_parse_pdf", lambda *args: pdf_table)

    report = injury_reports.fetch_report(timestamp).iloc[0]

    assert report["game_date"] == pd.Timestamp("2025-10-21")
    assert report["report_datetime"] == datetime(2025, 10, 21, 18, 30)
    assert report["team"] == "LAC"
    assert report["player_name"] == report["status"] == ""
    assert report["reason"] == "NOT YET SUBMITTED"


def test_cdn_missing_reports_raise_report_unavailable(monkeypatch):
    monkeypatch.setattr(injury_reports, "_get", lambda url: SimpleNamespace(status_code=403))

    with pytest.raises(injury_reports.ReportUnavailable, match="HTTP 403"):
        injury_reports.fetch_report(datetime(2026, 11, 8, 23, 59))
