# NBA Odds Predictor Browser App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing NBA Elo project into a polished multi-page browser app with a shared top hotbar, a homepage that explains the project and Bayesian Elo approach, a live leaderboard page, and a scraping page for importing NBA games.

**Architecture:** Use a lightweight Flask app with Jinja templates and static CSS to keep the UI simple, navigable, and resume-ready. Keep the current data/model pipeline in Python, then expose it through three focused web routes that all share one base layout and top navigation bar.

**Tech Stack:** Python, Flask, Jinja2 templates, pandas, requests, BeautifulSoup4, CSS.

---

### Task 1: Add the web app skeleton and shared layout

**Files:**
- Create: `app.py`
- Create: `templates/base.html`
- Create: `templates/home.html`
- Create: `static/styles.css`

- [ ] **Step 1: Write the failing test**

Create a simple route smoke-check plan by adding a minimal `app.py` that imports Flask and defines `create_app()`, but do not wire any templates yet.

- [ ] **Step 2: Run test to verify it fails**

Run: `py -c "from app import create_app; app = create_app(); print(app.url_map)"`
Expected: fail because `app.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from flask import Flask, render_template


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template("home.html")

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
```

Add `base.html` with a top hotbar that includes links to `Home`, `Leaderboard`, and `Scrape`, and a content block for page-specific content.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -c "from app import create_app; app = create_app(); print(sorted(rule.rule for rule in app.url_map.iter_rules()))"`
Expected: prints `/` plus Flask static routes.

- [ ] **Step 5: Commit**

```bash
git add app.py templates/base.html templates/home.html static/styles.css
git commit -m "feat: add flask app shell and shared layout"
```

### Task 2: Build the homepage content and Bayesian Elo explanation

**Files:**
- Modify: `templates/home.html`
- Modify: `static/styles.css`

- [ ] **Step 1: Write the failing test**

Add a homepage rendering expectation to the app smoke check by ensuring the home route returns HTML containing the project title and a Bayesian-Elo section heading.

```python
from app import create_app


def test_homepage_contains_project_sections():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "ESPN Odds Predictor" in body
    assert "Bayesian Elo" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_app.py -v`
Expected: fail because the test file and homepage content are not in place yet.

- [ ] **Step 3: Write minimal implementation**

Build the homepage as a polished landing page that:

- introduces the project as an NBA game scraper + Elo pipeline
- explains the data flow from Basketball-Reference to processed CSVs to ratings
- includes a plain-language Bayesian-Elo explanation
- highlights the model inputs: home-court advantage, margin-of-victory weighting, and season regression

Use this content structure in `home.html`:

```html
{% extends "base.html" %}
{% block content %}
<section class="hero">
  <h1>ESPN Odds Predictor</h1>
  <p>NBA game scraper and Elo rating pipeline built from Basketball-Reference data.</p>
</section>
<section class="card">
  <h2>What it does</h2>
  <ul>
    <li>Scrapes regular-season game results</li>
    <li>Cleans and normalizes team names</li>
    <li>Runs an Elo model with home-court advantage</li>
    <li>Saves reproducible CSV outputs for analysis</li>
  </ul>
</section>
<section class="card">
  <h2>Bayesian Elo</h2>
  <p>
    The model starts each team near a shared prior rating, then updates ratings after every game
    based on how surprising the result was. If a strong team wins as expected, the update is small.
    If an underdog wins, the posterior estimate shifts more aggressively. That gives the ratings a
    Bayesian feel: prior beliefs are adjusted by game evidence instead of being replaced all at once.
  </p>
</section>
{% endblock %}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_app.py::test_homepage_contains_project_sections -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add templates/home.html static/styles.css tests/test_app.py
git commit -m "feat: add homepage content and elo explanation"
```

### Task 3: Add the live leaderboard page

**Files:**
- Modify: `app.py`
- Create: `templates/leaderboard.html`
- Modify: `model.py`
- Modify: `main.py`

- [ ] **Step 1: Write the failing test**

Add a route test that checks the leaderboard page renders and contains a table of team ratings.

```python
from app import create_app


def test_leaderboard_route_renders():
    app = create_app()
    client = app.test_client()
    response = client.get("/leaderboard")
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Leaderboard" in body
    assert "Team" in body
    assert "Rating" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_app.py::test_leaderboard_route_renders -v`
Expected: fail because the route/template do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Expose a helper that loads the latest processed games, runs `run_elo`, and returns a current standings table. Route `/leaderboard` to render the standings table, sorted by rating descending.

Template structure:

```html
{% extends "base.html" %}
{% block content %}
<section class="page-header">
  <h1>Leaderboard</h1>
  <p>Live Elo standings from the latest processed dataset.</p>
</section>
<table>
  <thead>
    <tr><th>Team</th><th>Rating</th></tr>
  </thead>
  <tbody>
    {% for row in standings %}
    <tr>
      <td>{{ row.team }}</td>
      <td>{{ "%.1f"|format(row.rating) }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_app.py::test_leaderboard_route_renders -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py templates/leaderboard.html model.py main.py tests/test_app.py
git commit -m "feat: add live leaderboard page"
```

### Task 4: Add the scraping page and wire the import flow

**Files:**
- Modify: `app.py`
- Create: `templates/scrape.html`
- Modify: `scraper.py`
- Modify: `cleaner.py`
- Modify: `main.py`

- [ ] **Step 1: Write the failing test**

Add a route test that posts a season range to `/scrape` and checks for a success message or an error message instead of a crash.

```python
from app import create_app


def test_scrape_page_accepts_post():
    app = create_app()
    client = app.test_client()
    response = client.post("/scrape", data={"start_season": "2016", "end_season": "2016"})
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Scrape" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_app.py::test_scrape_page_accepts_post -v`
Expected: fail because the route does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create a `/scrape` page with a form for `start_season` and `end_season`. On submit, call the existing season scraper and then run the clean + normalize pipeline so the current data stays ready for the leaderboard page.

The page should:

- accept a season range
- show scrape progress or a completion message
- save raw and processed CSVs to the existing data folders
- fail gracefully with a human-readable error message if scraping fails

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_app.py::test_scrape_page_accepts_post -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py templates/scrape.html scraper.py cleaner.py main.py tests/test_app.py
git commit -m "feat: add scrape page and import workflow"
```

### Task 5: Trim legacy CLI paths and document the web app

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `main.py`
- Modify: `monthly_scraper.py`
- Modify: `updater.py`

- [ ] **Step 1: Write the failing test**

Add a smoke test that confirms the old monthly/daily update scripts are no longer the primary entrypoint and that the README points users to the browser app.

```python
def test_readme_mentions_web_app():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "browser app" in text.lower()
    assert "Flask" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_docs.py -v`
Expected: fail until the README is updated.

- [ ] **Step 3: Write minimal implementation**

Update the README to emphasize:

- the browser app entrypoint
- the three pages in the hotbar
- how the scraper and leaderboard connect

Keep `monthly_scraper.py` and `updater.py` in the repo for now, but mark them as legacy/manual tools in the README so they are not the main story.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_docs.py::test_readme_mentions_web_app -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt main.py monthly_scraper.py updater.py tests/test_docs.py
git commit -m "docs: position project around browser app"
```

### Task 6: Verify the app end to end

**Files:**
- Verify: `app.py`
- Verify: `templates/base.html`
- Verify: `templates/home.html`
- Verify: `templates/leaderboard.html`
- Verify: `templates/scrape.html`

- [ ] **Step 1: Run the app locally**

Run: `py app.py`
Expected: Flask starts successfully on the local development server.

- [ ] **Step 2: Exercise each page**

Open the home page, click leaderboard, click scrape, and confirm the hotbar works from every page.

- [ ] **Step 3: Confirm visual polish**

Check that the homepage reads like a project showcase instead of plain documentation, with strong section spacing, readable typography, and a clear hierarchy.

- [ ] **Step 4: Final sanity check**

Confirm the leaderboard can render from the latest saved Elo data and the scrape page can kick off a new import without breaking navigation.

