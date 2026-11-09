from hashlib import sha256
from pathlib import Path

from app import create_app


def test_predictor_links_static_assets_by_content():
    app = create_app()
    client = app.test_client()
    page = client.get("/predictor")
    assert page.status_code == 200

    for filename in ("styles.css", "app.js"):
        asset = (Path(app.static_folder) / filename).read_bytes()
        version = sha256(asset).hexdigest()[:12]
        url = f"/static/{filename}?v={version}"
        assert url in page.get_data(as_text=True)
        assert client.get(url).data == asset
