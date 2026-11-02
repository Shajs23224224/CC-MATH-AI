"""Static frontend delivery tests for the Training Dashboard TD04 UI."""

from fastapi.testclient import TestClient

from apps.dashboard.api import create_app
from core.observability.training import TrainingEventRecorder


def test_dashboard_root_redirects_to_frontend() -> None:
    client = TestClient(create_app(TrainingEventRecorder()), follow_redirects=False)

    response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard/"


def test_dashboard_serves_html_and_local_assets_from_same_origin() -> None:
    client = TestClient(create_app(TrainingEventRecorder()))

    page = client.get("/dashboard/")
    stylesheet = client.get("/dashboard/styles.css")
    script = client.get("/dashboard/app.js")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "Training overview" in page.text
    assert 'href="/dashboard/styles.css"' in page.text
    assert 'src="/dashboard/app.js"' in page.text

    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "--accent: #65e0b3" in stylesheet.text

    assert script.status_code == 200
    assert "text/javascript" in script.headers["content-type"]
    assert '"/api/v1"' in script.text
    assert "new EventSource" in script.text


def test_dashboard_assets_are_served_without_external_frontend_cdn() -> None:
    client = TestClient(create_app(TrainingEventRecorder()))

    page = client.get("/dashboard/").text

    assert "https://" not in page
    assert "http://" not in page
