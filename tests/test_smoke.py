from pathlib import Path

from fastapi.testclient import TestClient


def _prepare_env(tmp_path, monkeypatch):
    api_txt: Path = tmp_path / "api.txt"
    api_txt.write_text("AITUNNEL_KEY=test\nGEMINI_API_KEY=test\n", encoding="utf-8")
    monkeypatch.setenv("API_TXT_PATH", str(api_txt))


def test_health_smoke(tmp_path, monkeypatch):
    _prepare_env(tmp_path, monkeypatch)

    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.main import create_app

    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"
