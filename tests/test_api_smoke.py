from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_search_endpoint_returns_results(client):
    resp = client.post("/api/search", json={"query": "테스트", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "session" in data


@pytest.mark.smoke
def test_feedback_endpoint_accepts_actions(client):
    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "test",
            "action": "like",
            "ext": ".pdf",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
