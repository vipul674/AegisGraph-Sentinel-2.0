"""Regression tests for production-readiness hardening."""

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.api import main as api_main
from src.api.main import state


def _transaction(transaction_id="txn_001", amount=100.0):
    return {
        "transaction_id": transaction_id,
        "source_account": "acct_src",
        "target_account": "acct_dst",
        "amount": amount,
        "currency": "INR",
        "mode": "UPI",
        "timestamp": "2026-02-26T14:30:00Z",
    }


def test_health_smoke(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_stats_smoke(api_client):
    response = api_client.get("/stats")
    assert response.status_code == 200
    assert "total_requests" in response.json()


def test_missing_amount_returns_json_validation_error(api_client):
    payload = _transaction()
    payload.pop("amount")

    response = api_client.post("/api/v1/fraud/check", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "validation_errors" in body["error"]["details"]


def test_invalid_payload_returns_json_validation_error(api_client):
    response = api_client.post("/api/v1/fraud/check", json={"amount": "bad"})

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "ValidationException"


def test_batch_overflow_rejected(api_client):
    transactions = [_transaction(f"txn_{i}") for i in range(101)]

    response = api_client.post("/api/v1/fraud/batch", json={"transactions": transactions})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_missing_graph_artifact_does_not_crash(api_client):
    assert not Path("data/synthetic/graph.graphml").exists()
    assert not Path("data/synthetic/graph.gpickle").exists()

    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json()["graph_loaded"] is False
    assert state.graph_loaded is False


def test_validation_error_payload_is_json_safe(api_client):
    payload = _transaction()
    payload["amount"] = -1

    response = api_client.post("/api/v1/fraud/check", json=payload)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["details"]["validation_errors"]


def test_lateral_movement_initializes_even_when_other_innovations_are_unavailable(monkeypatch):
    dummy_detector = object()
    startup_logger = Mock()
    register_service = Mock()

    monkeypatch.setattr(api_main, "INNOVATIONS_AVAILABLE", False)
    monkeypatch.setattr(api_main, "LATERAL_MOVEMENT_AVAILABLE", True)
    monkeypatch.setattr(api_main, "LateralMovementDetector", lambda: dummy_detector)
    monkeypatch.setattr(api_main.state.services, "register_service", register_service)
    monkeypatch.setattr(api_main.state, "lateral_movement_detector", None, raising=False)

    api_main._initialize_innovation_runtime(startup_logger)

    assert api_main.state.lateral_movement_detector is dummy_detector
    register_service.assert_called_once_with("lateral_movement_detector", dummy_detector, replace=True)


@pytest.mark.parametrize(
    "base_score,lateral_boost,expected_decision",
    [
        (0.25, 0.35, "REVIEW"),
        (0.45, 0.35, "BLOCK"),
    ],
)
def test_scoring_applies_lateral_movement_even_when_innovations_flag_is_false(
    monkeypatch,
    base_score,
    lateral_boost,
    expected_decision,
):
    detector = Mock()
    detector.analyze_account.return_value = (lateral_boost, True)

    monkeypatch.setattr(
        api_main,
        "compute_risk_score",
        lambda transaction, biometrics=None, **kwargs: {
            "risk_score": base_score,
            "decision": "ALLOW",
            "confidence": 0.85,
            "breakdown": {"graph": 0.0, "velocity": 0.0, "behavior": 0.0, "entropy": 0.0},
        },
    )

    result = api_main._run_scoring_pipeline(
        transaction={"transaction_id": "txn_lateral_001"},
        biometrics=None,
        source_account="acct_src",
        target_account="acct_dst",
        lateral_detector=detector,
        innovations_available=False,
    )

    detector.update_graph.assert_called_once_with("acct_src", "acct_dst")
    detector.analyze_account.assert_called_once_with("acct_src")
    assert result["risk_score"] == pytest.approx(min(1.0, base_score + lateral_boost))
    assert result["breakdown"]["lateral_movement"] == lateral_boost
    assert result["lateral_movement_detected"] is True
    assert result["decision"] == expected_decision


def test_scoring_continues_when_lateral_detector_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "compute_risk_score",
        lambda transaction, biometrics=None, **kwargs: {
            "risk_score": 0.2,
            "decision": "ALLOW",
            "confidence": 0.85,
            "breakdown": {"graph": 0.0, "velocity": 0.0, "behavior": 0.0, "entropy": 0.0},
        },
    )

    result = api_main._run_scoring_pipeline(
        transaction={"transaction_id": "txn_lateral_none"},
        biometrics=None,
        source_account="acct_src",
        target_account="acct_dst",
        lateral_detector=None,
        innovations_available=False,
    )

    assert result["risk_score"] == pytest.approx(0.2)
    assert result["decision"] == "ALLOW"
    assert "lateral_movement" not in result["breakdown"]


def test_scoring_recovers_when_lateral_analysis_raises(monkeypatch):
    class RaisingDetector:
        def update_graph(self, source_account, target_account):
            raise RuntimeError("centrality backend unavailable")

        def analyze_account(self, source_account):
            raise RuntimeError("should not be reached")

    monkeypatch.setattr(
        api_main,
        "compute_risk_score",
        lambda transaction, biometrics=None, **kwargs: {
            "risk_score": 0.33,
            "decision": "ALLOW",
            "confidence": 0.85,
            "breakdown": {"graph": 0.0, "velocity": 0.0, "behavior": 0.0, "entropy": 0.0},
        },
    )

    result = api_main._run_scoring_pipeline(
        transaction={"transaction_id": "txn_lateral_error"},
        biometrics=None,
        source_account="acct_src",
        target_account="acct_dst",
        lateral_detector=RaisingDetector(),
        innovations_available=False,
    )

    assert result["risk_score"] == pytest.approx(0.33)
    assert result["decision"] == "ALLOW"
    assert "lateral_movement" not in result["breakdown"]


def test_startup_disk_reads_use_thread_pool(monkeypatch, tmp_path):
    class DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    graph_path = tmp_path / "graph.graphml"
    graph_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph edgedefault="directed" id="G">
    <node id="n0" />
  </graph>
</graphml>
""",
        encoding="utf-8",
    )
    graph_sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()

    chains_path = tmp_path / "fraud_chains.json"
    chains_path.write_text(json.dumps([{"accounts": ["mule_1", "mule_2"]}]), encoding="utf-8")
    accounts_path = tmp_path / "accounts.json"
    accounts_path.write_text(json.dumps([{"account_id": "acct_1", "score": 0.5}]), encoding="utf-8")

    original_graph_path = state.settings.graph.graph_path
    original_graph_sha = state.settings.graph.graph_sha256
    original_graph_loaded = state.graph_loaded
    original_transaction_graph = state.transaction_graph
    original_fraud_chains = state.fraud_chains
    original_account_profiles = state.account_profiles
    original_mule_accounts = set(state.mule_accounts)
    original_path = api_main.Path
    original_to_thread = api_main.asyncio.to_thread
    call_names = []

    async def recording_to_thread(func, *args, **kwargs):
        call_names.append(func.__name__)
        return await original_to_thread(func, *args, **kwargs)

    def fake_path(value):
        if value == "data/synthetic/fraud_chains.json":
            return chains_path
        if value == "data/synthetic/accounts.json":
            return accounts_path
        return Path(value)

    monkeypatch.setattr(api_main.asyncio, "to_thread", recording_to_thread)
    monkeypatch.setattr(api_main, "Path", fake_path)
    state.settings.graph.graph_path = graph_path
    state.settings.graph.graph_sha256 = graph_sha

    try:
        asyncio.run(api_main._load_graph_runtime_data(DummyLogger()))

        assert call_names == ["_read_file_bytes", "_read_json_file", "_read_json_file"]
        assert state.graph_loaded is True
        assert state.fraud_chains[0]["accounts"] == ["mule_1", "mule_2"]
        assert state.account_profiles["acct_1"]["score"] == 0.5
    finally:
        state.settings.graph.graph_path = original_graph_path
        state.settings.graph.graph_sha256 = original_graph_sha
        state.graph_loaded = original_graph_loaded
        state.transaction_graph = original_transaction_graph
        state.fraud_chains = original_fraud_chains
        state.account_profiles = original_account_profiles
        state.mule_accounts.clear()
        state.mule_accounts.update(original_mule_accounts)
        monkeypatch.setattr(api_main, "Path", original_path)


class _BoomOracle:
    def generate_explanation(self, *args, **kwargs):
        raise RuntimeError("oracle internal secret")


class _BoomVoiceAnalyzer:
    def analyze_voice(self, *args, **kwargs):
        raise RuntimeError("voice internal secret")


class _BoomMuleScorer:
    def score_account_opening(self, *args, **kwargs):
        raise RuntimeError("scoring internal secret")


@pytest.mark.parametrize(
    ("path", "payload", "attr", "stub", "secret"),
    [
        (
            "/api/v1/explain",
            {
                "decision": "ALLOW",
                "risk_score": 0.2,
            },
            "aegis_oracle",
            _BoomOracle(),
            "oracle internal secret",
        ),
        (
            "/api/v1/voice/analyze",
            {
                "transaction_id": "txn_voice",
                "audio_base64": "dGVzdA==",
                "sample_rate": 16000,
            },
            "voice_analyzer",
            _BoomVoiceAnalyzer(),
            "voice internal secret",
        ),
        (
            "/api/v1/accounts/score-opening",
            {
                "account_id": "acct_1",
                "name": "Test User",
                "age": 30,
                "profession": "Engineer",
                "email": "user@example.com",
                "phone": "9999999999",
                "device_id": "device-1",
                "ip_address": "127.0.0.1",
                "stated_address": "Test Address",
                "facial_match": 0.9,
                "document_type": "PAN",
                "initial_deposit": 1000.0,
            },
            "mule_scorer",
            _BoomMuleScorer(),
            "scoring internal secret",
        ),
    ],
)
def test_public_api_internal_errors_are_sanitized(
    api_client,
    monkeypatch,
    path,
    payload,
    attr,
    stub,
    secret,
):
    monkeypatch.setattr(api_main, "INNOVATIONS_AVAILABLE", True)
    monkeypatch.setattr(api_main.state, attr, stub, raising=False)

    response = api_client.post(path, json=payload)

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "Internal Server Error"
    assert secret not in response.text
