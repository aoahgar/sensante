from fastapi.testclient import TestClient

import api.main as main
from api.main import app


client = TestClient(app)


def test_health_loads_model():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert "model.pkl" in payload["model_path"]
    assert payload["classes"]
    assert payload["features"]


def test_predict_returns_diagnostic_and_probabilities():
    patient = {
        "age": 28,
        "sexe": "F",
        "temperature": 39.5,
        "tension_sys": 110,
        "toux": True,
        "fatigue": True,
        "maux_tete": True,
        "frissons": True,
        "nausee": False,
        "region": "Dakar",
    }

    response = client.post("/predict", json=patient)

    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnostic"]
    assert 0 <= payload["confidence"] <= 1
    assert set(payload["probabilities"]) == {"grippe", "paludisme", "sain", "typhoide"}
    assert payload["features_used"] == [
        "age",
        "sexe_encoded",
        "temperature",
        "tension_sys",
        "toux",
        "fatigue",
        "maux_tete",
        "frissons",
        "nausee",
        "region_encoded",
    ]


def test_predict_rejects_unknown_region():
    patient = {
        "age": 28,
        "sexe": "F",
        "temperature": 39.5,
        "tension_sys": 110,
        "toux": True,
        "fatigue": True,
        "maux_tete": True,
        "frissons": True,
        "nausee": False,
        "region": "Region inconnue",
    }

    response = client.post("/predict", json=patient)

    assert response.status_code == 422
    assert response.json()["detail"]["message"] == "Region inconnue pour le modele."


def test_explain_degrades_cleanly_without_groq_key(monkeypatch):
    monkeypatch.setattr(main, "groq_client", None)
    payload = {
        "diagnostic": "paludisme",
        "probabilite": 0.72,
        "age": 28,
        "sexe": "F",
        "temperature": 39.5,
        "region": "Dakar",
    }

    response = client.post("/explain", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "Cle API Groq non configuree" in body["explication"]
    assert body["modele_llm"] == "aucun"
