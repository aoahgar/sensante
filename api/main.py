"""API REST FastAPI pour SenSante."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field, field_validator


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
ENCODER_SEXE_PATH = MODELS_DIR / "encoder_sexe.pkl"
ENCODER_REGION_PATH = MODELS_DIR / "encoder_region.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.pkl"


app = FastAPI(
    title="SenSante API",
    description="API de pre-diagnostic medical avec explication LLM via Groq.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv(ROOT / ".env")
GROQ_MODEL = "llama-3.1-8b-instant"
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera en mode degrade.")


class PatientInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(..., min_length=1)
    temperature: float = Field(..., ge=30.0, le=45.0)
    tension_sys: int = Field(..., ge=50, le=250)
    toux: bool
    fatigue: bool
    maux_tete: bool
    frissons: bool
    nausee: bool
    region: str = Field(..., min_length=1)

    @field_validator("sexe", "region")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("sexe")
    @classmethod
    def normalize_sexe(cls, value: str) -> str:
        return value.upper()


class PredictionResponse(BaseModel):
    diagnostic: str
    confidence: float
    probabilities: dict[str, float]
    features_used: list[str]
    input: PatientInput


class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., ge=0, le=1, description="Probabilite du diagnostic")
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(..., min_length=1)
    temperature: float = Field(..., ge=30.0, le=45.0)
    region: str = Field(..., min_length=1)


class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en francais")
    modele_llm: str = Field(default=GROQ_MODEL, description="Modele LLM utilise")


SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple, comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    missing_files = [
        str(path.relative_to(ROOT))
        for path in [
            MODEL_PATH,
            ENCODER_SEXE_PATH,
            ENCODER_REGION_PATH,
            FEATURE_COLS_PATH,
        ]
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Fichiers modele manquants : " + ", ".join(missing_files)
        )

    return {
        "model": joblib.load(MODEL_PATH),
        "encoder_sexe": joblib.load(ENCODER_SEXE_PATH),
        "encoder_region": joblib.load(ENCODER_REGION_PATH),
        "feature_cols": joblib.load(FEATURE_COLS_PATH),
    }


def _classes(encoder: Any) -> list[str]:
    return [str(value) for value in encoder.classes_]


def build_feature_frame(patient: PatientInput, artifacts: dict[str, Any]) -> pd.DataFrame:
    encoder_sexe = artifacts["encoder_sexe"]
    encoder_region = artifacts["encoder_region"]
    feature_cols = artifacts["feature_cols"]

    allowed_sexes = _classes(encoder_sexe)
    allowed_regions = _classes(encoder_region)

    if patient.sexe not in allowed_sexes:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Sexe inconnu pour le modele.",
                "received": patient.sexe,
                "allowed_values": allowed_sexes,
            },
        )

    if patient.region not in allowed_regions:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Region inconnue pour le modele.",
                "received": patient.region,
                "allowed_values": allowed_regions,
            },
        )

    encoded_patient = {
        "age": patient.age,
        "sexe_encoded": int(encoder_sexe.transform([patient.sexe])[0]),
        "temperature": patient.temperature,
        "tension_sys": patient.tension_sys,
        "toux": int(patient.toux),
        "fatigue": int(patient.fatigue),
        "maux_tete": int(patient.maux_tete),
        "frissons": int(patient.frissons),
        "nausee": int(patient.nausee),
        "region_encoded": int(encoder_region.transform([patient.region])[0]),
    }

    return pd.DataFrame(
        [[encoded_patient[col] for col in feature_cols]],
        columns=feature_cols,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Bienvenue sur l'API SenSante.",
        "health": "/health",
        "predict": "/predict",
        "explain": "/explain",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        artifacts = load_artifacts()
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "model_loaded": False,
            "detail": str(exc),
        }

    model = artifacts["model"]
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "classes": _classes(model),
        "features": artifacts["feature_cols"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput) -> PredictionResponse:
    try:
        artifacts = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    model = artifacts["model"]
    feature_frame = build_feature_frame(patient, artifacts)
    diagnostic = str(model.predict(feature_frame)[0])
    probabilities_array = model.predict_proba(feature_frame)[0]
    probabilities = {
        str(classe): round(float(proba), 4)
        for classe, proba in zip(model.classes_, probabilities_array)
    }

    return PredictionResponse(
        diagnostic=diagnostic,
        confidence=max(probabilities.values()),
        probabilities=probabilities,
        features_used=list(feature_frame.columns),
        input=patient,
    )


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput) -> ExplainOutput:
    """Expliquer un diagnostic en francais avec un LLM Groq."""
    if not groq_client:
        return ExplainOutput(
            explication=(
                "Service d'explication indisponible. Cle API Groq non configuree. "
                "Le diagnostic reste disponible, mais l'explication LLM doit etre "
                "activee avec GROQ_API_KEY dans le fichier .env."
            ),
            modele_llm="aucun",
        )

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        "Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        explication = response.choices[0].message.content or ""
    except Exception as exc:
        explication = f"Erreur lors de l'appel au LLM : {exc}"

    return ExplainOutput(explication=explication, modele_llm=GROQ_MODEL)
