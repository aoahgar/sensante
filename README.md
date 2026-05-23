---
title: SenSante
sdk: docker
app_port: 8000
pinned: false
---

# SenSante

Assistant de pre-diagnostic medical pour le Senegal.

## Demo en ligne

URL Hugging Face Spaces cible : `https://aoahgar-sensante.hf.space`

Etat : pret pour deploiement Docker. Le remote Hugging Face doit etre ajoute
apres creation du Space public `aoahgar/sensante`.

## Description

SenSante utilise un modele Machine Learning pour proposer un pre-diagnostic
medical a partir des symptomes d'un patient. Le Lab 5 ajoute une explication
en francais simple avec Llama 3 via l'API Groq.

## Stack

- scikit-learn : modele ML RandomForest
- FastAPI : API REST et serveur du frontend
- Tailwind CSS : interface responsive
- Groq / Llama 3 : explication LLM
- Docker : conteneurisation et deploiement Hugging Face Spaces

## Structure

- `data/` : donnees patients
- `models/` : modele ML et encodeurs serialises
- `api/` : API FastAPI
- `frontend/` : interface web Tailwind
- `notebooks/` : scripts de test et d'entrainement
- `tests/` : tests automatises

## Lancer l'API

```bash
venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Endpoints principaux :

- Frontend : `http://127.0.0.1:8000`
- Info API : `GET /api`
- `GET /health`
- `POST /predict`
- `POST /explain`
- Swagger UI : `http://127.0.0.1:8000/docs`

## Configurer Groq

Creer un fichier `.env` a la racine du projet :

```env
GROQ_API_KEY=gsk_votre_cle_ici
```

Le fichier `.env` est deja ignore par Git. Sans cette cle, `/explain` renvoie
un message propre indiquant que le service LLM n'est pas configure.

## Tester Groq

```bash
venv\Scripts\python.exe notebooks\test_groq.py
```

## Lancer le frontend

```bash
venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Puis ouvrir `http://127.0.0.1:8000`.

## Docker

Construire l'image :

```bash
docker build -t sensante .
```

Lancer le conteneur :

```bash
docker run --rm -p 8000:8000 --env-file .env sensante
```

Tester :

- Frontend : `http://127.0.0.1:8000`
- Documentation API : `http://127.0.0.1:8000/docs`
- Sante API : `http://127.0.0.1:8000/health`

## Tests

```bash
venv\Scripts\python.exe -m pytest tests\test_api.py
```

## Auteur

Ousman Ali Haggar - L2 GLSI C - ESP / UCAD

## Cours

Integration de Modeles IA - Dr. El Hadji Bassirou TOURE
