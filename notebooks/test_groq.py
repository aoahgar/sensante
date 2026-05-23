"""Test simple de l'API Groq avec Llama 3 pour SenSante."""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


ROOT = Path(__file__).resolve().parents[1]
MODEL = "llama-3.1-8b-instant"


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("ERREUR : GROQ_API_KEY non trouvee dans .env")
        print("Creez le fichier .env avec : GROQ_API_KEY=gsk_votre_cle_ici")
        return

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant medical senegalais. "
                    "Reponds en francais simple. Maximum 3 phrases."
                ),
            },
            {
                "role": "user",
                "content": "Quels sont les symptomes du paludisme ?",
            },
        ],
        max_tokens=200,
        temperature=0.3,
    )

    print("=== Reponse de Llama 3 ===")
    print(response.choices[0].message.content)
    print(f"\nTokens utilises : {response.usage.total_tokens}")

    response2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant medical senegalais. "
                    "Tu recois un diagnostic et des donnees patient. "
                    "Explique le resultat en francais simple, comme un medecin "
                    "parlerait a son patient. Sois rassurant mais recommande une "
                    "consultation. Maximum 3 phrases. Ne fais JAMAIS de diagnostic "
                    "toi-meme."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Patient : Femme, 28 ans, region Dakar\n"
                    "Symptomes : temperature 39.5, toux, fatigue, maux de tete\n"
                    "Diagnostic du modele : paludisme (probabilite 72%)\n"
                    "Explique ce resultat au patient."
                ),
            },
        ],
        max_tokens=200,
        temperature=0.3,
    )

    print("\n=== Explication SenSante ===")
    print(response2.choices[0].message.content)


if __name__ == "__main__":
    main()
