"""
SenSante - Entrainement et serialisation du modele ML
Lab 2 : Entrainement RandomForest + model.pkl
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "patients_dakar.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("SENSANTE - Lab 2 : entrainer et serialiser un modele")
    print("=" * 60)

    # ===== CHARGER LES DONNEES =====
    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
    print(f"Colonnes : {list(df.columns)}")
    print("\nDiagnostics :")
    print(df["diagnostic"].value_counts())

    # ===== PREPARER LES FEATURES =====
    le_sexe = LabelEncoder()
    le_region = LabelEncoder()
    df["sexe_encoded"] = le_sexe.fit_transform(df["sexe"])
    df["region_encoded"] = le_region.fit_transform(df["region"])

    feature_cols = [
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
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df["diagnostic"]
    print(f"\nFeatures : {X.shape}")
    print(f"Cible : {y.shape}")

    # ===== TRAIN / TEST =====
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nEntrainement : {X_train.shape[0]} patients")
    print(f"Test : {X_test.shape[0]} patients")

    # ===== ENTRAINER LE MODELE =====
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("\nModele entraine !")
    print(f"Nombre d'arbres : {model.n_estimators}")
    print(f"Nombre de features : {model.n_features_in_}")
    print(f"Classes : {list(model.classes_)}")

    # ===== EVALUER =====
    y_pred = model.predict(X_test)
    comparison = pd.DataFrame(
        {
            "Vrai diagnostic": y_test.values[:10],
            "Prediction": y_pred[:10],
        }
    )
    print("\n--- Comparaison des 10 premieres predictions ---")
    print(comparison)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {accuracy:.2%}")

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print("\nMatrice de confusion :")
    print(cm)
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    plt.xlabel("Prediction du modele")
    plt.ylabel("Vrai diagnostic")
    plt.title("Matrice de confusion - SenSante")
    plt.tight_layout()
    confusion_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=150)
    plt.close()
    print(f"\nFigure sauvegardee : {confusion_path.relative_to(ROOT)}")

    # ===== SERIALISER =====
    model_path = MODELS_DIR / "model.pkl"
    encoder_sexe_path = MODELS_DIR / "encoder_sexe.pkl"
    encoder_region_path = MODELS_DIR / "encoder_region.pkl"
    feature_cols_path = MODELS_DIR / "feature_cols.pkl"

    joblib.dump(model, model_path)
    joblib.dump(le_sexe, encoder_sexe_path)
    joblib.dump(le_region, encoder_region_path)
    joblib.dump(feature_cols, feature_cols_path)

    size = model_path.stat().st_size
    print(f"\nModele sauvegarde : {model_path.relative_to(ROOT)}")
    print(f"Taille : {size / 1024:.1f} Ko")
    print("Encodeurs et metadata sauvegardes.")

    # ===== TESTER LE MODELE SERIALISE =====
    model_loaded = joblib.load(model_path)
    le_sexe_loaded = joblib.load(encoder_sexe_path)
    le_region_loaded = joblib.load(encoder_region_path)
    feature_cols_loaded = joblib.load(feature_cols_path)

    print(f"\nModele recharge : {type(model_loaded).__name__}")
    print(f"Classes : {list(model_loaded.classes_)}")

    nouveau_patient = {
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

    patient_encoded = {
        "age": nouveau_patient["age"],
        "sexe_encoded": le_sexe_loaded.transform([nouveau_patient["sexe"]])[0],
        "temperature": nouveau_patient["temperature"],
        "tension_sys": nouveau_patient["tension_sys"],
        "toux": int(nouveau_patient["toux"]),
        "fatigue": int(nouveau_patient["fatigue"]),
        "maux_tete": int(nouveau_patient["maux_tete"]),
        "frissons": int(nouveau_patient["frissons"]),
        "nausee": int(nouveau_patient["nausee"]),
        "region_encoded": le_region_loaded.transform([nouveau_patient["region"]])[0],
    }
    features = pd.DataFrame(
        [[patient_encoded[col] for col in feature_cols_loaded]],
        columns=feature_cols_loaded,
    )

    diagnostic = model_loaded.predict(features)[0]
    probas = model_loaded.predict_proba(features)[0]
    proba_max = probas.max()

    print("\n--- Resultat du pre-diagnostic ---")
    print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
    print(f"Diagnostic : {diagnostic}")
    print(f"Probabilite : {proba_max:.1%}")
    print("\nProbabilites par classe :")
    for classe, proba in zip(model_loaded.classes_, probas):
        bar = "#" * int(proba * 30)
        print(f"{classe:10s} : {proba:.1%} {bar}")

    print("\n" + "=" * 60)
    print("Lab 2 termine : modele entraine, evalue, serialise et teste.")
    print("=" * 60)


if __name__ == "__main__":
    main()
