import os
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

DATA_PATH = "data/training"
MODEL_PATH = "models"


def compute_ks(y_true, y_prob):
    return ks_2samp(
        y_prob[y_true == 1],
        y_prob[y_true == 0]
    ).statistic


def main():

    print("Loading training data...")

    X_train = pd.read_csv(f"{DATA_PATH}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv").values.ravel()

    # -----------------------------
    # 1️⃣ Eliminar Target 
    # -----------------------------
    if "ID" in X_train.columns:
        X_train = X_train.drop(columns=["ID"])
        X_test = X_test.drop(columns=["ID"])

    # -----------------------------
    # 2️⃣ Definir variables categóricas
    #     # -----------------------------
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    numerical_features = [
        col for col in X_train.columns if col not in categorical_features
    ]

    # -----------------------------
    # 3️⃣ Preprocessing pipeline
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # =============================
    # MODELO 1: Logistic Regression
    # =============================

    log_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    print("Training Logistic Regression...")
    log_pipeline.fit(X_train, y_train)

    log_probs = log_pipeline.predict_proba(X_test)[:, 1]
    log_auc = roc_auc_score(y_test, log_probs)

    print(f"\nLogistic Regression AUC: {log_auc:.4f}")

    # =============================
    # MODELO 2: Random Forest
    # =============================

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
        ]
    )

    print("Training Random Forest...")
    rf_pipeline.fit(X_train, y_train)

    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    print(f"Random Forest AUC: {rf_auc:.4f}")

    # -----------------------------
    # 📊 Métricas 
    # -----------------------------
    print("\nClassification Report - Random Forest:")
    rf_predictions = rf_pipeline.predict(X_test)
    print(classification_report(y_test, rf_predictions))

    rf_ks = compute_ks(y_test, rf_probs)
    print(f"Random Forest KS: {rf_ks:.4f}")

    # -----------------------------
    # 4️⃣ Selección del mejor modelo
    # -----------------------------
    if rf_auc > log_auc:
        champion_model = rf_pipeline
        print("\nChampion Model: Random Forest")
    else:
        champion_model = log_pipeline
        print("\nChampion Model: Logistic Regression")

    # -----------------------------
    # 5️⃣ Guardar modelo
    # -----------------------------
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(champion_model, f"{MODEL_PATH}/credit_model.pkl")

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()