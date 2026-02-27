import os
import joblib
import pandas as pd
import numpy as np

from scipy.stats import ks_2samp

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

from xgboost import XGBClassifier

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
    # Remove ID
    # -----------------------------
    if "ID" in X_train.columns:
        X_train = X_train.drop(columns=["ID"])
        X_test = X_test.drop(columns=["ID"])

    # -----------------------------
    # Feature definition
    # -----------------------------
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [
        col for col in X_train.columns if col not in categorical_features
    ]

    # -----------------------------
    # Preprocessing pipeline
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {}

    # =============================
    # Logistic Regression
    # =============================
    log_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    log_pipeline.fit(X_train, y_train)
    log_probs = log_pipeline.predict_proba(X_test)[:, 1]
    log_auc = roc_auc_score(y_test, log_probs)
    log_ks = compute_ks(y_test, log_probs)

    print("\nLogistic Regression Results")
    print("AUC:", round(log_auc, 4))
    print("KS:", round(log_ks, 4))

    models["Logistic Regression"] = (log_pipeline, log_auc)

    # =============================
    # Random Forest
    # =============================
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ]
    )

    rf_pipeline.fit(X_train, y_train)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_ks = compute_ks(y_test, rf_probs)

    print("\nRandom Forest Results")
    print("AUC:", round(rf_auc, 4))
    print("KS:", round(rf_ks, 4))

    models["Random Forest"] = (rf_pipeline, rf_auc)

    # =============================
    # XGBoost
    # =============================
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                eval_metric="logloss"
            ))
        ]
    )

    xgb_pipeline.fit(X_train, y_train)
    xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    xgb_ks = compute_ks(y_test, xgb_probs)

    print("\nXGBoost Results")
    print("AUC:", round(xgb_auc, 4))
    print("KS:", round(xgb_ks, 4))

    models["XGBoost"] = (xgb_pipeline, xgb_auc)

    # =============================
    # Champion Selection
    # =============================
    champion_name = max(models, key=lambda k: models[k][1])
    champion_model = models[champion_name][0]

    print("\nChampion Model:", champion_name)

    # -----------------------------
    # Save model
    # -----------------------------
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(champion_model, f"{MODEL_PATH}/credit_model.pkl")

    print("Model saved successfully!")


if __name__ == "__main__":
    main()