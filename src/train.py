import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


# ==========================================================
# CONFIGURACIÓN MLFLOW (Tracking Local con SQLite)
# ==========================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Credit Default Experiment")


# ==========================================================
# CARGA DE DATA
# ==========================================================

print("Loading training data...")

X_train = pd.read_csv("data/training/X_train.csv")
X_test = pd.read_csv("data/training/X_test.csv")
y_train = pd.read_csv("data/training/y_train.csv").values.ravel()
y_test = pd.read_csv("data/training/y_test.csv").values.ravel()


# ==========================================================
# DEFINICIÓN DE MODELOS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
}


results = {}
trained_models = {}


# ==========================================================
# ENTRENAMIENTO + TRACKING
# ==========================================================

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)

        ks_stat = ks_2samp(
            y_pred_proba[y_test == 0],
            y_pred_proba[y_test == 1]
        ).statistic

        # Log parámetros básicos
        mlflow.log_param("model_type", name)

        # Log métricas
        mlflow.log_metric("ROC_AUC", round(auc, 4))
        mlflow.log_metric("KS", round(ks_stat, 4))

        # Log modelo como artifact
        mlflow.sklearn.log_model(model, "model")

        results[name] = {"AUC": auc, "KS": ks_stat}
        trained_models[name] = model

        print(f"AUC: {auc:.4f}")
        print(f"KS: {ks_stat:.4f}")


# ==========================================================
# SELECCIÓN DEL MEJOR MODELO
# ==========================================================

champion = max(results, key=lambda x: results[x]["AUC"])

print("\n===================================")
print(f"Champion Model: {champion}")
print("===================================")

best_model = trained_models[champion]


# ==========================================================
# GUARDADO LOCAL PARA API
# ==========================================================

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/credit_model.pkl")

print("Model saved successfully in models/credit_model.pkl")