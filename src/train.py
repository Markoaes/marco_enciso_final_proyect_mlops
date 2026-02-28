import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.stats import ks_2samp


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Credit Default Experiment")

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Loading training data...")

X_train = pd.read_csv("data/training/X_train.csv")
X_test = pd.read_csv("data/training/X_test.csv")
y_train = pd.read_csv("data/training/y_train.csv").values.ravel()
y_test = pd.read_csv("data/training/y_test.csv").values.ravel()


# ==========================================================
# MODELOS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
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
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        ks = ks_2samp(
            y_prob[y_test == 0],
            y_prob[y_test == 1]
        ).statistic

        mlflow.log_param("model_type", name)
        mlflow.log_metric("ROC_AUC", round(auc, 4))
        mlflow.log_metric("KS", round(ks, 4))

        # =============================
        # CURVA ROC
        # =============================

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        roc_path = f"reports/roc_{name.replace(' ', '_')}.png"
        plt.savefig(roc_path)
        plt.close()

        mlflow.log_artifact(roc_path)

        # =============================
        # CURVA KS
        # =============================

        thresholds = np.linspace(0, 1, 100)
        ks_values = []

        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            tp = ((pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
            fp = ((pred == 1) & (y_test == 0)).sum() / (y_test == 0).sum()
            ks_values.append(tp - fp)

        plt.figure()
        plt.plot(thresholds, ks_values)
        plt.title(f"KS Curve - {name}")
        plt.xlabel("Threshold")
        plt.ylabel("TPR - FPR")
        ks_path = f"reports/ks_{name.replace(' ', '_')}.png"
        plt.savefig(ks_path)
        plt.close()

        mlflow.log_artifact(ks_path)

        # =============================
        # MATRIZ DE CONFUSIÓN
        # =============================

        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = f"reports/conf_matrix_{name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(model, "model")

        results[name] = auc
        trained_models[name] = model

        print(f"AUC: {auc:.4f}")
        print(f"KS: {ks:.4f}")


# ==========================================================
# SELECCIÓN CAMPEÓN
# ==========================================================

champion = max(results, key=results.get)
print("\n===================================")
print(f"Champion Model: {champion}")
print("===================================")

best_model = trained_models[champion]
joblib.dump(best_model, "models/credit_model.pkl")

print("Model saved successfully!")