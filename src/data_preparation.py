import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/UCI_Credit_Card.csv"
OUTPUT_PATH = "data/training"
REPORTS_PATH = "reports"


def main():

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)

    print("Columns found:")
    print(df.columns)

    # Renombrar target
    df = df.rename(columns={"default.payment.next.month": "target"})

    print("Dataset shape:", df.shape)

    # ===============================
    # Separación de variables
    # ===============================

    X = df.drop("target", axis=1)
    y = df["target"]

    # ===============================
    # Train / Test Split
    # ===============================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)

    # Guardar datasets
    X_train.to_csv(f"{OUTPUT_PATH}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_PATH}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_PATH}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_PATH}/y_test.csv", index=False)

    print("Training data saved successfully!")

    # ==========================================================
    # ANÁLISIS EXPLORATORIO BÁSICO (EDA)
    # ==========================================================

    print("Generating EDA visualizations...")

    # -------------------------------
    # 1. Distribución del Target
    # -------------------------------

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_train)
    plt.title("Distribución del Target (Train)")
    plt.xlabel("Default (0 = No, 1 = Sí)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}/target_distribution.png")
    plt.close()

    # -------------------------------
    # 2. Boxplot variables numéricas (subset)
    # -------------------------------

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns[:5]

    plt.figure(figsize=(10, 6))
    X_train[numeric_cols].boxplot()
    plt.xticks(rotation=45)
    plt.title("Boxplot Variables Numéricas (Muestra)")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}/boxplot_numeric_variables.png")
    plt.close()

    # -------------------------------
    # 3. Matriz de correlación (subset)
    # -------------------------------

    corr = X_train[numeric_cols].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación (Subset)")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}/correlation_matrix.png")
    plt.close()

    # -------------------------------
    # 4. Histograma variable ejemplo
    # -------------------------------

    col_example = numeric_cols[0]

    plt.figure(figsize=(6, 4))
    sns.histplot(X_train[col_example], kde=True)
    plt.title(f"Distribución de {col_example}")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}/distribution_example_variable.png")
    plt.close()

    print("EDA plots saved in reports/")


if __name__ == "__main__":
    main()