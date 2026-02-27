import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/UCI_Credit_Card.csv"
OUTPUT_PATH = "data/training"

def main():

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)

    print("Columns found:")
    print(df.columns)

    df = df.rename(columns={"default.payment.next.month": "target"})

    print("Dataset shape:", df.shape)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    X_train.to_csv(f"{OUTPUT_PATH}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_PATH}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_PATH}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_PATH}/y_test.csv", index=False)

    print("Training data saved successfully!")

if __name__ == "__main__":
    main()