"""
modelling.py (versi MLProject)

Melatih model machine learning (Scikit-Learn) untuk memprediksi `Survived`
pada Titanic Dataset. Dirancang untuk dijalankan sebagai MLflow Project
(lewat perintah `mlflow run`), baik secara lokal maupun otomatis lewat
GitHub Actions (Workflow CI).

Catatan penting:
Saat dieksekusi lewat `mlflow run`, MLflow Project SUDAH membuat sebuah run
aktif (di experiment default) dan menyuntikkan MLFLOW_RUN_ID ke environment.
Script ini mendeteksi apakah sudah ada run aktif; jika ada, ia akan
menempel pada run tersebut. Jika dijalankan langsung (`python modelling.py`),
script akan membuat run barunya sendiri.
"""

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

TARGET_COL = "Survived"


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def main(data_path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.sklearn.autolog()

    # Jika MLflow Project sudah menyiapkan run aktif, tempel di situ.
    # Jika tidak (dijalankan langsung), buat run baru.
    started_own_run = mlflow.active_run() is None
    if started_own_run:
        mlflow.start_run(run_name="CI_RandomForest_Retrain")

    try:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=2, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Akurasi model pada data test: {acc:.4f}")
    finally:
        if started_own_run:
            mlflow.end_run()

    print("Training via MLflow Project selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="namadataset_preprocessing.csv"
    )
    args = parser.parse_args()
    main(args.data_path)
