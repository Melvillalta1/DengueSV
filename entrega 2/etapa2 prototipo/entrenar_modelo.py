"""
Entrenamiento y guardado del modelo para el prototipo Streamlit
Proyecto: Predicción de Brotes de Dengue en El Salvador
Universidad de El Salvador — Ingeniería

Este script entrena el mejor modelo (XGBoost optimizado) y lo guarda en un archivo .pkl
que la aplicación Streamlit cargará. Ejecutar UNA sola vez antes de lanzar la app.

Uso:
    python entrenar_modelo.py

Requisitos:
    pip install pandas numpy xgboost scikit-learn joblib
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

URL_DATOS = "https://raw.githubusercontent.com/Melvillalta1/DengueSV/refs/heads/main/dataset_pais_semana.csv"
ARCHIVO_MODELO = "modelo_dengue.pkl"

# Hiperparámetros óptimos (hallados con GridSearchCV en la etapa de optimización)
HIPERPARAMS = {
    "learning_rate": 0.03,
    "max_depth": 3,
    "n_estimators": 200,
    "subsample": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def main():
    print("Cargando datos...")
    df = pd.read_csv(URL_DATOS)
    df = df.sort_values(["anio_epi", "semana_epi"]).reset_index(drop=True)

    OBJETIVO = "casos_dengue"
    excluir = ["casos_dengue", "anio_epi", "calendar_start_date",
               "calendar_end_date", "indice_temporal"]
    predictoras = [c for c in df.columns if c not in excluir]

    # Entrenar con todos los datos disponibles hasta 2021 (dejando 2022-2024 como
    # referencia histórica para la app). Para producción real se entrenaría con todo.
    train = df[df["anio_epi"] <= 2021].copy()
    X = train[predictoras].copy()
    y = train[OBJETIVO].copy()
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    print(f"Entrenando XGBoost con {len(X)} muestras y {len(predictoras)} variables...")
    modelo = XGBRegressor(**HIPERPARAMS)
    modelo.fit(X, np.log1p(y))  # objetivo transformado con log

    # Guardar el modelo junto con metadata necesaria para la app
    paquete = {
        "modelo": modelo,
        "predictoras": predictoras,
        "objetivo": OBJETIVO,
        "hiperparametros": HIPERPARAMS,
        # Guardar valores medianos de cada variable como valores por defecto en la app
        "valores_mediana": df[predictoras].median().to_dict(),
        # Guardar rangos para los sliders
        "valores_min": df[predictoras].min().to_dict(),
        "valores_max": df[predictoras].max().to_dict(),
    }
    joblib.dump(paquete, ARCHIVO_MODELO)

    print(f"\n✓ Modelo guardado en '{ARCHIVO_MODELO}'")
    print(f"  Este archivo lo usará la app de Streamlit (app.py)")

    # Verificación rápida
    pred = np.expm1(modelo.predict(X.head(3)))
    print(f"\n  Verificación (primeras 3 predicciones): {pred.round(0)}")
    print(f"  Valores reales correspondientes:       {y.head(3).values}")


if __name__ == "__main__":
    main()
