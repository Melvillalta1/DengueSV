"""
Prototipo de Predicción de Dengue en El Salvador
Universidad de El Salvador — Ingeniería

Aplicación Streamlit con dos secciones:
  1. Predictor interactivo: el usuario ajusta condiciones y ve la predicción
  2. Tablero histórico: predicciones del modelo vs casos reales

Uso:
    1. Ejecutar primero:  python entrenar_modelo.py   (genera modelo_dengue.pkl)
    2. Luego lanzar:       streamlit run app.py

Requisitos:
    pip install streamlit pandas numpy xgboost scikit-learn joblib matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predicción de Dengue - El Salvador",
    page_icon="🦟",
    layout="wide",
)

URL_DATOS = "https://raw.githubusercontent.com/Melvillalta1/DengueSV/refs/heads/main/dataset_pais_semana.csv"
ARCHIVO_MODELO = "modelo_dengue.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELO Y DATOS (con caché para eficiencia)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def cargar_modelo():
    return joblib.load(ARCHIVO_MODELO)

@st.cache_data
def cargar_datos():
    df = pd.read_csv(URL_DATOS)
    df = df.sort_values(["anio_epi", "semana_epi"]).reset_index(drop=True)
    return df


try:
    paquete = cargar_modelo()
    modelo = paquete["modelo"]
    predictoras = paquete["predictoras"]
    medianas = paquete["valores_mediana"]
    minimos = paquete["valores_min"]
    maximos = paquete["valores_max"]
    df = cargar_datos()
    modelo_ok = True
except FileNotFoundError:
    modelo_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────────────────────────────────────

st.title("🦟 Predicción de Brotes de Dengue en El Salvador")
st.markdown(
    "Herramienta de apoyo a la vigilancia epidemiológica. "
    "Modelo XGBoost entrenado con datos de OpenDengue (casos) y NASA POWER (clima), "
    "a nivel nacional y resolución semanal."
)

if not modelo_ok:
    st.error(
        "No se encontró el archivo 'modelo_dengue.pkl'. "
        "Ejecuta primero `python entrenar_modelo.py` para generar el modelo."
    )
    st.stop()

# Pestañas
tab1, tab2 = st.tabs(["🔮 Predictor interactivo", "📊 Tablero histórico"])


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 1 — PREDICTOR INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Predictor interactivo")
    st.markdown(
        "Ajusta las condiciones recientes y climáticas para obtener una predicción "
        "de casos de dengue para la siguiente semana."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Historial reciente de casos")
        casos_lag1 = st.slider(
            "Casos la semana pasada", 0, 3000,
            int(medianas.get("casos_dengue_lag1", 150)),
            help="Número de casos reportados la semana anterior"
        )
        casos_prom4 = st.slider(
            "Promedio de casos últimas 4 semanas", 0, 3000,
            int(medianas.get("casos_promedio_4sem", 150)),
            help="Promedio de casos de las 4 semanas previas"
        )
        semana = st.slider("Semana epidemiológica", 1, 52, 26)

    with col2:
        st.subheader("Condiciones climáticas")
        temp = st.slider(
            "Temperatura promedio (°C)", 22.0, 30.0,
            float(medianas.get("t2m_prom_sem", 26.3)), 0.1
        )
        precip = st.slider(
            "Precipitación semanal (mm)", 0.0, 250.0,
            float(medianas.get("precip_total", 19.0)), 1.0
        )
        humedad = st.slider(
            "Humedad promedio (%)", 49.0, 91.0,
            float(medianas.get("humedad_prom", 74.7)), 0.5
        )

    # Construir el vector de entrada con TODAS las predictoras.
    # Los valores no controlados por sliders usan la mediana histórica.
    entrada = {col: medianas.get(col, 0) for col in predictoras}

    # Sobrescribir con los valores de los sliders
    entrada["casos_dengue_lag1"] = casos_lag1
    entrada["casos_dengue_lag2"] = casos_lag1  # aproximación
    entrada["casos_dengue_lag3"] = casos_lag1
    entrada["casos_dengue_lag4"] = casos_lag1
    entrada["casos_promedio_4sem"] = casos_prom4
    entrada["casos_tendencia"] = 0
    entrada["t2m_prom_sem"] = temp
    entrada["precip_total"] = precip
    entrada["humedad_prom"] = humedad
    entrada["semana_sin"] = np.sin(2 * np.pi * semana / 52)
    entrada["semana_cos"] = np.cos(2 * np.pi * semana / 52)
    entrada["estacion_lluviosa"] = 1 if 18 <= semana <= 43 else 0

    # Predecir
    X_pred = pd.DataFrame([entrada])[predictoras]
    pred_log = modelo.predict(X_pred)[0]
    pred_casos = max(0, np.expm1(pred_log))

    st.markdown("---")
    colp1, colp2, colp3 = st.columns([1, 1, 1])
    with colp2:
        st.metric(
            label="Casos previstos para la próxima semana",
            value=f"{pred_casos:.0f}",
        )

    # Clasificación cualitativa del nivel de riesgo (referencial)
    if pred_casos < 100:
        st.success("Nivel de transmisión: BAJO")
    elif pred_casos < 300:
        st.warning("Nivel de transmisión: MODERADO")
    else:
        st.error("Nivel de transmisión: ALTO — considerar acciones preventivas")

    st.caption(
        "Nota: las variables no ajustadas manualmente usan su valor mediano histórico. "
        "Esta es una herramienta de apoyo; las decisiones deben validarse con criterio "
        "epidemiológico profesional."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 2 — TABLERO HISTÓRICO
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Tablero histórico: predicción vs realidad")
    st.markdown(
        "Comparación de las predicciones del modelo contra los casos reales "
        "en el período de prueba (2022-2024)."
    )

    # Generar predicciones sobre el período de prueba
    test = df[df["anio_epi"].isin([2022, 2023, 2024])].copy()
    X_test = test[predictoras].fillna(0)
    test["prediccion"] = np.clip(np.expm1(modelo.predict(X_test)), 0, None)

    # Selector de año
    anio_sel = st.selectbox("Selecciona un año", [2022, 2023, 2024])
    datos_anio = test[test["anio_epi"] == anio_sel].sort_values("semana_epi")

    # Gráfica
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(datos_anio["semana_epi"], datos_anio["casos_dengue"],
            marker="o", color="crimson", linewidth=2, label="Casos reales")
    ax.plot(datos_anio["semana_epi"], datos_anio["prediccion"],
            marker="s", color="steelblue", linewidth=2, linestyle="--",
            label="Predicción del modelo")
    ax.set_xlabel("Semana epidemiológica")
    ax.set_ylabel("Casos de dengue")
    ax.set_title(f"Casos reales vs predichos — {anio_sel}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Métricas del año seleccionado
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(datos_anio["casos_dengue"], datos_anio["prediccion"])
    total_real = datos_anio["casos_dengue"].sum()
    total_pred = datos_anio["prediccion"].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Error absoluto medio (MAE)", f"{mae:.1f} casos")
    m2.metric("Casos reales del año", f"{total_real:,.0f}")
    m3.metric("Casos predichos del año", f"{total_pred:,.0f}")

    st.caption(
        f"En {anio_sel}, el modelo estimó un total de {total_pred:,.0f} casos "
        f"frente a {total_real:,.0f} reales "
        f"(diferencia de {abs(total_pred-total_real)/total_real*100:.1f}%)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Proyecto de grado — Universidad de El Salvador. "
    "Datos: OpenDengue (CC BY-SA) y NASA POWER. "
    "Modelo de apoyo a la vigilancia epidemiológica, no sustituye el criterio profesional."
)
