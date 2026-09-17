# Predicción de Brotes de Dengue en El Salvador — Etapa 2

Proyecto de grado — Universidad de El Salvador, Facultad de Ingeniería y Arquitectura,
Escuela de Ingeniería de Sistemas Informáticos.

Predicción de casos de dengue en El Salvador mediante modelos de Machine Learning,
integrando datos epidemiológicos y climáticos a nivel nacional con resolución semanal.

**Autores:** Castillo Hernández, Mónica Georgina · Villalta Martínez, Mahalaleel

---

## Descripción

Esta etapa desarrolla y optimiza modelos predictivos sobre un conjunto de datos país-semana
(2014–2024), aplicando optimización de hiperparámetros, técnicas de ensemble, análisis de
interpretabilidad, evaluación de sesgos y fairness, y un prototipo funcional.

El modelo final seleccionado es un **XGBoost optimizado**, que alcanza un coeficiente de
determinación (R²) de 0.875 sobre el conjunto de prueba, con un error absoluto medio (MAE)
de 34 casos.

---

## Estructura del repositorio

| Archivo / Carpeta | Descripción |
|-------------------|-------------|
| `01_EDA_pais_semana.ipynb` | Análisis exploratorio del dataset país-semana |
| `02_Entrenamiento_pais_semana.ipynb` | Entrenamiento inicial de los tres modelos |
| `03_Optimizacion.ipynb` | Optimización de hiperparámetros con GridSearchCV + validación cruzada temporal |
| `04_Modelos_avanzados.ipynb` | Ensembles (Voting, Stacking) y modelo sin rezagos |
| `05_Fairness.ipynb` | Análisis de sesgos y equidad del modelo |
| `06_Interpretabilidad_SHAP.ipynb` | Interpretabilidad del modelo mediante SHAP |
| `dataset_pais_semana.csv` | Conjunto de datos utilizado |
| `documento/` | Documento completo de la etapa |
| `prototipo/` | Aplicación web de demostración (Streamlit) |

**Orden de ejecución recomendado:** los notebooks están numerados según el flujo lógico
del proyecto, de 01 a 06.

---

## Datos

**Datos epidemiológicos:** OpenDengue (https://opendengue.org), base de datos desarrollada por
la London School of Hygiene & Tropical Medicine, que estandariza datos públicos de dengue de
diversas fuentes. Cobertura nacional, resolución semanal, período 2014–2024. Variable objetivo:
casos sospechosos de dengue. Licencia: Creative Commons CC BY-SA.

**Datos climáticos:** NASA POWER (https://power.larc.nasa.gov), datos meteorológicos derivados
de observaciones satelitales. Variables: temperatura, precipitación, humedad y viento,
promediados a nivel nacional. Dominio público.

---

## Cómo ejecutar los notebooks

Los notebooks están preparados para ejecutarse en Google Colab y leen el conjunto de datos
directamente desde este repositorio. Basta con abrir cada notebook en Colab y ejecutar las
celdas en orden.

Librerías principales utilizadas: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn.

---

## Cómo ejecutar el prototipo

El prototipo es una aplicación web desarrollada con Streamlit. Instrucciones detalladas en
`prototipo/INSTRUCCIONES_APP.md`. En resumen:

```bash
cd prototipo
pip install -r requirements.txt
python entrenar_modelo.py     # genera el modelo (una sola vez)
streamlit run app.py          # lanza la aplicación
```

La aplicación incluye dos secciones: un predictor interactivo, donde el usuario ajusta las
condiciones y obtiene una predicción, y un tablero histórico, que compara las predicciones
del modelo con los casos reales.

---

## Resumen de resultados

| Modelo | MAE | RMSE | R² |
|--------|-----|------|-----|
| Baseline (persistencia) | 29.82 | 50.86 | 0.875 |
| Random Forest optimizado | 37.55 | — | 0.858 |
| XGBoost optimizado (final) | 34.07 | 50.82 | 0.875 |
| Ensemble Voting | 34.76 | — | 0.874 |

**Hallazgos principales:**
- El modelo dimensiona correctamente los años epidémicos (error del 4% en el total de 2022).
- Los rezagos de casos son los predictores más influyentes, reflejando la autocorrelación
  temporal del dengue.
- Un modelo basado únicamente en clima no logra predecir los brotes, lo que evidencia la
  necesidad de mantener la vigilancia epidemiológica continua.
- El análisis de fairness muestra un comportamiento equitativo entre años, temporadas y
  niveles de casos.

---

## Nota

Este proyecto tiene fines académicos. El modelo constituye una herramienta de apoyo a la
vigilancia epidemiológica y no sustituye el criterio de los profesionales de la salud.
