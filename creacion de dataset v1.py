"""
Generación del Dataset v1 (preliminar) para el modelo de predicción de dengue
Proyecto: Predicción de Brotes de Dengue - Universidad de El Salvador

Este script:
  1. Une la tabla de dengue con la de clima (LEFT JOIN desde dengue)
  2. Genera variables biológicas derivadas basadas en literatura del Aedes aegypti
  3. Genera variables temporales/estacionales
  4. Genera rezagos epidemiológicos (casos de semanas anteriores)
  5. Guarda el dataset preliminar en CSV y PostgreSQL

NOTA: Este es el dataset PRELIMINAR para EDA. Mantiene todos los años y filas
      para poder analizar cobertura y justificar decisiones. El dataset FINAL
      (depurado) se genera después con base en las conclusiones del EDA.

Umbrales biológicos (fuente: literatura científica Aedes aegypti):
  - Temperatura letal por frío: mínima < 10°C (aumento rápido de mortalidad larval)
  - Umbral inferior de desarrollo: ~10°C (base para grados-día)
  - Rango óptimo de reproducción: 22–32°C
  - Estrés/mortalidad por calor: > 35°C (desarrollo se detiene)

Requisitos:
  pip install pandas numpy sqlalchemy psycopg2-binary
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "DengueSV",   # ← cambia esto
    "user":     "postgres",            # ← cambia esto
    "password": "Mel05252026$",         # ← cambia esto
}

TABLA_DENGUE = "dengue_dep_sem"
TABLA_CLIMA  = "clima_departamental_semanal"
TABLA_SALIDA = "dataset_v1_preliminar"
CSV_SALIDA   = "dataset_v1_preliminar.csv"

# Umbrales biológicos del Aedes aegypti (de la literatura)
TEMP_LETAL_FRIO     = 10    # °C — bajo esto aumenta mortalidad larval
TEMP_BASE_GRADOS    = 10    # °C — base para acumulación de grados-día
TEMP_OPTIMA_MIN     = 22    # °C — límite inferior rango óptimo
TEMP_OPTIMA_MAX     = 32    # °C — límite superior rango óptimo
TEMP_ESTRES_CALOR   = 35    # °C — sobre esto se detiene el desarrollo


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — Cargar y unir datos
# ─────────────────────────────────────────────────────────────────────────────

def cargar_join(engine) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print("PASO 1 — Uniendo dengue + clima (LEFT JOIN desde dengue)")
    print(f"{'─'*60}")

    query = f"""
        SELECT
            d.anio_epi,
            d.semana_epi,
            d.departamento,
            d.casos_confirmados,
            d.casos_probables,
            d.casos_total,
            c.t2m_max_sem,
            c.t2m_min_sem,
            c.t2m_prom_sem,
            c.precip_total,
            c.precip_dias_lluvia,
            c.humedad_prom,
            c.viento_prom,
            c.t2m_prom_sem_lag1,
            c.t2m_prom_sem_lag2,
            c.t2m_prom_sem_lag3,
            c.t2m_prom_sem_lag4,
            c.precip_total_lag1,
            c.precip_total_lag2,
            c.precip_total_lag3,
            c.precip_total_lag4,
            c.humedad_prom_lag1,
            c.humedad_prom_lag2,
            c.humedad_prom_lag3,
            c.humedad_prom_lag4,
            c.t2m_max_sem_lag1,
            c.t2m_max_sem_lag2,
            c.t2m_max_sem_lag3,
            c.t2m_max_sem_lag4
        FROM {TABLA_DENGUE} d
        LEFT JOIN {TABLA_CLIMA} c
               ON d.departamento = c.departamento
              AND d.anio_epi     = c.anio_epi
              AND d.semana_epi   = c.semana_epi
        ORDER BY d.departamento, d.anio_epi, d.semana_epi
    """
    df = pd.read_sql(query, engine)

    sin_clima = df["t2m_prom_sem"].isna().sum()
    print(f"  ✓ {len(df):,} filas unidas")
    print(f"  ✓ Filas sin clima correspondiente: {sin_clima}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — Variables biológicas derivadas
# ─────────────────────────────────────────────────────────────────────────────

def variables_biologicas(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print("PASO 2 — Generando variables biológicas (Aedes aegypti)")
    print(f"{'─'*60}")

    # Amplitud térmica diurna (diferencia max - min)
    df["amplitud_termica"] = df["t2m_max_sem"] - df["t2m_min_sem"]

    # Grados-día de desarrollo (calor útil acumulado sobre la base)
    df["grados_dia"] = (df["t2m_prom_sem"] - TEMP_BASE_GRADOS).clip(lower=0)

    # Semana en rango óptimo de reproducción (binaria)
    df["en_rango_optimo"] = (
        (df["t2m_prom_sem"] >= TEMP_OPTIMA_MIN) &
        (df["t2m_prom_sem"] <= TEMP_OPTIMA_MAX)
    ).astype(int)

    # Temperatura mínima bajo umbral letal por frío (binaria)
    df["temp_bajo_letal_frio"] = (df["t2m_min_sem"] < TEMP_LETAL_FRIO).astype(int)

    # Temperatura máxima sobre umbral de estrés por calor (binaria)
    df["temp_sobre_estres_calor"] = (df["t2m_max_sem"] > TEMP_ESTRES_CALOR).astype(int)

    # Distancia al centro del rango óptimo (27°C es el punto medio ideal)
    centro_optimo = (TEMP_OPTIMA_MIN + TEMP_OPTIMA_MAX) / 2  # 27°C
    df["dist_temp_optima"] = (df["t2m_prom_sem"] - centro_optimo).abs()

    # Indicador de semana con lluvia significativa (criaderos)
    df["semana_lluviosa"] = (df["precip_total"] > 10).astype(int)

    nuevas = ["amplitud_termica", "grados_dia", "en_rango_optimo",
              "temp_bajo_letal_frio", "temp_sobre_estres_calor",
              "dist_temp_optima", "semana_lluviosa"]
    for v in nuevas:
        print(f"  ✓ {v}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — Variables temporales / estacionales
# ─────────────────────────────────────────────────────────────────────────────

def variables_temporales(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print("PASO 3 — Generando variables temporales / estacionales")
    print(f"{'─'*60}")

    # Estación lluviosa en El Salvador: aprox. mayo–octubre (SE ~18–43)
    df["estacion_lluviosa"] = df["semana_epi"].between(18, 43).astype(int)

    # Semana del año como variables cíclicas (seno/coseno)
    # Esto permite al modelo entender que SE52 y SE1 están "cerca"
    df["semana_sin"] = np.sin(2 * np.pi * df["semana_epi"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_epi"] / 52)

    for v in ["estacion_lluviosa", "semana_sin", "semana_cos"]:
        print(f"  ✓ {v}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — Rezagos epidemiológicos (casos de semanas anteriores)
# ─────────────────────────────────────────────────────────────────────────────

def rezagos_epidemiologicos(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print("PASO 4 — Generando rezagos epidemiológicos (casos previos)")
    print(f"{'─'*60}")

    # IMPORTANTE: ordenar por departamento y tiempo antes de rezagar
    df = df.sort_values(["departamento", "anio_epi", "semana_epi"]).reset_index(drop=True)

    # Rezagos de casos: predictor más fuerte del dengue (autocorrelación)
    # Solo usar casos de semanas ANTERIORES para no filtrar información futura
    for lag in range(1, 5):
        df[f"casos_total_lag{lag}"] = df.groupby("departamento")["casos_total"].shift(lag)
        print(f"  ✓ casos_total_lag{lag}")

    # Tendencia: diferencia entre semana anterior y hace 2 semanas
    df["casos_tendencia"] = df["casos_total_lag1"] - df["casos_total_lag2"]
    print(f"  ✓ casos_tendencia")

    # Promedio móvil de las últimas 4 semanas (suaviza ruido)
    df["casos_promedio_4sem"] = (
        df.groupby("departamento")["casos_total"]
          .shift(1)
          .rolling(window=4, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )
    print(f"  ✓ casos_promedio_4sem")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — Guardar
# ─────────────────────────────────────────────────────────────────────────────

def guardar(df: pd.DataFrame, engine):
    print(f"\n{'─'*60}")
    print("PASO 5 — Guardando dataset v1 preliminar")
    print(f"{'─'*60}")

    # CSV
    df.to_csv(CSV_SALIDA, index=False, encoding="utf-8")
    print(f"  ✓ CSV guardado: {CSV_SALIDA}")

    # PostgreSQL
    try:
        df.to_sql(TABLA_SALIDA, engine, index=False, if_exists="replace")
        print(f"  ✓ Tabla '{TABLA_SALIDA}' guardada en PostgreSQL")
    except Exception as e:
        print(f"  ✗ Error PostgreSQL (CSV ya guardado): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — Resumen para EDA
# ─────────────────────────────────────────────────────────────────────────────

def resumen_eda(df: pd.DataFrame):
    print(f"\n{'─'*60}")
    print("PASO 6 — Resumen para EDA")
    print(f"{'─'*60}")

    print(f"\n  Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    print(f"\n  Cobertura por año (filas y % con clima):")
    resumen = df.groupby("anio_epi").agg(
        filas       = ("departamento",  "count"),
        deptos      = ("departamento",  "nunique"),
        semanas     = ("semana_epi",    "nunique"),
        con_clima   = ("t2m_prom_sem",  lambda x: x.notna().sum()),
        casos_total = ("casos_total",   "sum"),
    )
    resumen["pct_clima"] = (resumen["con_clima"] / resumen["filas"] * 100).round(1)
    print(resumen.to_string())

    print(f"\n  Columnas del dataset ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

    # Filas utilizables para el modelo (con clima y con rezagos)
    utilizables = df.dropna(subset=["t2m_prom_sem", "casos_total_lag4"])
    print(f"\n  Filas con clima Y rezagos completos: {len(utilizables):,}")
    print(f"  (Estas son las filas entrenables tras eliminar NaN de rezagos)")


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Dataset v1 Preliminar — Predicción de Dengue")
    print("  Universidad de El Salvador")
    print("=" * 60)

    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    engine = create_engine(url)

    df = cargar_join(engine)
    df = variables_biologicas(df)
    df = variables_temporales(df)
    df = rezagos_epidemiologicos(df)
    guardar(df, engine)
    resumen_eda(df)

    print(f"\n{'='*60}")
    print("  ✓ Dataset v1 preliminar completado")
    print("  Siguiente paso: EDA para decidir años/filas del dataset final")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()