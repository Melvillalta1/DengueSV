"""
Agregación de datos climáticos diarios → semanas epidemiológicas
Proyecto: Predicción de Brotes de Dengue - Universidad de El Salvador

Pasos que realiza este script:
  1. Lee los datos diarios desde PostgreSQL
  2. Calcula la semana epidemiológica OPS para cada fila
  3. Agrega a nivel departamento × semana
  4. Genera variables rezagadas (lag 1 a 4 semanas)
  5. Guarda la tabla de semanas en PostgreSQL (referencia)
  6. Guarda el dataset final en PostgreSQL y en CSV

Requisitos:
  pip install pandas psycopg2-binary epiweeks sqlalchemy
"""

import pandas as pd
from epiweeks import Week, Year
from sqlalchemy import create_engine, text
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estos valores a tu entorno
# ─────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "DengueSV",
    "user":     "postgres",   
    "password": "Mel05252026$",
}

ANIO_INICIO = 2015
ANIO_FIN    = 2025

# Número de semanas de rezago a generar
SEMANAS_LAG = 4

# Tabla fuente en PostgreSQL
TABLA_ORIGEN  = "clima_nasa_power"

# Tablas destino en PostgreSQL
TABLA_SEMANAS = "semanas_epidemiologicas"
TABLA_DESTINO = "clima_departamental_semanal"

# Archivo CSV de salida
CSV_SALIDA    = "clima_departamental_semanal.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CONEXIÓN
# ─────────────────────────────────────────────────────────────────────────────

def conectar() -> create_engine:
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    engine = create_engine(url)
    print("✓ Conexión a PostgreSQL establecida")
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — Generar y guardar tabla de semanas epidemiológicas en PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def crear_tabla_semanas(engine) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print("PASO 1 — Generando tabla de semanas epidemiológicas OPS")
    print(f"{'─'*60}")

    filas = []
    for anio in range(ANIO_INICIO, ANIO_FIN + 1):
        for semana in Year(anio).iterweeks():
            filas.append({
                "anio_epi":     semana.year,
                "semana_epi":   semana.week,
                "fecha_inicio": semana.startdate(),   # domingo (estándar OPS)
                "fecha_fin":    semana.enddate(),      # sábado
                "label":        f"SE{semana.week:02d}-{semana.year}"
            })

    df_semanas = pd.DataFrame(filas)

    # Guardar en PostgreSQL
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLA_SEMANAS}"))

    df_semanas.to_sql(TABLA_SEMANAS, engine, index=False, if_exists="replace")

    # Agregar clave primaria e índice
    with engine.begin() as conn:
        conn.execute(text(
            f"ALTER TABLE {TABLA_SEMANAS} "
            f"ADD PRIMARY KEY (anio_epi, semana_epi)"
        ))

    print(f"  ✓ {len(df_semanas)} semanas generadas ({ANIO_INICIO}–{ANIO_FIN})")
    print(f"  ✓ Tabla '{TABLA_SEMANAS}' guardada en PostgreSQL")
    print(f"\n  Primeras 5 semanas:")
    print(df_semanas.head().to_string(index=False))
    return df_semanas


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — Leer datos diarios desde PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def leer_datos_diarios(engine) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print("PASO 2 — Leyendo datos diarios desde PostgreSQL")
    print(f"{'─'*60}")

    inicio = time.time()
    query  = f"""
        SELECT municipio, departamento, fecha,
               t2m_max, t2m_min, t2m,
               prectotcorr, rh2m, ws2m
        FROM {TABLA_ORIGEN}
        ORDER BY departamento, municipio, fecha
    """
    df = pd.read_sql(query, engine, parse_dates=["fecha"])
    duracion = time.time() - inicio

    print(f"  ✓ {len(df):,} registros leídos en {duracion:.1f}s")
    print(f"  ✓ Departamentos: {df['departamento'].nunique()}")
    print(f"  ✓ Municipios   : {df['municipio'].nunique()}")
    print(f"  ✓ Período      : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"  ✓ Nulos en t2m : {df['t2m'].isna().sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — Calcular semana epidemiológica por fila
# ─────────────────────────────────────────────────────────────────────────────

def agregar_semana_epi(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print("PASO 3 — Calculando semana epidemiológica OPS por fila")
    print(f"{'─'*60}")

    inicio = time.time()

    # Calcular semana epidemiológica vectorizada
    semanas = df["fecha"].apply(lambda d: Week.fromdate(d.date()))
    df["anio_epi"]   = semanas.apply(lambda w: w.year)
    df["semana_epi"] = semanas.apply(lambda w: w.week)

    duracion = time.time() - inicio
    print(f"  ✓ Semanas calculadas en {duracion:.1f}s")
    print(f"  ✓ Semanas únicas: {df[['anio_epi','semana_epi']].drop_duplicates().shape[0]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — Agregar a nivel departamento × semana
# ─────────────────────────────────────────────────────────────────────────────

def agregar_por_semana(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print("PASO 4 — Agregando a nivel departamento × semana epidemiológica")
    print(f"{'─'*60}")

    # Temperatura y humedad: promedio entre municipios
    # Precipitación: promedio espacial (mm/día promedio del depto) luego suma semanal
    # Nota: se promedian primero entre municipios del depto para cada día,
    #       luego se agrega semanalmente

    # Paso 4a — promediar municipios por día dentro de cada departamento
    diario_depto = df.groupby(["departamento", "fecha", "anio_epi", "semana_epi"]).agg(
        t2m_max     = ("t2m_max",     "mean"),
        t2m_min     = ("t2m_min",     "mean"),
        t2m         = ("t2m",         "mean"),
        prectotcorr = ("prectotcorr", "mean"),   # mm/día promedio del depto
        rh2m        = ("rh2m",        "mean"),
        ws2m        = ("ws2m",        "mean"),
        n_municipios= ("municipio",   "count"),
    ).reset_index()

    # Paso 4b — agregar los días a semana epidemiológica
    semanal = diario_depto.groupby(["departamento", "anio_epi", "semana_epi"]).agg(
        t2m_max_sem  = ("t2m_max",     "max"),    # máxima de la semana
        t2m_min_sem  = ("t2m_min",     "min"),    # mínima de la semana
        t2m_prom_sem = ("t2m",         "mean"),   # temperatura media semanal
        precip_total = ("prectotcorr", "sum"),    # mm acumulados en la semana
        precip_dias_lluvia = ("prectotcorr", lambda x: (x > 0.1).sum()),  # días con lluvia
        humedad_prom = ("rh2m",        "mean"),
        viento_prom  = ("ws2m",        "mean"),
        dias_datos   = ("fecha",       "count"),  # días con datos (control calidad)
        n_municipios = ("n_municipios","first"),
    ).reset_index()

    # Agregar etiqueta legible
    semanal["label_semana"] = semanal.apply(
        lambda r: f"SE{int(r['semana_epi']):02d}-{int(r['anio_epi'])}", axis=1
    )

    print(f"  ✓ Dataset semanal generado: {len(semanal):,} filas")
    print(f"  ✓ Estructura esperada: 14 deptos × ~573 semanas = ~8,022 filas")
    print(f"\n  Columnas generadas:")
    for col in semanal.columns:
        print(f"    · {col}")
    return semanal


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — Generar variables rezagadas (lag 1 a 4 semanas)
# ─────────────────────────────────────────────────────────────────────────────

def generar_rezagos(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print(f"PASO 5 — Generando variables rezagadas (lag 1 a {SEMANAS_LAG} semanas)")
    print(f"{'─'*60}")

    # Variables sobre las que se calculan los rezagos
    vars_lag = ["t2m_prom_sem", "precip_total", "humedad_prom", "t2m_max_sem"]

    df = df.sort_values(["departamento", "anio_epi", "semana_epi"]).copy()

    for var in vars_lag:
        for lag in range(1, SEMANAS_LAG + 1):
            nombre = f"{var}_lag{lag}"
            df[nombre] = df.groupby("departamento")[var].shift(lag)
            print(f"  ✓ {nombre}")

    # Cuántas filas pierden datos por rezago (las primeras semanas de cada depto)
    n_nulos = df[[f"{v}_lag{SEMANAS_LAG}" for v in vars_lag]].isna().any(axis=1).sum()
    print(f"\n  ℹ Filas con NaN por rezago (primeras semanas): {n_nulos}")
    print(f"    → Se pueden mantener para variables sin lag y excluir al entrenar el modelo")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────

def guardar_resultados(df: pd.DataFrame, engine):
    print(f"\n{'─'*60}")
    print("PASO 6 — Guardando resultados")
    print(f"{'─'*60}")

    # Guardar en PostgreSQL
    df.to_sql(TABLA_DESTINO, engine, index=False, if_exists="replace")
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_semanal_depto_semana "
            f"ON {TABLA_DESTINO} (departamento, anio_epi, semana_epi)"
        ))
    print(f"  ✓ Tabla '{TABLA_DESTINO}' guardada en PostgreSQL ({len(df):,} filas)")

    # Guardar en CSV
    df.to_csv(CSV_SALIDA, index=False, encoding="utf-8")
    print(f"  ✓ CSV guardado: {CSV_SALIDA}")

    # Vista previa
    print(f"\n  Vista previa (San Salvador, primeras 3 semanas):")
    preview = df[df["departamento"] == "San Salvador"].head(3)
    cols_preview = ["departamento", "anio_epi", "semana_epi", "label_semana",
                    "t2m_prom_sem", "precip_total", "humedad_prom"]
    print(preview[cols_preview].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — Consulta de verificación en PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def verificar(engine):
    print(f"\n{'─'*60}")
    print("PASO 7 — Verificación en PostgreSQL")
    print(f"{'─'*60}")

    query = f"""
        SELECT
            departamento,
            COUNT(*)                              AS semanas,
            ROUND(AVG(t2m_prom_sem)::numeric, 2) AS temp_media,
            ROUND(SUM(precip_total)::numeric, 1)  AS precip_total_mm,
            MIN(anio_epi)                         AS desde,
            MAX(anio_epi)                         AS hasta
        FROM {TABLA_DESTINO}
        GROUP BY departamento
        ORDER BY departamento
    """
    resultado = pd.read_sql(query, engine)
    print(resultado.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Agregación Climática → Semanas Epidemiológicas OPS")
    print("  Proyecto Dengue - Universidad de El Salvador")
    print("=" * 60)

    inicio_total = time.time()

    engine    = conectar()
    df_sem    = crear_tabla_semanas(engine)
    df_diario = leer_datos_diarios(engine)
    df_diario = agregar_semana_epi(df_diario)
    df_sem    = agregar_por_semana(df_diario)
    df_final  = generar_rezagos(df_sem)
    guardar_resultados(df_final, engine)
    verificar(engine)

    duracion = time.time() - inicio_total
    print(f"\n{'='*60}")
    print(f"  ✓ Proceso completado en {duracion:.1f}s")
    print(f"  Dataset listo para unirse con datos epidemiológicos del MINSAL")
    print(f"  Clave de unión: departamento + anio_epi + semana_epi")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()