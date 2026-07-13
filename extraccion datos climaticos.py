"""
Descarga automatizada de datos climáticos NASA POWER
Proyecto: Predicción de Brotes de Dengue en El Salvador
Universidad de El Salvador - Ingeniería de Sistemas Informáticos

API utilizada: https://power.larc.nasa.gov/api/temporal/daily/point
Sin necesidad de registro ni API key.
"""

import requests
import pandas as pd
import time
import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────────────────────

FECHA_INICIO = "20150101"   # 1 enero 2015
FECHA_FIN    = "20251231"   # 31 diciembre 2025

# Parámetros NASA POWER que necesita el proyecto
# Documentación completa: https://power.larc.nasa.gov/parameters/
PARAMETROS = [
    "T2M_MAX",       # Temperatura máxima a 2 metros (°C)
    "T2M_MIN",       # Temperatura mínima a 2 metros (°C)
    "T2M",           # Temperatura media a 2 metros (°C)
    "PRECTOTCORR",   # Precipitación corregida (mm/día)
    "RH2M",          # Humedad relativa a 2 metros (%)
    "WS2M",          # Velocidad del viento a 2 metros (m/s)
]

# Comunidad AG = Agroclimatología (la más adecuada para estudios de vectores)
COMUNIDAD = "AG"

# Ruta al CSV con los municipios (generado previamente)
CSV_MUNICIPIOS = "data/raw/nasa_power/municipios_el_salvador_completo.csv"

# Carpeta donde se guardarán los datos descargados
CARPETA_SALIDA = "data/raw/nasa_power"

# Pausa entre peticiones para no sobrecargar el servidor (segundos)
PAUSA_ENTRE_PETICIONES = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE MUNICIPIOS DESDE CSV
# ─────────────────────────────────────────────────────────────────────────────

def cargar_municipios(ruta_csv: str) -> list[dict]:
    """Lee el CSV de municipios y retorna una lista de dicts."""
    if not os.path.exists(ruta_csv):
        print(f"✗ ERROR: No se encontró el archivo '{ruta_csv}'")
        print(f"  Asegúrate de que el CSV esté en la misma carpeta que este script.")
        sys.exit(1)

    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")  # utf-8-sig maneja el BOM si existe

    # Validar columnas requeridas
    columnas_requeridas = {"municipio", "departamento", "lat", "lon"}
    columnas_faltantes  = columnas_requeridas - set(df.columns.str.lower())
    if columnas_faltantes:
        print(f"✗ ERROR: El CSV no tiene las columnas requeridas: {columnas_faltantes}")
        sys.exit(1)

    # Normalizar nombres de columnas a minúsculas por si acaso
    df.columns = df.columns.str.lower().str.strip()

    municipios = df.to_dict(orient="records")
    print(f"✓ {len(municipios)} municipios cargados desde '{ruta_csv}'")
    return municipios


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE DESCARGA
# ─────────────────────────────────────────────────────────────────────────────

def construir_url(lat: float, lon: float) -> str:
    """Construye la URL de la API de NASA POWER para un punto geográfico."""
    parametros_str = ",".join(PARAMETROS)
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={parametros_str}"
        f"&community={COMUNIDAD}"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={FECHA_INICIO}"
        f"&end={FECHA_FIN}"
        f"&format=JSON"
    )
    return url


def descargar_municipio(municipio: dict) -> pd.DataFrame | None:
    """Descarga los datos climáticos de un municipio y retorna un DataFrame."""
    nombre = municipio["municipio"]
    depto  = municipio["departamento"]
    lat    = municipio["lat"]
    lon    = municipio["lon"]

    url = construir_url(lat, lon)
    print(f"  Descargando: {nombre} ({depto}) — lat={lat}, lon={lon}")

    try:
        respuesta = requests.get(url, timeout=60)
        respuesta.raise_for_status()
        datos_json = respuesta.json()

        # Extraer los parámetros del JSON de respuesta
        parametros_data = datos_json["properties"]["parameter"]

        # Convertir a DataFrame: cada parámetro es un dict {fecha: valor}
        df = pd.DataFrame(parametros_data)
        df.index.name = "fecha"
        df.reset_index(inplace=True)

        # Convertir la fecha de YYYYMMDD a formato datetime
        df["fecha"] = pd.to_datetime(df["fecha"], format="%Y%m%d")

        # Agregar columnas de identificación del municipio
        df.insert(0, "municipio",    nombre)
        df.insert(1, "departamento", depto)
        df.insert(2, "latitud",      lat)
        df.insert(3, "longitud",     lon)

        # Reemplazar valores -999 (código de dato faltante de NASA) con NaN
        df.replace(-999.0, float("nan"), inplace=True)

        print(f"    ✓ {len(df)} registros descargados")
        return df

    except requests.exceptions.Timeout:
        print(f"    ✗ TIMEOUT para {nombre}. Se omite.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"    ✗ Error HTTP {e.response.status_code} para {nombre}.")
        return None
    except Exception as e:
        print(f"    ✗ Error inesperado para {nombre}: {e}")
        return None


def guardar_municipio(df: pd.DataFrame, municipio: dict):
    """Guarda el DataFrame de un municipio como CSV individual."""
    nombre_archivo = municipio["municipio"].replace(" ", "_").lower()
    depto_archivo  = municipio["departamento"].replace(" ", "_").lower()
    ruta = os.path.join(CARPETA_SALIDA, depto_archivo)
    os.makedirs(ruta, exist_ok=True)
    archivo = os.path.join(ruta, f"{nombre_archivo}.csv")
    df.to_csv(archivo, index=False, encoding="utf-8")
    print(f"    💾 Guardado en: {archivo}")


def consolidar_todo(carpeta: str) -> pd.DataFrame:
    """Lee todos los CSV individuales y los combina en un solo archivo."""
    print("\n📦 Consolidando todos los archivos...")
    dfs = []
    for raiz, _, archivos in os.walk(carpeta):
        for archivo in archivos:
            if archivo.endswith(".csv") and archivo != "consolidado.csv":
                ruta = os.path.join(raiz, archivo)
                dfs.append(pd.read_csv(ruta, parse_dates=["fecha"]))
    if not dfs:
        print("  No se encontraron archivos para consolidar.")
        return pd.DataFrame()
    consolidado = pd.concat(dfs, ignore_index=True)
    consolidado.sort_values(["municipio", "fecha"], inplace=True)
    ruta_consolidado = os.path.join(carpeta, "consolidado.csv")
    consolidado.to_csv(ruta_consolidado, index=False, encoding="utf-8")
    print(f"  ✓ Consolidado guardado: {ruta_consolidado}")
    print(f"  ✓ Total de filas: {len(consolidado):,}")
    return consolidado


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def ya_descargado(municipio: dict) -> bool:
    """Verifica si el CSV de un municipio ya existe (para reanudar descargas)."""
    nombre_archivo = municipio["municipio"].replace(" ", "_").lower()
    depto_archivo  = municipio["departamento"].replace(" ", "_").lower()
    ruta = os.path.join(CARPETA_SALIDA, depto_archivo, f"{nombre_archivo}.csv")
    return os.path.exists(ruta)


def main():
    # ── Cargar municipios desde el CSV ────────────────────────────────────
    municipios = cargar_municipios(CSV_MUNICIPIOS)

    print("=" * 65)
    print("  NASA POWER — Descarga de datos climáticos para El Salvador")
    print(f"  Período    : {FECHA_INICIO} → {FECHA_FIN}")
    print(f"  Municipios : {len(municipios)}")
    print(f"  Parámetros : {', '.join(PARAMETROS)}")
    print(f"  Salida     : {CARPETA_SALIDA}/")
    print("=" * 65)

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    inicio    = datetime.now()
    exitosos  = 0
    omitidos  = 0   # ya existían
    fallidos  = []

    for i, municipio in enumerate(municipios, start=1):
        nombre = municipio["municipio"]

        # ── Reanudación automática: saltar si ya está descargado ──────────
        if ya_descargado(municipio):
            print(f"[{i}/{len(municipios)}] ⏭  {nombre} — ya descargado, se omite")
            omitidos += 1
            continue

        print(f"\n[{i}/{len(municipios)}] {nombre} ({municipio['departamento']})")
        df = descargar_municipio(municipio)

        if df is not None:
            guardar_municipio(df, municipio)
            exitosos += 1
        else:
            fallidos.append(nombre)

        # Pausa para no saturar el servidor
        if i < len(municipios):
            time.sleep(PAUSA_ENTRE_PETICIONES)

    # ── Consolidar todo en un archivo único ───────────────────────────────
    consolidar_todo(CARPETA_SALIDA)

    # ── Resumen final ─────────────────────────────────────────────────────
    duracion = datetime.now() - inicio
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL")
    print(f"  Descargados esta sesión : {exitosos}")
    print(f"  Ya existían (omitidos)  : {omitidos}")
    print(f"  Fallidos                : {len(fallidos)}")
    if fallidos:
        print(f"  Municipios fallidos: {', '.join(fallidos)}")
        print("  → Puedes volver a ejecutar el script para reintentar los fallidos.")
    print(f"  Tiempo total: {str(duracion).split('.')[0]}")
    print("=" * 65)


if __name__ == "__main__":
    main()