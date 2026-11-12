"""Descarga el dataset de Expedia (Kaggle) y genera la muestra de trabajo del TPI.

Requisitos (una sola vez):
  1. Cuenta en Kaggle y reglas de la competencia aceptadas en
     https://www.kaggle.com/c/expedia-personalized-sort/rules
  2. Autenticación de la API de Kaggle (kaggle>=2): `kaggle auth login`, o un token guardado en
     %USERPROFILE%\\.kaggle\\access_token, o la variable de entorno KAGGLE_API_TOKEN.

Uso (desde la raíz del repo):
  python data/download_data.py                      # descarga + muestra por defecto
  python data/download_data.py --n-por-estrato 5000 # muestra más chica
  python data/download_data.py --solo-muestra       # rehace la muestra sin volver a descargar

Salidas:
  data/raw/train.csv                        archivo original (no se versiona)
  data/processed/expedia_muestra.parquet    búsquedas completas, estratificadas por random_bool
  data/processed/manifiesto.json            hash del crudo, semilla, tamaños y violaciones
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpi.config import PROCESSED_DIR, RAW_DIR, SEED  # noqa: E402
from tpi.data_contract import (  # noqa: E402
    ARCHIVO_TRAIN,
    CLAVE_BUSQUEDA,
    COMPETICION,
    DTYPES_LECTURA,
    columnas_faltantes,
    validar_contrato,
)
from tpi.sampling import muestrear_busquedas  # noqa: E402

RUTA_TRAIN = RAW_DIR / ARCHIVO_TRAIN
RUTA_MUESTRA = PROCESSED_DIR / "expedia_muestra.parquet"
RUTA_MANIFIESTO = PROCESSED_DIR / "manifiesto.json"
CHUNK = 1_000_000


def descargar() -> None:
    if RUTA_TRAIN.exists():
        print(f"[ok] {RUTA_TRAIN} ya existe; no se descarga de nuevo.")
        return
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[..] Descargando {ARCHIVO_TRAIN} de la competencia '{COMPETICION}' (~GB)...")
    api.competition_download_file(COMPETICION, ARCHIVO_TRAIN, path=str(RAW_DIR), unzip=True)
    # Algunas versiones dejan el zip sin extraer: lo resolvemos acá
    for zp in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zp) as z:
            z.extractall(RAW_DIR)
        zp.unlink()
    if not RUTA_TRAIN.exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_TRAIN} tras la descarga. Archivos en data/raw: "
            f"{[p.name for p in RAW_DIR.iterdir()]}"
        )


def sha256(ruta: Path, bloque: int = 2**22) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while b := f.read(bloque):
            h.update(b)
    return h.hexdigest()


def generar_muestra(n_por_estrato: int, seed: int) -> None:
    # Pasada 1: una fila por búsqueda (solo dos columnas: entra cómodo en memoria)
    print("[..] Pasada 1: inventario de búsquedas...")
    partes = []
    n_filas = 0
    for chunk in pd.read_csv(
        RUTA_TRAIN,
        usecols=[CLAVE_BUSQUEDA, "random_bool"],
        dtype={CLAVE_BUSQUEDA: "int32", "random_bool": "int8"},
        chunksize=CHUNK,
    ):
        n_filas += len(chunk)
        partes.append(chunk.drop_duplicates())
    busquedas = pd.concat(partes).drop_duplicates()
    n_inconsistentes = int(busquedas[CLAVE_BUSQUEDA].duplicated().sum())
    busquedas = busquedas.drop_duplicates(CLAVE_BUSQUEDA)  # si hay, se reporta en el manifiesto
    ids = set(muestrear_busquedas(busquedas, n_por_estrato, seed=seed).tolist())

    # Pasada 2: filas completas de las búsquedas elegidas
    print(f"[..] Pasada 2: extrayendo {len(ids)} búsquedas completas...")
    trozos = []
    for chunk in pd.read_csv(
        RUTA_TRAIN, dtype=DTYPES_LECTURA, parse_dates=["date_time"], chunksize=CHUNK
    ):
        trozos.append(chunk[chunk[CLAVE_BUSQUEDA].isin(ids)])
    muestra = pd.concat(trozos, ignore_index=True)

    faltan = columnas_faltantes(muestra)
    violaciones = validar_contrato(muestra)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    muestra.to_parquet(RUTA_MUESTRA, index=False)

    manifiesto = {
        "fuente": f"kaggle competitions: {COMPETICION}/{ARCHIVO_TRAIN}",
        "sha256_train_csv": sha256(RUTA_TRAIN),
        "seed": seed,
        "n_por_estrato": n_por_estrato,
        "train_n_filas": n_filas,
        "train_n_busquedas": int(busquedas.shape[0]),
        "train_busquedas_por_random_bool": {
            str(k): int(v) for k, v in busquedas["random_bool"].value_counts().items()
        },
        "train_busquedas_random_bool_inconsistente": n_inconsistentes,
        "muestra_n_filas": int(len(muestra)),
        "muestra_n_busquedas": int(muestra[CLAVE_BUSQUEDA].nunique()),
        "columnas_faltantes": faltan,
        "violaciones_contrato": violaciones,
    }
    RUTA_MANIFIESTO.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    if violaciones:
        print("[!!] El contrato de datos tiene violaciones: revisar antes de la Fase 1.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n-por-estrato", type=int, default=15_000, help="búsquedas por random_bool")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--solo-muestra", action="store_true", help="no intenta descargar")
    a = p.parse_args()
    if not a.solo_muestra:
        descargar()
    generar_muestra(a.n_por_estrato, a.seed)


if __name__ == "__main__":
    main()
