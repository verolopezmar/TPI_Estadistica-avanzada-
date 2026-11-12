"""Descarga el dataset de Expedia (Kaggle) y genera la muestra de trabajo del TPI.

Requisitos (una sola vez):
  1. Cuenta en Kaggle y reglas de la competencia aceptadas en
     https://www.kaggle.com/c/expedia-personalized-sort/rules
  2. Autenticación de la API de Kaggle (kaggle>=2): `kaggle auth login`, o un token guardado en
     ~/.kaggle/access_token, o la variable de entorno KAGGLE_API_TOKEN.

Confidencialidad: las reglas prohíben compartir los datos con no participantes. Por eso el crudo
se guarda FUERA del repo (por defecto ~/data/tpi-eaymr/raw; configurable con TPI_RAW_DIR) y la
muestra queda en data/processed/, ignorada por git. Solo se versiona el manifiesto (hashes y
tamaños, sin registros). tests/test_confidencialidad.py falla si se trackea un archivo de datos.

Uso (desde la raíz del repo):
  python data/download_data.py                      # descarga + muestra por defecto
  python data/download_data.py --n-por-estrato 5000 # muestra más chica
  python data/download_data.py --solo-muestra       # rehace la muestra sin volver a descargar

Estructura de la competencia: Kaggle publica `data.zip` (train.csv + test.csv) y tres zips de
benchmarks que no usamos. test.csv no trae position, click_bool, booking_bool ni
gross_bookings_usd, así que solo extraemos train.csv.

Salidas:
  $TPI_RAW_DIR/data.zip                     zip original (fuera del repo)
  $TPI_RAW_DIR/train.csv                    único archivo extraído (fuera del repo)
  data/processed/expedia_muestra.parquet    búsquedas completas, estratificadas por random_bool
  data/manifiesto.json                      SHA-256 del zip y del csv, semilla, tamaños, violaciones
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpi.config import MANIFEST_PATH, PROCESSED_DIR, RAW_DIR, SEED  # noqa: E402
from tpi.data_contract import (  # noqa: E402
    ARCHIVO_TRAIN,
    ARCHIVO_ZIP,
    CLAVE_BUSQUEDA,
    COMPETICION,
    DTYPES_LECTURA,
    columnas_faltantes,
    validar_contrato,
)
from tpi.sampling import muestrear_busquedas  # noqa: E402

RUTA_ZIP = RAW_DIR / ARCHIVO_ZIP
RUTA_TRAIN = RAW_DIR / ARCHIVO_TRAIN
RUTA_MUESTRA = PROCESSED_DIR / "expedia_muestra.parquet"
RUTA_MANIFIESTO = MANIFEST_PATH
CHUNK = 1_000_000
ZIP_DEFLATE64 = 9


def descargar() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not RUTA_ZIP.exists():
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"[..] Descargando {ARCHIVO_ZIP} de '{COMPETICION}' a {RAW_DIR} (~435 MB)...")
        api.competition_download_file(COMPETICION, ARCHIVO_ZIP, path=str(RAW_DIR), unzip=False)
    else:
        print(f"[ok] {RUTA_ZIP} ya existe; no se descarga de nuevo.")
    if RUTA_TRAIN.exists():
        print(f"[ok] {RUTA_TRAIN} ya existe; no se extrae de nuevo.")
        return
    extraer_miembro(RUTA_ZIP, ARCHIVO_TRAIN, RUTA_TRAIN)


def extraer_miembro(ruta_zip: Path, nombre: str, destino: Path) -> None:
    """Extrae un único archivo del zip, verificando tamaño y CRC-32.

    Kaggle comprime `data.zip` con Deflate64 (método 9), que `zipfile` no implementa. En ese caso
    descomprimimos el flujo crudo con `inflate64` (wheels para Windows/Linux/macOS, sin compilar).
    """
    with zipfile.ZipFile(ruta_zip) as z:
        miembros = [m for m in z.infolist() if Path(m.filename).name == nombre]
        if len(miembros) != 1:
            raise FileNotFoundError(f"{nombre} no está en {ruta_zip.name}: {z.namelist()}")
        info = miembros[0]
    print(f"[..] Extrayendo solo {info.filename} (método {info.compress_type})...")
    tmp = destino.with_suffix(destino.suffix + ".part")
    crc, n = 0, 0
    with open(ruta_zip, "rb") as f, open(tmp, "wb") as out:
        if info.compress_type == ZIP_DEFLATE64:
            import inflate64

            f.seek(info.header_offset)
            cabecera = f.read(30)
            largo_nombre, largo_extra = struct.unpack("<HH", cabecera[26:30])
            f.seek(info.header_offset + 30 + largo_nombre + largo_extra)
            restante = info.compress_size
            inflador = inflate64.Inflater()
            while restante > 0:
                bloque = f.read(min(2**22, restante))
                restante -= len(bloque)
                datos = inflador.inflate(bloque)
                crc, n = zlib.crc32(datos, crc), n + len(datos)
                out.write(datos)
        else:
            with zipfile.ZipFile(f) as z, z.open(info) as origen:
                while datos := origen.read(2**22):
                    crc, n = zlib.crc32(datos, crc), n + len(datos)
                    out.write(datos)
    if n != info.file_size or crc != info.CRC:
        tmp.unlink()
        raise OSError(
            f"Extracción corrupta de {nombre}: {n} bytes, CRC {crc:08x} != {info.CRC:08x}"
        )
    tmp.replace(destino)


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
        "fuente": f"kaggle competitions: {COMPETICION}/{ARCHIVO_ZIP} -> {ARCHIVO_TRAIN}",
        "bytes_data_zip": RUTA_ZIP.stat().st_size if RUTA_ZIP.exists() else None,
        "sha256_data_zip": sha256(RUTA_ZIP) if RUTA_ZIP.exists() else None,
        "bytes_train_csv": RUTA_TRAIN.stat().st_size,
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
