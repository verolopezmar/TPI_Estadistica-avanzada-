"""Plan B: descarga Online Shoppers Purchasing Intention (UCI #468, CC BY 4.0).

Disponible pero NO activo: Verónica decidió (12/11) seguir con Expedia (ver docs/plan_b.md). Licencia
abierta: los datos se podrían versionar, pero por consistencia seguimos la misma política que con
Expedia (crudo fuera del repo, muestra ignorada por git, manifiesto versionado).

Uso:  python data/download_plan_b.py      y luego   $env:TPI_DATASET = "online_shoppers"
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpi.config import DATA_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

URL = (
    "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
)
RUTA_ZIP = RAW_DIR / "online_shoppers.zip"
RUTA_MUESTRA = PROCESSED_DIR / "online_shoppers.parquet"
RUTA_MANIFIESTO = DATA_DIR / "manifiesto_plan_b.json"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not RUTA_ZIP.exists():
        print(f"[..] Descargando {URL}")
        urllib.request.urlretrieve(URL, RUTA_ZIP)  # noqa: S310 - URL fija y https
    with zipfile.ZipFile(RUTA_ZIP) as z:
        df = pd.read_csv(z.open("online_shoppers_intention.csv"))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RUTA_MUESTRA, index=False)
    manifiesto = {
        "fuente": URL,
        "licencia": "CC BY 4.0",
        "cita": "Sakar, C. & Kastro, Y. (2018). UCI ML Repository. doi:10.24432/C5F88Q",
        "sha256_zip": hashlib.sha256(RUTA_ZIP.read_bytes()).hexdigest(),
        "n_filas": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "revenue_positivos": int(df["Revenue"].sum()),
    }
    RUTA_MANIFIESTO.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(manifiesto, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
