"""Artefactos agregados para la app (Fase 5). Nada de registros individuales.

Reglas de confidencialidad (decisión de Verónica del 12/11: seguir con Expedia publicando solo
agregados; reglas propuestas por Claude Code):
* Toda tabla de conteos suprime celdas con menos de `K_MIN` filas (se informan como 0 y se marca la
  supresión). Así, ningún valor publicado describe a menos de K_MIN hoteles-búsqueda.
* Nunca se exportan identificadores (`srch_id`, `prop_id`, `site_id`, países, destino) ni fechas.
* El modelo de la app es un Random Forest compacto con hojas de al menos `MIN_HOJA` filas: cada
  predicción promedia, como mínimo, MIN_HOJA observaciones de entrenamiento.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

K_MIN = 10
MIN_HOJA = 100
IDENTIFICADORES = (
    "srch_id",
    "prop_id",
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "srch_destination_id",
    "date_time",
)


def suprimir(conteos: np.ndarray, k: int = K_MIN) -> tuple[list, int]:
    c = np.asarray(conteos, dtype=int).copy()
    n_sup = int(((c > 0) & (c < k)).sum())
    c[(c > 0) & (c < k)] = 0
    return c.tolist(), n_sup


def histograma(serie: pd.Series, bordes: np.ndarray) -> dict:
    conteos, _ = np.histogram(serie.dropna(), bins=bordes)
    c, n_sup = suprimir(conteos)
    return {"bordes": [float(b) for b in bordes], "conteos": c, "celdas_suprimidas": n_sup}


def grilla_2d(x: np.ndarray, y: np.ndarray, bordes_x: np.ndarray, bordes_y: np.ndarray) -> dict:
    h, _, _ = np.histogram2d(x, y, bins=[bordes_x, bordes_y])
    c, n_sup = suprimir(h.ravel())
    return {
        "bordes_x": [float(b) for b in bordes_x],
        "bordes_y": [float(b) for b in bordes_y],
        "conteos": np.array(c).reshape(h.shape).tolist(),
        "celdas_suprimidas": n_sup,
    }


class _Codificador(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer | np.bool_):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def guardar_json(obj: dict, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=1, cls=_Codificador), "utf-8")
    return ruta


def claves_prohibidas(obj, ruta: str = "") -> list[str]:
    """Busca identificadores en las claves de un artefacto JSON (recursivo)."""
    halladas = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in IDENTIFICADORES:
                halladas.append(f"{ruta}/{k}")
            halladas += claves_prohibidas(v, f"{ruta}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            halladas += claves_prohibidas(v, f"{ruta}[{i}]")
    return halladas
