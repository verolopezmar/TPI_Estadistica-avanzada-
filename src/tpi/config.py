"""Configuración global: semilla y rutas relativas a la raíz del repositorio.

Todas las rutas se derivan de la ubicación de este archivo, de modo que el código
funciona igual en Windows, Linux (CI) y Streamlit Community Cloud.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED: int = 42
ALPHA: float = 0.05  # nivel de significancia fijado a priori para todo el TPI

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT / "models"
FIGURES_DIR: Path = ROOT / "report" / "figures"


def set_global_seed(seed: int = SEED) -> np.random.Generator:
    """Fija la semilla en `random`, `numpy` y PYTHONHASHSEED; devuelve un Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - algunas librerías (dowhy) usan el estado global
    return np.random.default_rng(seed)
