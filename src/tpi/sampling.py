"""Muestreo de búsquedas completas, estratificado por `random_bool`.

La unidad de muestreo es la **búsqueda** (`srch_id`), no la fila búsqueda×hotel: los hoteles de
una misma lista compiten entre sí y comparten usuario y contexto, así que sus filas no son
independientes. Muestrear filas sueltas rompería las listas, y en la Fase 3 filtraría
información entre Train y Test (por eso usamos particiones agrupadas por `srch_id`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tpi.config import SEED


def muestrear_busquedas(
    busquedas: pd.DataFrame,
    n_por_estrato: int,
    seed: int = SEED,
    col_id: str = "srch_id",
    col_estrato: str = "random_bool",
) -> np.ndarray:
    """Elige `n_por_estrato` búsquedas por valor de `col_estrato`, sin reemplazo.

    `busquedas` tiene una fila por búsqueda (columnas `col_id`, `col_estrato`). Si un estrato
    tiene menos búsquedas que `n_por_estrato`, se toman todas. Devuelve los ids ordenados, de
    modo que el resultado depende solo de la semilla y no del orden de entrada.
    """
    if busquedas[col_id].duplicated().any():
        raise ValueError("`busquedas` debe tener una fila por búsqueda")
    rng = np.random.default_rng(seed)
    elegidos = []
    for _, grupo in busquedas.sort_values(col_id).groupby(col_estrato, sort=True):
        ids = grupo[col_id].to_numpy()
        k = min(n_por_estrato, len(ids))
        elegidos.append(rng.choice(ids, size=k, replace=False))
    return np.sort(np.concatenate(elegidos))
