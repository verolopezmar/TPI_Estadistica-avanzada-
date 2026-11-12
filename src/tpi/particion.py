"""Partición Train/Test única para las Fases 2 a 4 (sin fuga de información).

Diseño:
* **Agrupada por búsqueda**: todas las filas de un `srch_id` caen del mismo lado. Si una lista se
  partiera, el modelo vería en Train a los competidores directos de un hotel de Test (mismo usuario,
  misma fecha, mismo destino) y la métrica de Test sería optimista.
* **Estratificada por `random_bool`**: Train y Test conservan la misma mezcla de orden algorítmico y
  aleatorio, que tienen tasas base muy distintas (Fase 0).
* **Aleatoria, no cronológica**: el período es de 8 meses (11/2012-06/2013) y la Fase 2 no encuentra
  autocorrelación temporal en los residuos (Durbin-Watson ≈ 2). El objetivo es generalizar a
  búsquedas nuevas del mismo período, no pronosticar a futuro (se declara como limitación).
* 80/20 con `SEED = 42`, como pide el enunciado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tpi.config import SEED

PROPORCION_TEST = 0.20


def busquedas_de_test(
    df: pd.DataFrame,
    proporcion: float = PROPORCION_TEST,
    seed: int = SEED,
    col_id: str = "srch_id",
    col_estrato: str = "random_bool",
) -> set:
    """Ids de búsqueda asignados a Test, estratificando por `col_estrato`."""
    rng = np.random.default_rng(seed)
    busq = df[[col_id, col_estrato]].drop_duplicates(col_id).sort_values(col_id)
    test = []
    for _, g in busq.groupby(col_estrato, sort=True):
        ids = g[col_id].to_numpy()
        k = int(round(proporcion * len(ids)))
        test.append(rng.choice(ids, size=k, replace=False))
    return set(np.concatenate(test).tolist())


def es_train(df: pd.DataFrame, col_id: str = "srch_id", **kw) -> pd.Series:
    """Máscara booleana por fila: True = Train."""
    test = busquedas_de_test(df, col_id=col_id, **kw)
    return ~df[col_id].isin(test)
