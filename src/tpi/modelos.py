"""Fase 3: torneo predictivo (P(reserva | X)) con validación agrupada por búsqueda.

Decisiones (propuestas por Claude Code; ver docs/ia_log.md):
* Unidad: fila búsqueda × hotel; Y = `booking_bool` (prevalencia ≈ 2 %).
* Métrica de selección: PR-AUC (`average_precision`), adecuada para clase positiva rara; se
  reportan además ROC-AUC, Brier, precisión, recall y F1.
* Validación: `StratifiedGroupKFold(5)` por `srch_id` dentro de Train (ninguna lista se parte
  entre pliegues).
* Sin fuga: se excluyen variables posteriores a la exposición o derivadas de la respuesta
  (`click_bool`, `gross_bookings_usd`, `booking_bool`). `position` se conoce al mostrar la lista y
  sí es predictor legítimo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from tpi.config import SEED

OBJETIVO = "booking_bool"
PROHIBIDAS = ("click_bool", "gross_bookings_usd", "booking_bool")
FEATURES = [
    # hotel
    "prop_starrating",
    "prop_review_score",
    "sin_resenas",
    "prop_location_score1",
    "prop_location_score2",
    "prop_location_score2_faltante",
    "prop_log_historical_price",
    "sin_historial_precio",
    "prop_brand_bool",
    "promotion_flag",
    "log_precio",
    "log_precio_rel",
    "comp_n_info",
    "comp_saldo",
    "slot_especial",
    # búsqueda / usuario
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "usuario_recurrente",
    "con_afinidad",
    "orig_destination_distance",
    "orig_destination_distance_faltante",
    "random_bool",
    # exposición
    "position",
]


def matriz(df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Matriz de diseño numérica sin NaN. `comp_saldo` es NaN si no hay competidores con dato:
    se completa con 0 (neutro) porque `comp_n_info = 0` ya codifica la ausencia."""
    fuga = set(features) & set(PROHIBIDAS)
    if fuga:
        raise ValueError(f"Variables con fuga de información en X: {sorted(fuga)}")
    X = df[features].astype(float).copy()
    if "comp_saldo" in X:
        X["comp_saldo"] = X["comp_saldo"].fillna(0.0)
    if X.isna().any().any():
        raise ValueError(f"NaN en X: {X.columns[X.isna().any()].tolist()}")
    return X


def cv_agrupado(n_splits: int = 5, seed: int = SEED) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def submuestra_busquedas(df: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    """Búsquedas completas al azar (para búsqueda de hiperparámetros con cómputo acotado)."""
    ids = df["srch_id"].drop_duplicates().sort_values()
    elegidas = ids.sample(frac=frac, random_state=seed)
    return df[df["srch_id"].isin(set(elegidas))]


def umbral_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Umbral que maximiza F1 de la clase positiva (se elige en Train, nunca en Test)."""
    prec, rec, umb = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(umb[np.nanargmax(f1[:-1])])


def metricas(y: np.ndarray, p: np.ndarray, umbral: float) -> dict[str, float]:
    yhat = (p >= umbral).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat),
        "F1": f1_score(y, yhat),
        "F1_macro": f1_score(y, yhat, average="macro"),
        "F1_ponderado": f1_score(y, yhat, average="weighted"),
        "umbral": umbral,
    }


def acierto_top_k(df: pd.DataFrame, score: pd.Series, k: int = 5) -> float:
    """Entre búsquedas con reserva: fracción en la que el hotel reservado está entre los k de
    mayor `score` de su lista (métrica de ranking, extra al enunciado)."""
    d = df.assign(_s=score.to_numpy())
    d = d[d.groupby("srch_id")[OBJETIVO].transform("max") == 1]
    rango = d.groupby("srch_id")["_s"].rank(ascending=False, method="first")
    return float((rango[d[OBJETIVO] == 1] <= k).mean())
