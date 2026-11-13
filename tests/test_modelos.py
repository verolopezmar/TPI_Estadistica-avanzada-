"""Fase 3: sin fuga de información, validación agrupada y métricas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tpi.config import SEED
from tpi.modelos import (
    FEATURES,
    PROHIBIDAS,
    acierto_top_k,
    cv_agrupado,
    matriz,
    metricas,
    submuestra_busquedas,
    umbral_f1,
)

rng = np.random.default_rng(SEED)


@pytest.mark.leakage
def test_features_no_incluyen_variables_con_fuga():
    assert not set(FEATURES) & set(PROHIBIDAS)


@pytest.mark.leakage
def test_matriz_rechaza_fuga_explicita():
    df = pd.DataFrame({"click_bool": [0, 1], "position": [1, 2]})
    with pytest.raises(ValueError, match="fuga"):
        matriz(df, ["position", "click_bool"])


def test_matriz_completa_saldo_sin_competidores_con_cero():
    df = pd.DataFrame({"comp_saldo": [np.nan, 0.5], "comp_n_info": [0, 2]})
    X = matriz(df, ["comp_saldo", "comp_n_info"])
    assert X["comp_saldo"].tolist() == [0.0, 0.5]


@pytest.mark.leakage
def test_cv_agrupado_no_parte_busquedas():
    grupos = np.repeat(np.arange(400), 10)
    y = (rng.uniform(size=len(grupos)) < 0.05).astype(int)
    for tr, va in cv_agrupado().split(np.zeros(len(y)), y, grupos):
        assert set(grupos[tr]).isdisjoint(set(grupos[va]))


def test_submuestra_conserva_busquedas_completas():
    df = pd.DataFrame({"srch_id": np.repeat(np.arange(100), 7)})
    sub = submuestra_busquedas(df, 0.3)
    assert sub.groupby("srch_id").size().eq(7).all() and sub["srch_id"].nunique() == 30


def test_metricas_y_umbral():
    y = np.r_[np.zeros(900), np.ones(100)].astype(int)
    p = np.r_[rng.uniform(0, 0.6, 900), rng.uniform(0.4, 1, 100)]
    u = umbral_f1(y, p)
    m = metricas(y, p, u)
    assert 0.4 <= u <= 0.6 and m["ROC_AUC"] > 0.9 and 0 <= m["Brier"] <= 1


def test_acierto_top_k():
    df = pd.DataFrame({"srch_id": [1, 1, 1, 2, 2, 2], "booking_bool": [0, 1, 0, 0, 0, 0]})
    assert acierto_top_k(df, pd.Series([0.1, 0.9, 0.2, 1, 1, 1]), k=1) == 1.0
    assert acierto_top_k(df, pd.Series([0.9, 0.1, 0.2, 1, 1, 1]), k=1) == 0.0
