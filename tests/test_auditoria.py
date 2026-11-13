"""Auditoría forense: cada regla detecta lo que dice detectar (datos sintéticos con verdad conocida)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tpi.auditoria import (
    agregar_competidores,
    aplicar_centinelas,
    contraste_mcar,
    diagnostico_escala_precio,
    imputar_con_indicador,
    precio_relativo,
)
from tpi.config import SEED

rng = np.random.default_rng(SEED)


def test_centinelas_pasan_a_nan_con_indicador_y_cuantifican_el_sesgo():
    df = pd.DataFrame(
        {
            "prop_review_score": [0.0, 4.0, 4.0, 5.0],
            "prop_starrating": [3.0, 0.0, 4.0, 5.0],
            "prop_log_historical_price": [0.0, 0.0, 5.0, 5.0],
        }
    )
    out, tabla = aplicar_centinelas(df)
    assert out["prop_review_score"].isna().tolist() == [True, False, False, False]
    assert out["sin_resenas"].tolist() == [1, 0, 0, 0]
    assert out["sin_historial_precio"].sum() == 2
    # media con centinela 2,5 vs. sin centinela 5 -> sesgo de -50 %
    assert tabla.loc["prop_log_historical_price", "sesgo_media_pct"] == pytest.approx(-50.0)


def _busquedas(n_busq=200, n_hot=20, escala=1.0):
    filas = []
    for s in range(n_busq):
        base = rng.lognormal(4.8, 0.4)
        for _ in range(n_hot):
            filas.append({"srch_id": s, "price_usd": base * rng.lognormal(0, 0.3) * escala})
    return pd.DataFrame(filas)


def test_precio_relativo_es_invariante_a_la_escala_de_la_busqueda():
    df = _busquedas()
    a = precio_relativo(df)["log_precio_rel"]
    df2 = df.copy()
    df2.loc[df2["srch_id"] < 50, "price_usd"] *= 4  # 50 búsquedas "por estadía de 4 noches"
    b = precio_relativo(df2)["log_precio_rel"]
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_outlier_absoluto_detecta_busquedas_enteras_imposibles():
    """Caso real de la muestra: una búsqueda con precios de millones; la regla relativa sola no
    la ve porque la mediana de la búsqueda también es absurda."""
    df = _busquedas()
    df.loc[df["srch_id"] == 0, "price_usd"] = 4_000_000.0
    out = precio_relativo(df)
    assert out.loc[df["srch_id"] == 0, "precio_outlier"].all()
    assert out.loc[df["srch_id"] != 0, "precio_outlier"].mean() < 0.01


def test_outlier_relativo_detecta_un_hotel_desfasado_en_su_lista():
    df = _busquedas()
    en_0 = df["srch_id"] == 0
    df.loc[en_0, "price_usd"] = 100.0
    idx = df.index[en_0][0]
    df.loc[idx, "price_usd"] = 1500.0  # 15 veces la mediana de su lista
    out = precio_relativo(df.assign(price_usd=df["price_usd"]))
    assert out.loc[idx, "precio_outlier"] == 1
    assert out.loc[en_0 & (df.index != idx), "precio_outlier"].sum() == 0


def test_diagnostico_de_escala_clasifica_por_noche_y_por_estadia():
    df = pd.DataFrame(
        {
            "booking_bool": [1, 1, 1],
            "srch_length_of_stay": [3, 3, 3],
            "srch_room_count": [1, 1, 1],
            "price_usd": [100.0, 330.0, 100.0],
            "gross_bookings_usd": [340.0, 370.0, 50.0],  # por noche, por estadía, incompatible
        }
    )
    assert diagnostico_escala_precio(df)["escala"].tolist() == [
        "por_noche",
        "por_estadia",
        "incompatible",
    ]


def test_contraste_mcar_no_detecta_dependencia_si_el_faltante_es_aleatorio():
    n = 20_000
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "b": rng.integers(0, 2, n), "y": rng.normal(size=n)}
    )
    df.loc[rng.uniform(size=n) < 0.3, "y"] = np.nan  # MCAR por construcción
    t = contraste_mcar(df, "y", ["x", "b"])
    assert not t["dependencia_apreciable"].any()


def test_contraste_mcar_detecta_faltante_mar():
    n = 20_000
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": rng.normal(size=n)})
    df.loc[x > 0.5, "y"] = np.nan  # MAR: depende de x observada
    t = contraste_mcar(df, "y", ["x"])
    assert t["dependencia_apreciable"].all() and t["p_bonferroni"].iloc[0] < 1e-10


def test_imputacion_por_grupo_con_indicador():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "c"], "v": [1.0, np.nan, 10.0, np.nan, np.nan]})
    out = imputar_con_indicador(df, "v", por="g")
    assert out["v"].tolist() == [1.0, 1.0, 10.0, 10.0, 5.5]  # c sin datos: mediana global
    assert out["v_faltante"].tolist() == [0, 1, 0, 1, 1]


def test_agregado_de_competidores_ignora_diferencias_imposibles():
    fila = {f"comp{i}_rate": np.nan for i in range(1, 9)}
    fila |= {f"comp{i}_rate_percent_diff": np.nan for i in range(1, 9)}
    fila |= {"comp2_rate": 1, "comp3_rate": -1, "comp5_rate": 1}
    fila |= {"comp2_rate_percent_diff": 12.0, "comp5_rate_percent_diff": 269_379.0}
    out = agregar_competidores(pd.DataFrame([fila]))
    assert out.loc[0, "comp_n_info"] == 3
    assert out.loc[0, "comp_saldo"] == pytest.approx(1 / 3)
    assert out.loc[0, "comp_diff_max"] == 12.0
