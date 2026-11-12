"""Fase 4: los estimadores recuperan la verdad en simulaciones y las trampas se ven donde deben."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tpi.causal import (
    efecto_marginal_real,
    logit_condicional,
    lpm_efectos_fijos,
    simular_colisionador,
    simular_listas,
)

EFECTO = 0.8


@pytest.fixture(scope="module")
def listas() -> pd.DataFrame:
    return simular_listas(n_listas=3000, efecto_logit=EFECTO)


def test_logit_condicional_recupera_el_or_verdadero(listas):
    e = logit_condicional(listas, "T", "Y", ["calidad"])
    assert e.ic95[0] < np.exp(EFECTO) < e.ic95[1]


def test_or_condicional_sobrevive_al_filtro_de_seleccion(listas):
    """Amenaza A (Entrada 4): quedarse solo con listas con ≥ 1 evento no cambia el OR intra-lista."""
    s = listas[listas.groupby("srch_id")["Y"].transform("max") == 1]
    e = logit_condicional(s, "T", "Y", ["calidad"])
    assert e.ic95[0] < np.exp(EFECTO) < e.ic95[1]


def test_diferencia_absoluta_se_infla_con_el_filtro(listas):
    """En cambio, la diferencia de probabilidad sí se infla al condicionar en S."""
    real = efecto_marginal_real(listas, EFECTO)
    todo = lpm_efectos_fijos(listas, "T", "Y", ["calidad"])
    s = listas[listas.groupby("srch_id")["Y"].transform("max") == 1]
    filtrado = lpm_efectos_fijos(s, "T", "Y", ["calidad"])
    assert todo.ic95[0] < real < todo.ic95[1]
    assert filtrado.estimacion > 1.5 * real


def test_condicionar_en_colisionador_induce_berkson():
    d = simular_colisionador()
    assert abs(np.corrcoef(d["T"], d["Zh"])[0, 1]) < 0.02
    sel = d[d["C"] == 1]
    assert np.corrcoef(sel["T"], sel["Zh"])[0, 1] < -0.05
    ingenuo = d.groupby("T")["Y"].mean().diff().iloc[-1]
    sesgado = sel.groupby("T")["Y"].mean().diff().iloc[-1]
    assert sesgado < 0.5 * ingenuo  # condicionar en C sesga hacia abajo (incluso cambia el signo)


def test_dowhy_identifica_backdoor_sin_mediador_y_recupera_el_efecto():
    """Error 26 del ia_log: dowhy 0.12 rompe con networkx >= 3.5 al identificar. Este test lo
    detecta (un test de imports no alcanza) y verifica que el mediador no entra al ajuste."""
    import logging

    from tpi.causal import modelo_dowhy

    logging.disable(logging.WARNING)
    rng = np.random.default_rng(0)
    n = 20_000
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(z1 - z2)))).astype(int)
    m = (rng.uniform(size=n) < 0.2 + 0.3 * t).astype(int)
    y = 0.5 * m + 0.3 * z1 + 0.2 * z2 + rng.normal(size=n) * 0.1  # efecto total = 0,15
    d = pd.DataFrame({"T": t, "Y": y, "z1": z1, "z2": z2, "M": m})
    modelo, estimando = modelo_dowhy(d, "T", "Y", ["z1", "z2"], "M")
    assert set(estimando.get_backdoor_variables()) == {"z1", "z2"}
    e = modelo.estimate_effect(estimando, "backdoor.linear_regression")
    assert e.value == pytest.approx(0.15, abs=0.02)
