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


def _listas_una_eleccion(n_listas=1500, n_hot=20, efecto=0.7, seed=3):
    """Cada lista tiene exactamente una elección (como las reservas de Expedia)."""
    rng = np.random.default_rng(seed)
    filas = []
    for s in range(n_listas):
        t = (rng.permutation(n_hot) < 5).astype(int)
        q = rng.normal(size=n_hot)
        u = efecto * t + 0.5 * q + rng.gumbel(size=n_hot)
        y = (u == u.max()).astype(int)
        filas.append(pd.DataFrame({"srch_id": s, "T": t, "q": q, "Y": y}))
    return pd.concat(filas, ignore_index=True)


def test_eleccion_logit_coincide_con_statsmodels():
    from statsmodels.discrete.conditional_models import ConditionalLogit

    from tpi.causal import eleccion_logit

    d = _listas_una_eleccion()
    rapido = eleccion_logit(d, "T", "Y", ["q"])
    X = d[["T", "q"]].astype(float).assign(q=lambda x: (x.q - x.q.mean()) / x.q.std())
    ref = ConditionalLogit(d.Y, X, groups=d.srch_id).fit(disp=0, method="newton")
    assert rapido.params["T"] == pytest.approx(ref.params["T"], rel=1e-6)
    assert abs(rapido.params["T"] - 0.7) < 3 * rapido.ee["T"]


def test_peso_dos_equivale_a_duplicar_la_busqueda():
    from tpi.causal import eleccion_logit

    d = _listas_una_eleccion(n_listas=300)
    w = pd.Series(1.0, index=d.srch_id.unique())
    w.iloc[:50] = 2.0
    dup = pd.concat([d, d[d.srch_id < 50].assign(srch_id=lambda x: x.srch_id + 10_000)])
    a = eleccion_logit(d, "T", "Y", ["q"], pesos_grupo=w)
    b = eleccion_logit(dup, "T", "Y", ["q"])
    assert a.params["T"] == pytest.approx(b.params["T"], rel=1e-5)


def test_efecto_marginal_de_eleccion_recupera_la_verdad():
    from scipy.special import softmax

    from tpi.causal import efecto_marginal_eleccion

    d = _listas_una_eleccion(n_listas=3000, efecto=0.7)
    # verdad: con utilidades conocidas, P(elegido) con y sin top-5 manteniendo la lista
    reales = []
    for _, g in d.groupby("srch_id"):
        v = 0.7 * g["T"].to_numpy() + 0.5 * g["q"].to_numpy()
        for i in range(len(g)):
            v1, v0 = v.copy(), v.copy()
            v1[i] = 0.5 * g["q"].iloc[i] + 0.7
            v0[i] = 0.5 * g["q"].iloc[i]
            reales.append(softmax(v1)[i] - softmax(v0)[i])
    assert efecto_marginal_eleccion(d, "T", "Y", ["q"]) == pytest.approx(np.mean(reales), abs=0.01)


def test_remuestreo_por_busqueda_conserva_listas_completas():
    from tpi.causal import _remuestrear

    d = _listas_una_eleccion(n_listas=50)
    r = _remuestrear(d, np.random.default_rng(0))
    assert r["srch_id"].nunique() == 50
    assert r.groupby("srch_id").size().eq(20).all() and r.groupby("srch_id")["Y"].sum().eq(1).all()
