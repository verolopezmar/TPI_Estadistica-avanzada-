"""Fase 2: partición sin fuga, diagnósticos de Gauss-Markov, influencia, VIF y PCA solo en Train."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from tpi.config import SEED
from tpi.particion import busquedas_de_test, es_train
from tpi.regresion import (
    ajustar_pca,
    diagnostico_gauss_markov,
    influencia,
    resumen_pca,
    tabla_vif,
)

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------------------- partición
@pytest.fixture(scope="module")
def filas() -> pd.DataFrame:
    n_busq = 3000
    ids = np.repeat(np.arange(n_busq), 10)
    return pd.DataFrame({"srch_id": ids, "random_bool": (ids % 3 == 0).astype(int)})


def test_particion_agrupada_sin_busquedas_compartidas(filas):
    m = es_train(filas)
    assert set(filas.loc[m, "srch_id"]).isdisjoint(set(filas.loc[~m, "srch_id"]))


def test_particion_80_20_estratificada_y_determinista(filas):
    test = busquedas_de_test(filas)
    assert test == busquedas_de_test(filas.sample(frac=1, random_state=1))  # no depende del orden
    busq = filas.drop_duplicates("srch_id").set_index("srch_id")
    frac = busq.assign(t=busq.index.isin(test)).groupby("random_bool")["t"].mean()
    assert frac.between(0.195, 0.205).all()


# --------------------------------------------------------------------------- Gauss-Markov
def _datos(n=3000, hetero=False):
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    e = rng.normal(size=n) * (np.exp(x1) if hetero else 1.0)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 1 + 2 * x1 - x2 + e})


def test_breusch_pagan_detecta_heterocedasticidad_y_no_da_falsos_positivos():
    hom = diagnostico_gauss_markov(smf.ols("y ~ x1 + x2", _datos()).fit())
    het = diagnostico_gauss_markov(smf.ols("y ~ x1 + x2", _datos(hetero=True)).fit())
    assert hom.bp_p > 0.01 and het.bp_p < 1e-10
    assert 1.8 < hom.durbin_watson < 2.2


def test_influencia_marca_un_punto_plantado():
    d = _datos(500)
    d.loc[0, ["x1", "y"]] = [8.0, -40.0]  # alto leverage y residuo enorme
    inf = influencia(smf.ols("y ~ x1 + x2", d).fit())
    assert inf.loc[0, ["outlier_y", "alto_leverage", "influyente_4n", "influyente_05"]].all()
    assert inf["influyente_05"].sum() == 1


def test_vif_detecta_colinealidad():
    d = _datos(2000)
    d["x3"] = d["x1"] + rng.normal(scale=0.1, size=len(d))  # casi copia de x1
    vif = tabla_vif(d[["x1", "x2", "x3"]])
    assert vif["x3"] > 10 and vif["x2"] < 1.5


# ------------------------------------------------------------------------------------ PCA
def test_pca_se_ajusta_solo_con_train():
    """Regla estricta del enunciado: media, desvío y autovectores salen solo de Train."""
    d = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["a", "b", "c"])
    d.loc[800:, "a"] += 50  # Test con distribución corrida
    train, test = d.iloc[:800], d.iloc[800:]
    pipe = ajustar_pca(train, ["a", "b", "c"])
    np.testing.assert_allclose(pipe.named_steps["escala"].mean_, train.mean().to_numpy())
    antes = pipe.named_steps["pca"].components_.copy()
    z = pipe.transform(test[["a", "b", "c"]])  # rotación ciega: no reajusta
    np.testing.assert_array_equal(antes, pipe.named_steps["pca"].components_)
    assert z[:, :].mean(axis=0).__abs__().max() > 5  # el corrimiento de Test se ve, no se "absorbe"


def test_resumen_pca_varianza_suma_100():
    d = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    var, cargas = resumen_pca(ajustar_pca(d, list("abcd")), list("abcd"))
    assert var["pct_acum"].iloc[-1] == pytest.approx(100)
    np.testing.assert_allclose((cargas**2).sum(axis=0), 1.0)
