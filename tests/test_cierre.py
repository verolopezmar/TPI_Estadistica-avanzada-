"""Cierre (13/11): no-fuga de la preparación, test de Little, IPW con overlap y efectos fijos.

Cada test usa datos con verdad conocida y puede fallar en el sentido que importa.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tpi.auditoria import imputar_con_indicador, little_mcar, precio_relativo
from tpi.causal import (
    bootstrap_ipw,
    diagnostico_overlap,
    ipw_hajek,
    lpm_efectos_fijos,
)
from tpi.config import SEED
from tpi.contrato import contrato_activo

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------------- no-fuga (leakage)
@pytest.mark.leakage
def test_imputacion_usa_solo_medianas_de_train():
    df = pd.DataFrame(
        {
            "pais": ["A"] * 6 + ["B"] * 4,
            "x": [1.0, 2.0, 3.0, np.nan, 100.0, np.nan, 10.0, 20.0, np.nan, 900.0],
            "train": [True, True, True, True, False, False, True, True, True, False],
        }
    )
    out = imputar_con_indicador(df, "x", por="pais", ajuste=df["train"])
    # Mediana de A en Train = 2 (el 100 de Test no cuenta); de B en Train = 15 (el 900 no cuenta)
    assert out.loc[3, "x"] == 2.0 and out.loc[5, "x"] == 2.0 and out.loc[8, "x"] == 15.0
    assert out["x_faltante"].tolist() == [0, 0, 0, 1, 0, 1, 0, 0, 1, 0]


@pytest.mark.leakage
def test_imputacion_es_invariante_a_los_valores_de_test():
    n = 2000
    df = pd.DataFrame(
        {
            "g": rng.integers(0, 5, n),
            "x": np.where(rng.uniform(size=n) < 0.2, np.nan, rng.normal(size=n)),
            "train": rng.uniform(size=n) < 0.8,
        }
    )
    otro = df.copy()
    otro.loc[~otro.train, "x"] = otro.loc[~otro.train, "x"] * 1000 + 5  # Test totalmente distinto
    a = imputar_con_indicador(df, "x", por="g", ajuste=df["train"])
    b = imputar_con_indicador(otro, "x", por="g", ajuste=otro["train"])
    pd.testing.assert_series_equal(a.loc[a.train, "x"], b.loc[b.train, "x"])


@pytest.mark.leakage
def test_grupo_ausente_en_train_recibe_la_mediana_global_de_train():
    df = pd.DataFrame({"g": ["A", "A", "C"], "x": [1.0, 3.0, np.nan], "train": [True, True, False]})
    out = imputar_con_indicador(df, "x", por="g", ajuste=df["train"])
    assert out.loc[2, "x"] == 2.0


@pytest.mark.leakage
def test_cercas_y_winsorizacion_del_precio_salen_de_train():
    n = 4000
    ids = np.repeat(np.arange(n // 20), 20)
    df = pd.DataFrame({"srch_id": ids, "price_usd": np.exp(rng.normal(4.5, 0.5, n))})
    tr = pd.Series(df["srch_id"] < 160, index=df.index)
    otro = df.copy()
    otro.loc[~tr, "price_usd"] *= 1e4  # precios absurdos solo en Test
    a = precio_relativo(df, ajuste=tr)
    b = precio_relativo(otro, ajuste=tr)
    pd.testing.assert_frame_equal(a.loc[tr], b.loc[tr])


@pytest.mark.data
@pytest.mark.leakage
def test_preparar_no_fuga_informacion_de_test_a_train():
    """Sobre la muestra real: perturbar Test no cambia ninguna fila de Train."""
    contrato = contrato_activo()
    try:
        crudo = contrato.cargar()
    except FileNotFoundError:
        pytest.skip("Sin datos locales")
    from tpi.particion import es_train
    from tpi.preparacion import preparar

    crudo = crudo[crudo["srch_id"].isin(crudo["srch_id"].drop_duplicates().iloc[:3000])]
    crudo = crudo.reset_index(drop=True)
    te = ~es_train(crudo).to_numpy()
    perturbado = crudo.copy()
    for c in ["price_usd", "prop_location_score2", "prop_log_historical_price"]:
        perturbado.loc[te, c] = perturbado.loc[te, c] * 7 + 3
    a, b = preparar(crudo), preparar(perturbado)
    pd.testing.assert_frame_equal(a.loc[~te], b.loc[~te])


# ------------------------------------------------------------------------ test de Little (MCAR)
def _normal_correlacionada(n):
    cov = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    return pd.DataFrame(rng.multivariate_normal(np.zeros(3), cov, n), columns=list("abc"))


def test_little_no_rechaza_con_faltantes_mcar():
    d = _normal_correlacionada(3000)
    d.loc[rng.uniform(size=len(d)) < 0.3, "c"] = np.nan
    d.loc[rng.uniform(size=len(d)) < 0.2, "b"] = np.nan
    r = little_mcar(d, list("abc"))
    assert r["p_valor"] > 0.01 and r["gl"] > 0


def test_little_rechaza_con_faltantes_mar():
    d = _normal_correlacionada(3000)
    d.loc[d["a"] > 0.3, "c"] = np.nan  # la falta de c depende de a (observada): MAR
    r = little_mcar(d, list("abc"))
    assert r["p_valor"] < 1e-6


def test_little_sin_faltantes_tiene_cero_grados_de_libertad():
    r = little_mcar(_normal_correlacionada(200), list("abc"))
    assert r["gl"] == 0 and r["patrones"] == 1


# ------------------------------------------------------------------------- IPW y positividad
def _confundido(n=40_000, fuerza=1.5, tau=0.10):
    z = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-fuerza * z))
    t = rng.uniform(size=n) < ps
    y = 0.2 + tau * t + 0.15 * z + rng.normal(0, 0.1, n)
    return y, t.astype(int), ps, z


def test_ipw_hajek_recupera_el_ate_que_el_ingenuo_sesga():
    y, t, ps, _ = _confundido()
    ingenuo = y[t == 1].mean() - y[t == 0].mean()
    assert abs(ingenuo - 0.10) > 0.05  # el ingenuo está sesgado por z
    assert ipw_hajek(y, t, ps) == pytest.approx(0.10, abs=0.01)


def test_excluir_sin_overlap_no_sesga_y_truncar_si():
    """Con efecto homogéneo, excluir las unidades fuera de [0,01; 0,99] mantiene el ATE; truncar
    la propensión deja pesos incorrectos y sesga (por eso el recorte principal es "excluir")."""
    y, t, ps, _ = _confundido(fuerza=4.0)
    excluido = ipw_hajek(y, t, ps, limites=(0.01, 0.99))
    truncado = ipw_hajek(y, t, ps, limites=(0.01, 0.99), modo="truncar")
    assert excluido == pytest.approx(0.10, abs=0.015)
    assert abs(truncado - 0.10) > 2 * abs(excluido - 0.10)
    with pytest.raises(ValueError):
        ipw_hajek(y, t, ps, limites=(0.01, 0.99), modo="otro")


def test_overlap_detecta_falta_de_positividad():
    _, t_bien, ps_bien, _ = _confundido(fuerza=0.5)
    _, t_mal, ps_mal, _ = _confundido(fuerza=6.0)
    bien, mal = diagnostico_overlap(ps_bien, t_bien), diagnostico_overlap(ps_mal, t_mal)
    assert bien["pct_fuera_T1"] < 1 and mal["pct_fuera_T1"] > 5
    assert mal["peso_max"] > bien["peso_max"]
    assert mal["ess_T0"] / mal["n_T0"] < bien["ess_T0"] / bien["n_T0"]


def test_bootstrap_ipw_por_cluster_cubre_el_valor_verdadero():
    y, t, ps, _ = _confundido(n=20_000)
    grupos = np.repeat(np.arange(2000), 10)
    r = bootstrap_ipw(y, t, ps, grupos, B=300)
    assert r["IC95_inf"] < 0.10 < r["IC95_sup"]
    assert r["ATE"] == pytest.approx(ipw_hajek(y, t, ps))


# ------------------------------------------------------- efectos fijos sin matriz singular
def test_efectos_fijos_descarta_covariables_constantes_en_el_grupo():
    n_g, k = 500, 20
    g = np.repeat(np.arange(n_g), k)
    contexto = np.repeat(rng.normal(size=n_g), k)  # constante dentro de la búsqueda
    x = rng.normal(size=n_g * k)
    t = (rng.uniform(size=n_g * k) < 0.3).astype(int)
    y = 0.05 * t + 0.02 * x + 0.3 * contexto + rng.normal(0, 0.1, n_g * k)
    d = pd.DataFrame({"srch_id": g, "y": y, "t": t, "x": x, "contexto": contexto})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # un SingularMatrixWarning haría fallar el test
        r = lpm_efectos_fijos(d, "t", "y", ["x", "contexto"])
    sin = lpm_efectos_fijos(d, "t", "y", ["x"])
    assert r.estimacion == pytest.approx(sin.estimacion) and r.estimacion == pytest.approx(
        0.05, abs=0.01
    )


# ------------------------------------------------------------------- deploy reproducible de la app
def _pins(ruta):
    pins = {}
    for linea in ruta.read_text("utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea:
            nombre, _, version = linea.partition("==")
            pins[nombre.lower()] = version
    return pins


def test_requirements_de_la_app_fijados_y_coherentes_con_la_raiz():
    """Streamlit Cloud y Docker instalan app/requirements.txt: todo paquete va fijado, con la misma
    versión que el entorno donde se serializó el modelo, y pyarrow incluido (sin él, pip trae una
    versión que exige numpy>=2 y st.dataframe falla)."""
    from tpi.config import ROOT

    app, raiz = _pins(ROOT / "app" / "requirements.txt"), _pins(ROOT / "requirements.txt")
    assert all(v for v in app.values()), "dependencia de la app sin versión fijada"
    assert "pyarrow" in app
    distintas = {k: (v, raiz[k]) for k, v in app.items() if k in raiz and raiz[k] != v}
    assert not distintas, distintas
