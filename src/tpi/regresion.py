"""Fase 2: MCO de referencia, auditoría de Gauss-Markov, observaciones influyentes, VIF y PCA.

Umbrales del enunciado: |r_i| > 3 (residuo estudentizado externo), h_ii > 2(p+1)/n (leverage),
D_i > 4/n o D_i > 0,5 (Cook), VIF > 5 o > 10.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tools.sm_exceptions import SingularMatrixWarning


@dataclass(frozen=True)
class DiagnosticoGM:
    bp_lm: float
    bp_gl: int
    bp_p: float
    white_lm: float
    white_p: float
    jb: float
    jb_p: float
    asimetria: float
    curtosis: float
    durbin_watson: float


def diagnostico_gauss_markov(m: RegressionResultsWrapper) -> DiagnosticoGM:
    """Breusch-Pagan y White (homocedasticidad), Jarque-Bera (normalidad) y Durbin-Watson
    (independencia; los residuos deben estar en el orden temporal de las observaciones)."""
    bp = het_breuschpagan(m.resid, m.model.exog)
    # En la regresión auxiliar de White el cuadrado de una dummy es la misma dummy: la matriz es
    # de rango incompleto por construcción. statsmodels toma los gl del rango (df_model), así que
    # el contraste es válido y el aviso es ruido.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularMatrixWarning)
        wh = het_white(m.resid, m.model.exog)
    jb, jb_p, asim, curt = jarque_bera(m.resid)
    return DiagnosticoGM(
        bp_lm=float(bp[0]),
        bp_gl=int(m.model.exog.shape[1] - 1),
        bp_p=float(bp[1]),
        white_lm=float(wh[0]),
        white_p=float(wh[1]),
        jb=float(jb),
        jb_p=float(jb_p),
        asimetria=float(asim),
        curtosis=float(curt),
        durbin_watson=float(durbin_watson(m.resid)),
    )


def influencia(m: RegressionResultsWrapper) -> pd.DataFrame:
    """Residuo estudentizado externo, leverage y distancia de Cook, con las banderas del enunciado."""
    inf = OLSInfluence(m)
    n, k = int(m.nobs), int(m.model.exog.shape[1])  # k = p + 1
    out = pd.DataFrame(
        {
            "r_ext": inf.resid_studentized_external,
            "leverage": inf.hat_matrix_diag,
            "cook": inf.cooks_distance[0],
        },
        index=m.model.data.row_labels,
    )
    out["outlier_y"] = out["r_ext"].abs() > 3
    out["alto_leverage"] = out["leverage"] > 2 * k / n
    out["influyente_4n"] = out["cook"] > 4 / n
    out["influyente_05"] = out["cook"] > 0.5
    return out


def tabla_vif(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 − R²_j), con constante en la regresión auxiliar."""
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    vals = [variance_inflation_factor(Xc.to_numpy(), i) for i in range(1, Xc.shape[1])]
    return pd.Series(vals, index=X.columns, name="VIF").sort_values(ascending=False)


def ajustar_pca(train: pd.DataFrame, columnas: list[str], seed: int = 0) -> Pipeline:
    """Estandariza y descompone **solo con Train**. Para Test se usa `pipe.transform` (rotación
    ciega con la media, el desvío y los autovectores de Train)."""
    pipe = Pipeline([("escala", StandardScaler()), ("pca", PCA(random_state=seed))])
    pipe.fit(train[columnas].astype(float))
    return pipe


def resumen_pca(pipe: Pipeline, columnas: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(varianza por componente con autovalor, % y % acumulado; matriz de cargas)."""
    pca: PCA = pipe.named_steps["pca"]
    nombres = [f"PC{i + 1}" for i in range(pca.n_components_)]
    var = pd.DataFrame(
        {
            "autovalor": pca.explained_variance_,
            "pct_var": 100 * pca.explained_variance_ratio_,
            "pct_acum": 100 * np.cumsum(pca.explained_variance_ratio_),
        },
        index=nombres,
    )
    cargas = pd.DataFrame(pca.components_.T, index=columnas, columns=nombres)
    return var, cargas
