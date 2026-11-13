"""Plan B: Online Shoppers (UCI #468, CC BY 4.0) con tratamiento semisintético.

Covariables y tasa base de compra REALES; tratamiento, mediador y efecto SIMULADOS con un
mecanismo conocido, de modo que el ATE y el CATE verdaderos están en el propio DataFrame
(`tau_real`) y cada estimador se puede auditar contra la verdad.

Mecanismo (todo con semilla fija):
    e(Z)  = expit(a0 + a1·z(log1p PageValues) + a2·returning + a3·z(log1p ProductRelated))
    expuesto ~ Bernoulli(e(Z))                         <- "el recomendador" expone según Z
    p0(Z) = P̂(Revenue = 1 | Z) por regresión logística sobre los datos reales
    M     ~ Bernoulli(m0 + δ·expuesto)                      <- interacción con el ítem (mediador)
    g(Z)  = min(γ·(1 + returning), 1 − p0(Z))
    Y     ~ Bernoulli(p0(Z) + g(Z)·M)
    τ(Z)  = E[Y(1) − Y(0) | Z] = δ·g(Z)              <- efecto total, todo a través de M

La confusión es por Z observado (Backdoor se cumple por construcción): el sesgo ingenuo existe
y un ajuste correcto lo elimina. Para la trampa del colisionador se simula C(T, Y) en la Fase 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from tpi.config import SEED

COLUMNAS_REALES = [
    "ProductRelated",
    "ProductRelated_Duration",
    "PageValues",
    "BounceRates",
    "VisitorType",
    "Weekend",
    "SpecialDay",
    "Revenue",
]
# Parámetros del mecanismo (fijados a priori, no ajustados a ningún resultado)
A0, A1, A2, A3 = -1.0, 1.2, 0.8, 0.5
M0, DELTA, GAMMA = 0.15, 0.35, 0.10


def _z(x: pd.Series) -> np.ndarray:
    v = np.log1p(x.to_numpy(dtype=float))
    return (v - v.mean()) / v.std()


def generar_semisintetico(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    faltan = [c for c in COLUMNAS_REALES if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas reales de Online Shoppers: {faltan}")
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["returning"] = (out["VisitorType"] == "Returning_Visitor").astype("int8")
    out["Weekend"] = out["Weekend"].astype("int8")

    zp, zr = _z(out["PageValues"]), _z(out["ProductRelated"])
    out["e_real"] = expit(A0 + A1 * zp + A2 * out["returning"] + A3 * zr)
    out["expuesto"] = (rng.uniform(size=len(out)) < out["e_real"]).astype("int8")

    x = np.column_stack(
        [zp, zr, _z(out["ProductRelated_Duration"]), out["BounceRates"], out["returning"]]
    )
    p0 = LogisticRegression(max_iter=1000).fit(x, out["Revenue"].astype(int)).predict_proba(x)[:, 1]
    g = np.minimum(GAMMA * (1 + out["returning"].to_numpy()), 1 - p0)

    out["M"] = (rng.uniform(size=len(out)) < M0 + DELTA * out["expuesto"]).astype("int8")
    out["Y"] = (rng.uniform(size=len(out)) < p0 + g * out["M"]).astype("int8")
    out["tau_real"] = DELTA * g
    return out
