"""Fase 4: estimadores causales y simulaciones con verdad conocida.

Decisiones (propuestas por Claude Code; ver docs/ia_log.md):
* **Escala relativa principal:** logit condicional por búsqueda (McFadden). Condiciona en el
  número de reservas de cada lista, así que el odds ratio intra-búsqueda no depende de cuántas
  listas con compra seleccionó el proveedor (BITÁCORA, Entrada 4, amenaza A).
* **Escala absoluta:** modelo lineal de probabilidad con efectos fijos por búsqueda (within) y
  errores por clúster de búsqueda.
* **Simulaciones:** filtro de selección S (≥ 1 clic por lista) y colisionador C = f(T, Y), para
  mostrar con verdad conocida qué recupera cada estimador.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tpi.config import SEED


@dataclass(frozen=True)
class Efecto:
    nombre: str
    estimacion: float
    ic95: tuple[float, float]
    p_valor: float
    n: int
    escala: str  # "OR", "diferencia de probabilidad", ...


def logit_condicional(
    df: pd.DataFrame, t: str, y: str, covariables: list[str], grupo: str = "srch_id"
) -> Efecto:
    """OR de T sobre Y dentro de cada `grupo` (las listas sin variación en Y no aportan)."""
    d = df[df.groupby(grupo)[y].transform("nunique") > 1]
    X = d[[t, *covariables]].astype(float)
    X = X.loc[:, X.groupby(d[grupo]).transform("std").fillna(0).gt(0).any()]  # sin variación intra
    covs = [c for c in X.columns if c != t]
    X[covs] = (X[covs] - X[covs].mean()) / X[covs].std()  # escala: ayuda a converger; no cambia τ
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        r = ConditionalLogit(d[y].astype(int), X, groups=d[grupo]).fit(
            disp=0, method="bfgs", maxiter=2000
        )
    if any(issubclass(a.category, ConvergenceWarning) for a in avisos):
        raise RuntimeError(f"El logit condicional no convergió ({t} ~ {covs})")
    b, ee = r.params[t], r.bse[t]
    return Efecto(
        "logit condicional",
        float(np.exp(b)),
        (float(np.exp(b - 1.96 * ee)), float(np.exp(b + 1.96 * ee))),
        float(r.pvalues[t]),
        int(d[grupo].nunique()),
        "OR intra-búsqueda",
    )


def lpm_efectos_fijos(
    df: pd.DataFrame, t: str, y: str, covariables: list[str], grupo: str = "srch_id"
) -> Efecto:
    """Diferencia de probabilidad de T sobre Y con efectos fijos de `grupo` (transformación
    within) y errores robustos por clúster."""
    cols = [y, t, *covariables]
    d = df[[grupo, *cols]].astype({c: float for c in cols})
    w = d[cols] - d.groupby(grupo)[cols].transform("mean")
    r = sm.OLS(w[y], w[[t, *covariables]]).fit(cov_type="cluster", cov_kwds={"groups": d[grupo]})
    ic = r.conf_int().loc[t]
    return Efecto(
        "MPL con efectos fijos",
        float(r.params[t]),
        (float(ic[0]), float(ic[1])),
        float(r.pvalues[t]),
        int(d[grupo].nunique()),
        "diferencia de probabilidad",
    )


def simular_listas(
    n_listas: int = 4000,
    n_hoteles: int = 25,
    efecto_logit: float = 0.8,
    base_logit: float = -4.0,
    sd_intencion: float = 1.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Listas con orden aleatorio, intención de compra latente por lista (I) y calidad por hotel.
    P(clic) = expit(base + efecto·T + calidad + I). Verdad: OR condicional = exp(efecto_logit)."""
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(n_listas), n_hoteles)
    pos = np.concatenate([rng.permutation(n_hoteles) + 1 for _ in range(n_listas)])
    t = (pos <= 5).astype(int)
    calidad = rng.normal(size=s.size)
    intencion = np.repeat(rng.normal(scale=sd_intencion, size=n_listas), n_hoteles)
    p = expit(base_logit + efecto_logit * t + 0.5 * calidad + intencion)
    y = (rng.uniform(size=s.size) < p).astype(int)
    return pd.DataFrame({"srch_id": s, "T": t, "calidad": calidad, "Y": y, "p_real": p})


def efecto_marginal_real(d: pd.DataFrame, efecto_logit: float, base_logit: float = -4.0) -> float:
    """Diferencia de probabilidad verdadera promedio (contrafácticos exactos de la simulación)."""
    lin = np.log(d["p_real"] / (1 - d["p_real"])) - efecto_logit * d["T"]
    return float((expit(lin + efecto_logit) - expit(lin)).mean())


def simular_colisionador(n: int = 50_000, seed: int = SEED) -> pd.DataFrame:
    """T asignado al azar, Zh independiente de T, Y depende de T y Zh, y C (destacado) depende de
    T y de Y. Condicionar en C induce asociación T–Zh (Berkson) y sesga T → Y."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    zh = rng.normal(size=n)
    y = (rng.uniform(size=n) < expit(-2 + 0.7 * t + 0.8 * zh)).astype(int)
    c = (rng.uniform(size=n) < expit(-3 + 2.0 * t + 2.5 * y)).astype(int)
    return pd.DataFrame({"T": t, "Zh": zh, "Y": y, "C": c})


def grafo_dot(t: str, y: str, confusores: list[str], mediador: str | None = None) -> str:
    """DAG en DOT con nombres de columnas reales (DoWhy). Z → T, Z → Y, T → M → Y (si hay M)."""
    arcos = [f"{z} -> {t}; {z} -> {y};" for z in confusores]
    if mediador:
        arcos += [f"{t} -> {mediador}; {mediador} -> {y};"] + [
            f"{z} -> {mediador};" for z in confusores
        ]
    else:
        arcos.append(f"{t} -> {y};")
    return "digraph {" + " ".join(arcos) + "}"


def modelo_dowhy(df: pd.DataFrame, t: str, y: str, confusores: list[str], mediador: str | None):
    """(CausalModel, estimando identificado por Backdoor) a partir del DAG explícito."""
    import networkx as nx
    from dowhy import CausalModel

    arcos = [(z, t) for z in confusores] + [(z, y) for z in confusores]
    if mediador:
        arcos += [(t, mediador), (mediador, y)] + [(z, mediador) for z in confusores]
    else:
        arcos.append((t, y))
    g = nx.DiGraph(arcos)
    cols = [t, y, *confusores, *([mediador] if mediador else [])]
    modelo = CausalModel(data=df[cols], treatment=t, outcome=y, graph=g)
    return modelo, modelo.identify_effect(proceed_when_unidentifiable=True)
