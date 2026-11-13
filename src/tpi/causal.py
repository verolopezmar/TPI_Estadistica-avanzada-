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
    """OR de T sobre Y dentro de cada `grupo` (las listas sin variación en Y no aportan).

    Con a lo sumo un evento por grupo (las reservas de Expedia) usa el estimador exacto por
    Newton-Raphson (`eleccion_logit`, errores sándwich por grupo); si hay grupos con varios eventos
    (p. ej. clics), usa `ConditionalLogit` de statsmodels con guarda de convergencia."""
    if df.groupby(grupo)[y].sum().max() <= 1:
        d = df[df.groupby(grupo)[y].transform("sum") == 1]
        covs = [c for c in covariables if d.groupby(grupo)[c].std().fillna(0).gt(0).any()]
        a = eleccion_logit(df, t, y, covs, grupo)
        b, ee = a.params[t], a.ee[t]
        from scipy.stats import norm

        return Efecto(
            "logit condicional",
            float(np.exp(b)),
            (float(np.exp(b - 1.96 * ee)), float(np.exp(b + 1.96 * ee))),
            float(2 * norm.sf(abs(b / ee))),
            a.n_grupos,
            "OR intra-búsqueda",
        )
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
    # Las covariables constantes dentro de cada grupo (contexto de la búsqueda) quedan en cero tras
    # la transformación within: no están identificadas y se descartan (evita una matriz singular;
    # el coeficiente de T no cambia).
    var_within = [c for c in covariables if w[c].abs().max() > 1e-12]
    r = sm.OLS(w[y], w[[t, *var_within]]).fit(cov_type="cluster", cov_kwds={"groups": d[grupo]})
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


# --------------------------------------------------------------------------------------------
# Logit condicional vectorizado (una reserva por búsqueda), con pesos por búsqueda.
# Con a lo sumo un evento por grupo, el logit condicional es un logit multinomial de elección
# dentro de la lista: P(i elegido | s) = exp(x_i'b) / sum_{j en s} exp(x_j'b). Es exacto (no una
# aproximación) y permite bootstrap por búsqueda en segundos.
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AjusteEleccion:
    params: pd.Series
    ee: pd.Series
    n_grupos: int


def _preparar_eleccion(df, t, y, covariables, grupo):
    d = df[df.groupby(grupo)[y].transform("sum") == 1].sort_values(grupo)
    if df.groupby(grupo)[y].sum().max() > 1:
        raise ValueError("Hay grupos con más de un evento: usar logit_condicional (statsmodels)")
    X = d[[t, *covariables]].astype(float)
    covs = [c for c in X.columns if c != t]
    if covs:
        X[covs] = (X[covs] - X[covs].mean()) / X[covs].std().replace(0, 1)
    codigos, inicio = np.unique(d[grupo].to_numpy(), return_index=True)
    return d, X, d[y].to_numpy(dtype=float), inicio, codigos


def eleccion_logit(
    df: pd.DataFrame,
    t: str,
    y: str,
    covariables: list[str],
    grupo: str = "srch_id",
    pesos_grupo: pd.Series | None = None,
) -> AjusteEleccion:
    """Logit condicional (≤ 1 evento por grupo) por máxima verosimilitud con gradiente analítico.
    `pesos_grupo` (indexado por id de grupo) pondera la log-verosimilitud de cada búsqueda.
    Errores estándar: sándwich robusto por grupo (válido también con pesos)."""
    d, X, yv, inicio, codigos = _preparar_eleccion(df, t, y, covariables, grupo)
    Xm = X.to_numpy()
    w = (
        np.ones(len(codigos))
        if pesos_grupo is None
        else pesos_grupo.reindex(codigos).to_numpy(float)
    )
    largo = np.diff(np.r_[inicio, len(d)])
    w_fila = np.repeat(w, largo)

    def evaluar(b):
        eta = Xm @ b
        m = np.maximum.reduceat(eta, inicio)
        e = np.exp(eta - np.repeat(m, largo))
        den = np.add.reduceat(e, inicio)
        p = e / np.repeat(den, largo)
        ll = np.sum(w * (np.add.reduceat(yv * eta, inicio) - m - np.log(den)))
        grad = Xm.T @ (w_fila * (yv - p))
        pX = p[:, None] * Xm
        media_g = np.add.reduceat(pX, inicio)
        H = Xm.T @ (w_fila[:, None] * pX) - (media_g * w[:, None]).T @ media_g  # −∂²ll
        return ll, grad, H, p

    # Newton-Raphson amortiguado (la log-verosimilitud es cóncava)
    b = np.zeros(Xm.shape[1])
    ll, grad, H, p = evaluar(b)
    for _ in range(100):
        paso = np.linalg.solve(H, grad)
        if grad @ paso < 1e-10:  # decremento de Newton: mejora esperada de la log-verosimilitud
            break
        a = 1.0
        while True:
            ll_n, grad_n, H_n, p_n = evaluar(b + a * paso)
            if ll_n >= ll - 1e-12 or a < 1e-8:
                break
            a /= 2
        b, ll, grad, H, p = b + a * paso, ll_n, grad_n, H_n, p_n
    else:
        raise RuntimeError("El logit de elección no convergió en 100 iteraciones de Newton")
    # Errores estándar sándwich robustos por grupo
    score_g = np.add.reduceat((yv - p)[:, None] * Xm, inicio) * w[:, None]
    Hinv = np.linalg.inv(H)
    V = Hinv @ (score_g.T @ score_g) @ Hinv
    return AjusteEleccion(
        pd.Series(b, index=X.columns), pd.Series(np.sqrt(np.diag(V)), index=X.columns), len(codigos)
    )


def efecto_marginal_eleccion(
    df, t, y, covariables, grupo="srch_id", pesos_grupo=None, ajuste=None
) -> float:
    """Efecto marginal promedio en la escala de probabilidad (colapsable): para cada hotel de las
    búsquedas con reserva, P(ser el elegido | T=1) − P(ser el elegido | T=0), con el resto de la
    lista fijo; promedio ponderado por búsqueda."""
    a = ajuste if ajuste is not None else eleccion_logit(df, t, y, covariables, grupo, pesos_grupo)
    d, X, _, inicio, codigos = _preparar_eleccion(df, t, y, covariables, grupo)
    Xm, b = X.to_numpy(), a.params.to_numpy()
    largo = np.diff(np.r_[inicio, len(d)])
    eta = Xm @ b
    den = np.repeat(np.add.reduceat(np.exp(eta), inicio), largo)
    bt = b[0]
    e0 = np.exp(eta - bt * Xm[:, 0])
    e1 = e0 * np.exp(bt)
    otros = den - np.exp(eta)
    ame_fila = e1 / (otros + e1) - e0 / (otros + e0)
    w = (
        np.ones(len(codigos))
        if pesos_grupo is None
        else pesos_grupo.reindex(codigos).to_numpy(float)
    )
    w_fila = np.repeat(w, largo)
    return float(np.sum(w_fila * ame_fila) / np.sum(w_fila))


def pesos_odds_busqueda(
    origen: pd.DataFrame, destino: pd.DataFrame, columnas: list[str], seed: int = SEED
) -> pd.Series:
    """Pesos por búsqueda que llevan la composición de `origen` a la de `destino`:
    w = P(destino | X) / P(origen | X) (pesos de odds), estimados con una logística estandarizada
    sobre covariables de nivel búsqueda. Normalizados a media 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    a, b = origen[columnas].assign(_d=0), destino[columnas].assign(_d=1)
    ab = pd.concat([a, b])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(
        ab[columnas], ab["_d"]
    )
    p = m.predict_proba(origen[columnas])[:, 1]
    w = pd.Series(p / (1 - p), index=origen.index)
    return w / w.mean()


def smd(a: pd.Series, b: pd.Series, w_a: np.ndarray | None = None) -> float:
    """Diferencia de medias estandarizada (desvío combinado sin ponderar, criterio de Austin)."""
    ma = np.average(a, weights=w_a) if w_a is not None else a.mean()
    return float((ma - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


def _remuestrear(
    df: pd.DataFrame, rng: np.random.Generator, grupo: str = "srch_id"
) -> pd.DataFrame:
    """Remuestra búsquedas completas con reposición; cada copia recibe un id nuevo."""
    pos = df.groupby(grupo, sort=True).indices
    ids = np.array(list(pos))
    elegidos = rng.choice(len(ids), size=len(ids), replace=True)
    filas = [pos[ids[k]] for k in elegidos]
    out = df.iloc[np.concatenate(filas)].copy()
    out[grupo] = np.repeat(np.arange(len(elegidos)), [len(f) for f in filas])
    return out


def _replica_brecha(d0, d1, t, y, covs, cols_busqueda, semilla):
    """Una réplica bootstrap (búsquedas remuestreadas con reposición en cada estrato): OR de cada
    estrato y su cociente (sin ponderar y con R0 re-ponderado a la composición de R1, pesos
    re-estimados en la réplica), efectos marginales promedio (AME) y diferencia de riesgos con
    efectos fijos (MPL-EF), con sus cocientes y diferencias."""
    rng = np.random.default_rng(semilla)
    b0, b1 = _remuestrear(d0, rng), _remuestrear(d1, rng)
    a1 = eleccion_logit(b1, t, y, covs)
    a0 = eleccion_logit(b0, t, y, covs)
    s0 = b0.groupby("srch_id")[cols_busqueda].first()
    s1 = b1.groupby("srch_id")[cols_busqueda].first()
    w = pesos_odds_busqueda(s0, s1, cols_busqueda)
    a0w = eleccion_logit(b0, t, y, covs, pesos_grupo=w)
    ame1 = efecto_marginal_eleccion(b1, t, y, covs, ajuste=a1)
    ame0 = efecto_marginal_eleccion(b0, t, y, covs, ajuste=a0)
    ame0w = efecto_marginal_eleccion(b0, t, y, covs, pesos_grupo=w, ajuste=a0w)
    fe0 = lpm_efectos_fijos(b0, t, y, covs).estimacion
    fe1 = lpm_efectos_fijos(b1, t, y, covs).estimacion
    return {
        "OR_R0": float(np.exp(a0.params[t])),
        "OR_R1": float(np.exp(a1.params[t])),
        "cociente_OR": float(np.exp(a0.params[t] - a1.params[t])),
        "cociente_OR_ponderado": float(np.exp(a0w.params[t] - a1.params[t])),
        "AME_R0": ame0,
        "AME_R1": ame1,
        "cociente_AME": ame0 / ame1,
        "diferencia_AME": ame0 - ame1,
        "cociente_AME_ponderado": ame0w / ame1,
        "DR_EF_R0": fe0,
        "DR_EF_R1": fe1,
        "diferencia_DR_EF": fe0 - fe1,
        "cociente_DR_EF": fe0 / fe1,
    }


def bootstrap_brecha(
    d0: pd.DataFrame,
    d1: pd.DataFrame,
    t: str,
    y: str,
    covariables: list[str],
    cols_busqueda: list[str],
    B: int = 200,
    seed: int = SEED,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Bootstrap por búsqueda (clúster), independiente en cada estrato, de la brecha entre el
    efecto observacional (d0) y el benchmark (d1). `cols_busqueda` son columnas constantes por
    búsqueda ya presentes en d0 y d1 (para re-estimar los pesos de balance en cada réplica)."""
    from joblib import Parallel, delayed

    semillas = np.random.default_rng(seed).integers(0, 2**31 - 1, size=B)
    filas = Parallel(n_jobs=n_jobs)(
        delayed(_replica_brecha)(d0, d1, t, y, covariables, cols_busqueda, int(s)) for s in semillas
    )
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ IPW: overlap y positividad
LIMITES_PS = (0.01, 0.99)  # recorte estándar de la propensión (Crump et al., 2009, regla práctica)


def diagnostico_overlap(
    ps: np.ndarray, t: np.ndarray, limites: tuple[float, float] = LIMITES_PS
) -> dict[str, float]:
    """Positividad: 0 < e(z) < 1. Resume la propensión estimada por grupo de tratamiento.

    Devuelve los percentiles 1 y 99 de ê por grupo, la fracción de unidades fuera de `limites`,
    el peso IPW máximo y el tamaño efectivo de Kish de cada grupo,
    ESS = (Σw)² / Σw², que cae cuando unos pocos pesos dominan la estimación.
    """
    ps, t = np.asarray(ps, float), np.asarray(t).astype(bool)
    w1, w0 = 1 / ps[t], 1 / (1 - ps[~t])
    fuera = (ps < limites[0]) | (ps > limites[1])
    return {
        "ps_p01_T1": float(np.percentile(ps[t], 1)),
        "ps_p99_T1": float(np.percentile(ps[t], 99)),
        "ps_p01_T0": float(np.percentile(ps[~t], 1)),
        "ps_p99_T0": float(np.percentile(ps[~t], 99)),
        "pct_fuera_T1": float(100 * fuera[t].mean()),
        "pct_fuera_T0": float(100 * fuera[~t].mean()),
        "peso_max": float(max(w1.max(), w0.max())),
        "ess_T1": float(w1.sum() ** 2 / (w1**2).sum()),
        "ess_T0": float(w0.sum() ** 2 / (w0**2).sum()),
        "n_T1": int(t.sum()),
        "n_T0": int((~t).sum()),
    }


def _sumas_ipw(y, t, ps, limites, modo):
    """Componentes del estimador de Hájek por unidad (y la máscara de unidades que entran)."""
    y, t, ps = np.asarray(y, float), np.asarray(t, float), np.asarray(ps, float)
    entra = np.ones(len(ps), bool)
    if limites is not None:
        if modo == "excluir":  # trimming: se descartan las unidades sin overlap
            entra = (ps >= limites[0]) & (ps <= limites[1])
        elif modo == "truncar":  # truncamiento: la propensión se lleva al límite
            ps = np.clip(ps, *limites)
        else:
            raise ValueError("modo debe ser 'excluir' o 'truncar'")
    w1, w0 = t / ps * entra, (1 - t) / (1 - ps) * entra
    return np.column_stack([w1 * y, w1, w0 * y, w0])


def ipw_hajek(
    y: np.ndarray,
    t: np.ndarray,
    ps: np.ndarray,
    limites: tuple[float, float] | None = None,
    modo: str = "excluir",
) -> float:
    """ATE por IPW normalizado (Hájek):
    τ̂ = Σ T·Y/ê / Σ T/ê − Σ (1−T)·Y/(1−ê) / Σ (1−T)/(1−ê).
    Con `limites`, recorta las unidades sin overlap (`modo="excluir"`) o trunca ê (`"truncar"`)."""
    s = _sumas_ipw(y, t, ps, limites, modo).sum(axis=0)
    return float(s[0] / s[1] - s[2] / s[3])


def bootstrap_ipw(
    y: np.ndarray,
    t: np.ndarray,
    ps: np.ndarray,
    grupos: np.ndarray,
    B: int = 1000,
    seed: int = SEED,
    limites: tuple[float, float] | None = None,
    modo: str = "excluir",
) -> dict[str, float]:
    """IC 95 % percentil del IPW de Hájek por bootstrap de clústeres (búsquedas completas).

    La propensión queda fija (no se re-estima en cada réplica). Con propensión estimada, ignorar
    esa etapa da un IC **conservador** para el ATE (Hirano, Imbens y Ridder, 2003), así que el IC
    no subestima la incertidumbre. Vectorizado: cada réplica es un vector de multiplicidades de
    clúster por la matriz de sumas por clúster.
    """
    comp = pd.DataFrame(_sumas_ipw(y, t, ps, limites, modo)).groupby(np.asarray(grupos)).sum()
    m = comp.to_numpy()
    rng = np.random.default_rng(seed)
    k = len(m)
    reps = np.empty(B)
    for b in range(B):
        s = np.bincount(rng.integers(0, k, k), minlength=k) @ m
        reps[b] = s[0] / s[1] - s[2] / s[3]
    s = m.sum(axis=0)
    return {
        "ATE": float(s[0] / s[1] - s[2] / s[3]),
        "IC95_inf": float(np.percentile(reps, 2.5)),
        "IC95_sup": float(np.percentile(reps, 97.5)),
        "ee_bootstrap": float(reps.std(ddof=1)),
        "B": B,
    }
