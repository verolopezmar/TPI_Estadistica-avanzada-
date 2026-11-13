"""Inferencia clásica de la Fase 1: contrastes, tamaños de efecto, potencia y ANOVA.

Reglas de reporte del TPI (todas aplicadas por `Contraste.texto()`):
* α fijado a priori (`config.ALPHA`); estadístico, grados de libertad, p-valor exacto, IC 95 %
  de la diferencia y tamaño de efecto.
* "Se rechaza H0" / "no se rechaza H0". Nunca "se acepta H0".

Cada función recibe arrays de **unidades independientes**: en Expedia, búsquedas (contrastes
pareados dentro de la búsqueda) o reservas (a lo sumo una por búsqueda). Nunca filas
búsqueda×hotel sueltas, que comparten usuario y contexto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower, TTestPower

from tpi.config import ALPHA


@dataclass(frozen=True)
class Contraste:
    nombre: str
    estadistico: str  # "t", "U", "W", "F", "H"
    valor: float
    gl: float | tuple[float, float] | None
    p_valor: float
    diferencia: float | None
    ic95: tuple[float, float] | None
    efecto_nombre: str
    efecto: float
    n: tuple[int, ...]
    alpha: float = ALPHA
    extra: dict = field(default_factory=dict)

    @property
    def rechaza_h0(self) -> bool:
        return self.p_valor < self.alpha

    def texto(self) -> str:
        if self.gl is None:
            est = f"{self.estadistico} = {self.valor:.4g}"
        elif isinstance(self.gl, tuple):
            est = f"{self.estadistico}({_gl(self.gl[0])}; {_gl(self.gl[1])}) = {self.valor:.4g}"
        else:
            est = f"{self.estadistico}({_gl(self.gl)}) = {self.valor:.4g}"
        partes = [est, _p(self.p_valor)]
        if self.ic95 is not None:
            partes.append(f"IC 95 % [{self.ic95[0]:.4g}; {self.ic95[1]:.4g}]")
        partes.append(f"{self.efecto_nombre} = {self.efecto:.3f}")
        decision = "se rechaza H0" if self.rechaza_h0 else "no se rechaza H0"
        return f"{self.nombre}: " + ", ".join(partes) + f" (α = {self.alpha}) → {decision}"


def _gl(gl: float) -> str:
    """Grados de libertad sin notación científica: enteros tal cual, Welch con un decimal."""
    return f"{gl:.0f}" if float(gl).is_integer() else f"{gl:.1f}"


def _p(p: float) -> str:
    """p-valor exacto; si el cálculo en doble precisión subdesborda a 0, se informa la cota."""
    return "p < 1e-300 (subdesborde de doble precisión)" if p == 0 else f"p = {p:.3g}"


# ------------------------------------------------------------------------- dos muestras
def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """d de Cohen con desvío combinado y corrección de sesgo de Hedges (J ≈ 1 − 3/(4gl − 1))."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (n1 + n2) - 9)))


def welch_dos_grupos(a, b, nombre: str = "Welch") -> Contraste:
    """t de Welch para H0: μ_a = μ_b (no asume varianzas iguales), con IC 95 % de μ_a − μ_b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.ttest_ind(a, b, equal_var=False)
    ic = r.confidence_interval(0.95)
    return Contraste(
        nombre=nombre,
        estadistico="t",
        valor=float(r.statistic),
        gl=float(r.df),
        p_valor=float(r.pvalue),
        diferencia=float(a.mean() - b.mean()),
        ic95=(float(ic.low), float(ic.high)),
        efecto_nombre="g de Hedges",
        efecto=hedges_g(a, b),
        n=(len(a), len(b)),
    )


def mann_whitney(a, b, nombre: str = "Mann-Whitney") -> Contraste:
    """U de Mann-Whitney (H0: P(A > B) = 1/2), con r biserial de rangos = 2·U/(n1·n2) − 1 y
    diferencia de Hodges-Lehmann (mediana de las diferencias a_i − b_j, por submuestreo si es
    grande)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    rng = np.random.default_rng(0)
    sa = a if n1 <= 3000 else rng.choice(a, 3000, replace=False)
    sb = b if n2 <= 3000 else rng.choice(b, 3000, replace=False)
    return Contraste(
        nombre=nombre,
        estadistico="U",
        valor=float(r.statistic),
        gl=None,
        p_valor=float(r.pvalue),
        diferencia=float(np.median(np.subtract.outer(sa, sb))),
        ic95=None,
        efecto_nombre="r biserial de rangos",
        efecto=float(2 * r.statistic / (n1 * n2) - 1),
        n=(n1, n2),
    )


def contraste_pareado(diferencias, nombre: str = "t pareado") -> Contraste:
    """t de una muestra sobre diferencias pareadas (H0: E[D] = 0) con d_z = D̄ / s_D.

    El Wilcoxon de rangos con signo se adjunta en `extra` como contraste de resguardo.
    """
    d = np.asarray(diferencias, float)
    r = stats.ttest_1samp(d, 0.0)
    ic = r.confidence_interval(0.95)
    w = stats.wilcoxon(d, zero_method="wilcox")
    return Contraste(
        nombre=nombre,
        estadistico="t",
        valor=float(r.statistic),
        gl=float(r.df),
        p_valor=float(r.pvalue),
        diferencia=float(d.mean()),
        ic95=(float(ic.low), float(ic.high)),
        efecto_nombre="d_z",
        efecto=float(d.mean() / d.std(ddof=1)),
        n=(len(d),),
        extra={"wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue)},
    )


# ------------------------------------------------------------------------------ potencia
def potencia(efecto: float, n1: int, n2: int | None = None, alpha: float = ALPHA) -> float:
    """1 − β para t de dos muestras (n2 dado) o pareado/una muestra (n2 = None)."""
    if n2 is None:
        return float(TTestPower().power(effect_size=efecto, nobs=n1, alpha=alpha))
    return float(TTestIndPower().power(effect_size=efecto, nobs1=n1, ratio=n2 / n1, alpha=alpha))


def efecto_minimo_detectable(
    n1: int, n2: int | None = None, alpha: float = ALPHA, potencia_obj: float = 0.80
) -> float:
    """Menor |d| detectable con la potencia objetivo: lo que la muestra puede y no puede ver."""
    if n2 is None:
        return float(TTestPower().solve_power(nobs=n1, alpha=alpha, power=potencia_obj))
    return float(
        TTestIndPower().solve_power(nobs1=n1, ratio=n2 / n1, alpha=alpha, power=potencia_obj)
    )


# ---------------------------------------------------------------------------------- ANOVA
def describir_grupos(df: pd.DataFrame, y: str, grupo: str) -> pd.DataFrame:
    """n, media, desvío, asimetría y curtosis en exceso por grupo (criterio |asimetría| ≤ 1)."""
    return df.groupby(grupo, observed=True)[y].agg(
        n="size",
        media="mean",
        desvio="std",
        asimetria=lambda s: stats.skew(s, bias=False),
        curtosis_exceso=lambda s: stats.kurtosis(s, bias=False),
    )


def anova_un_factor(df: pd.DataFrame, y: str, grupo: str) -> dict[str, Contraste]:
    """ANOVA clásico, Levene (centrado en la mediana: Brown-Forsythe), ANOVA de Welch y
    Kruskal-Wallis. Tamaños de efecto: η² y ω² (F), ε² (Kruskal-Wallis)."""
    d = df[[y, grupo]].dropna()
    muestras = [g[y].to_numpy(float) for _, g in d.groupby(grupo, observed=True)]
    k, n = len(muestras), len(d)

    f = stats.f_oneway(*muestras)
    media = d[y].mean()
    ss_b = sum(len(m) * (m.mean() - media) ** 2 for m in muestras)
    ss_t = ((d[y] - media) ** 2).sum()
    ms_w = (ss_t - ss_b) / (n - k)
    clasico = Contraste(
        "ANOVA clásico", "F", float(f.statistic), (k - 1, n - k), float(f.pvalue), None, None,
        "η²", float(ss_b / ss_t), tuple(len(m) for m in muestras),
        extra={"omega2": float((ss_b - (k - 1) * ms_w) / (ss_t + ms_w))},
    )  # fmt: skip

    lev = stats.levene(*muestras, center="median")
    levene = Contraste(
        "Levene (Brown-Forsythe)", "F", float(lev.statistic), (k - 1, n - k),
        float(lev.pvalue), None, None, "razón de varianzas máx/mín",
        float(max(np.var(m, ddof=1) for m in muestras) / min(np.var(m, ddof=1) for m in muestras)),
        tuple(len(m) for m in muestras),
    )  # fmt: skip

    w = pg.welch_anova(data=d, dv=y, between=grupo).iloc[0]
    welch = Contraste(
        "ANOVA de Welch", "F", float(w["F"]), (float(w["ddof1"]), float(w["ddof2"])),
        float(w["p_unc"]), None, None, "η²", float(w["np2"]), tuple(len(m) for m in muestras),
    )  # fmt: skip

    h = stats.kruskal(*muestras)
    kw = Contraste(
        "Kruskal-Wallis", "H", float(h.statistic), float(k - 1), float(h.pvalue), None, None,
        "ε²", float(h.statistic / ((n**2 - 1) / (n + 1))), tuple(len(m) for m in muestras),
    )  # fmt: skip
    return {"clasico": clasico, "levene": levene, "welch": welch, "kruskal": kw}


def posthoc(df: pd.DataFrame, y: str, grupo: str, alpha: float = ALPHA) -> pd.DataFrame:
    """Comparaciones múltiples controlando la tasa de error por familia, tres criterios:

    * Tukey HSD: asume varianzas iguales (referencia del enunciado).
    * Games-Howell: no asume varianzas iguales (el que corresponde si Levene rechaza).
    * Welch por pares + Bonferroni: conservador, sin supuesto de homocedasticidad.
    """
    d = df[[y, grupo]].dropna()
    tk = pairwise_tukeyhsd(d[y], d[grupo], alpha=alpha)
    tukey = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
    tukey = tukey.rename(columns={"group1": "A", "group2": "B", "p-adj": "p_tukey"})
    tukey[["A", "B"]] = tukey[["A", "B"]].astype(str)

    gh = pg.pairwise_gameshowell(data=d, dv=y, between=grupo)
    gh = gh.rename(columns={"pval": "p_games_howell", "diff": "diferencia", "se": "ee"})
    gh[["A", "B"]] = gh[["A", "B"]].astype(str)

    niveles = sorted(d[grupo].unique())
    m = len(niveles) * (len(niveles) - 1) // 2
    filas = []
    for a, b in combinations(niveles, 2):
        r = stats.ttest_ind(d.loc[d[grupo] == a, y], d.loc[d[grupo] == b, y], equal_var=False)
        filas.append({"A": str(a), "B": str(b), "p_bonferroni": min(1.0, r.pvalue * m)})
    bonf = pd.DataFrame(filas)

    def _par(t: pd.DataFrame) -> pd.Series:  # clave sin orden: Tukey y Games-Howell difieren
        return t[["A", "B"]].apply(lambda r: "|".join(sorted(r)), axis=1)

    gh, bonf, tukey = (t.assign(par=_par(t)) for t in (gh, bonf, tukey))
    out = gh[["par", "A", "B", "diferencia", "ee", "p_games_howell"]]
    out = out.merge(bonf[["par", "p_bonferroni"]], on="par")
    out = out.merge(tukey[["par", "p_tukey"]], on="par").drop(columns="par")
    for c in ("p_games_howell", "p_bonferroni", "p_tukey"):
        out[f"rechaza_{c[2:]}"] = out[c] < alpha
    return out
