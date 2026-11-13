"""Chequeo de la premisa experimental: ¿el orden con `random_bool = 1` es realmente aleatorio?

Si el orden de una lista es aleatorio, dentro de cada búsqueda la posición es independiente de
cualquier atributo del hotel, y la correlación de Spearman intra-búsqueda ρ_s entre `position` y
la calidad del hotel tiene esperanza 0. Si el orden lo decide el algoritmo, esperamos ρ_s < 0:
los hoteles mejor valuados aparecen más arriba (posición numéricamente menor).

Usamos la correlación **dentro** de cada búsqueda (y no la correlación agrupada) porque las
posiciones son una permutación por lista: mezclar listas de distinto largo y destino confundiría
el contraste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Margen de equivalencia fijado a priori (antes de ver los datos): |E[ρ_s]| < 0,03 se considera
# "sin asociación práctica" (un ρ de 0,03 explica < 0,1 % de la varianza de los rangos).
MARGEN_EQUIVALENCIA = 0.03


@dataclass(frozen=True)
class ResumenRho:
    variable: str
    n_busquedas: int
    rho_medio: float
    ic95_inf: float
    ic95_sup: float
    t: float
    gl: int
    p_valor: float

    @property
    def equivalente_a_cero(self) -> bool:
        """El IC 95 % de E[ρ_s] queda dentro de ±MARGEN (equivale a TOST con α = 0,025 por lado)."""
        return -MARGEN_EQUIVALENCIA < self.ic95_inf and self.ic95_sup < MARGEN_EQUIVALENCIA


def rho_intra_busqueda(
    df: pd.DataFrame, col: str, col_id: str = "srch_id", col_pos: str = "position"
) -> pd.Series:
    """ρ de Spearman entre `col_pos` y `col` dentro de cada búsqueda (una entrada por búsqueda).

    Las búsquedas donde `col` es constante (ρ indefinido) se descartan.
    """
    d = df[[col_id, col_pos, col]].dropna()
    rangos = d.groupby(col_id)[[col_pos, col]].rank()
    g_id = d[col_id]
    dx = rangos[col_pos] - rangos.groupby(g_id)[col_pos].transform("mean")
    dy = rangos[col] - rangos.groupby(g_id)[col].transform("mean")
    sumas = pd.DataFrame({"xy": dx * dy, "xx": dx**2, "yy": dy**2}).groupby(g_id).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = sumas["xy"] / np.sqrt(sumas["xx"] * sumas["yy"])
    return rho.replace([np.inf, -np.inf], np.nan).dropna().rename(f"rho_{col}")


def resumir_rho(rho: pd.Series, variable: str) -> ResumenRho:
    """Test t de una muestra para H0: E[ρ_s] = 0, con IC 95 %. Las búsquedas son independientes."""
    n = int(rho.shape[0])
    res = stats.ttest_1samp(rho, popmean=0.0)
    ic = res.confidence_interval(confidence_level=0.95)
    return ResumenRho(
        variable=variable,
        n_busquedas=n,
        rho_medio=float(rho.mean()),
        ic95_inf=float(ic.low),
        ic95_sup=float(ic.high),
        t=float(res.statistic),
        gl=n - 1,
        p_valor=float(res.pvalue),
    )
