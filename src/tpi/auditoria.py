"""Auditoría forense de datos (Fase 1, rúbrica 1.3): génesis de cada anomalía y distorsión
cuantificada antes/después del tratamiento.

Cuatro familias de anomalías en Expedia:

1. **Centinelas** (códigos que no son el valor real): `prop_review_score = 0` (hotel sin
   reseñas), `prop_starrating = 0` (sin categoría), `prop_log_historical_price = 0` (sin ventas
   en el período previo). Se pasan a NaN y se conserva un indicador binario, porque la ausencia
   de información es en sí una señal para el ranker y para el usuario.
2. **Desplazamiento de escala del precio**: `price_usd` es mayormente por noche y por habitación,
   pero una fracción de los registros está por estadía. Se diagnostica con las reservas
   (gasto real / precio informado) y se neutraliza con el precio **relativo a la mediana de la
   búsqueda**, que es invariante a la escala común de la lista.
3. **Valores extremos de precio** (millones de USD por noche): regla de outlier
   intra-búsqueda, más winsorización del precio relativo.
4. **Faltantes**: se contrasta MCAR comparando el indicador de faltante contra variables
   observadas (si depende de ellas, MCAR se rechaza) y se argumenta MAR/MNAR por el mecanismo de
   generación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tpi.config import ALPHA

CENTINELAS_A_NAN: dict[str, tuple[float, str]] = {
    "prop_review_score": (0.0, "sin_resenas"),
    "prop_starrating": (0.0, "sin_estrellas"),
    "prop_log_historical_price": (0.0, "sin_historial_precio"),
}
# Razón gasto/(precio·noches·habitaciones) compatible con "precio por noche" (impuestos y tasas
# incluidos en el gasto). Banda fijada mirando la distribución de las reservas de 1 noche, donde
# ambas escalas coinciden y no hay ambigüedad.
BANDA_RAZON = (0.85, 1.6)
# Outlier de precio: más de K veces (o menos de 1/K) la mediana de su propia búsqueda.
K_OUTLIER = 10.0
WINSOR = (0.01, 0.99)


def resumen_distribucion(s: pd.Series) -> dict[str, float]:
    s = s.dropna().astype(float)
    return {
        "n": int(s.size),
        "media": float(s.mean()),
        "desvio": float(s.std()),
        "asimetria": float(stats.skew(s)),
        "p01": float(s.quantile(0.01)),  # p1/p99 en lugar de mínimo y máximo: estos describen
        "p50": float(s.median()),  # un único registro (criterio "solo agregados")
        "p99": float(s.quantile(0.99)),
    }


# ------------------------------------------------------------------------------- centinelas
def aplicar_centinelas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pasa centinelas a NaN y agrega indicadores. Devuelve (df, tabla antes/después)."""
    out = df.copy()
    filas = []
    for col, (valor, indicador) in CENTINELAS_A_NAN.items():
        es = out[col] == valor
        antes = resumen_distribucion(out[col])
        out[indicador] = es.astype("int8")
        out.loc[es, col] = np.nan
        despues = resumen_distribucion(out[col])
        filas.append(
            {
                "variable": col,
                "centinela": valor,
                "pct_filas": 100 * es.mean(),
                "media_antes": antes["media"],
                "media_despues": despues["media"],
                "sesgo_media_pct": 100 * (antes["media"] / despues["media"] - 1),
                "desvio_antes": antes["desvio"],
                "desvio_despues": despues["desvio"],
            }
        )
    return out, pd.DataFrame(filas).set_index("variable")


# ------------------------------------------------------------------------- escala del precio
def diagnostico_escala_precio(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica cada reserva según la escala compatible de `price_usd`.

    razón_noche   = gasto / (precio · noches · habitaciones)  ≈ 1,1 si el precio es por noche
    razón_estadía = gasto / (precio · habitaciones)            ≈ 1,1 si el precio es por estadía
    Con 1 noche ambas coinciden, así que la clasificación usa solo estadías de 2 o más noches.
    """
    b = df[(df["booking_bool"] == 1) & (df["gross_bookings_usd"] > 0)].copy()
    b = b[b["srch_length_of_stay"] >= 2]
    hab = b["price_usd"] * b["srch_room_count"]
    r_noche = b["gross_bookings_usd"] / (hab * b["srch_length_of_stay"])
    r_estadia = b["gross_bookings_usd"] / hab
    lo, hi = BANDA_RAZON
    noche, estadia = r_noche.between(lo, hi), r_estadia.between(lo, hi)
    escala = np.select([noche, estadia], ["por_noche", "por_estadia"], "incompatible")
    return b.assign(razon_noche=r_noche, razon_estadia=r_estadia, escala=escala)


def cerca_tukey_log(precio: pd.Series, k: float = 3.0) -> tuple[float, float]:
    """Cercas "extremas" de Tukey (Q1 − k·IQR, Q3 + k·IQR) sobre log(precio), en USD."""
    lp = np.log(precio[precio > 0])
    q1, q3 = lp.quantile([0.25, 0.75])
    return float(np.exp(q1 - k * (q3 - q1))), float(np.exp(q3 + k * (q3 - q1)))


def precio_relativo(
    df: pd.DataFrame, col_id: str = "srch_id", ajuste: pd.Series | None = None
) -> pd.DataFrame:
    """Tratamiento del precio en dos capas:

    * `precio_outlier` = 1 si el precio cae fuera de las cercas extremas de Tukey sobre el log
      (anomalía absoluta: incluye búsquedas enteras con precios imposibles, que la regla relativa
      no detecta porque la mediana de la búsqueda también es absurda) **o** si es más de K veces
      (o menos de 1/K) la mediana de su búsqueda (anomalía relativa).
    * `precio_rel` = precio / mediana de la búsqueda, winsorizado en WINSOR; `log_precio_rel`.
      Es invariante a una escala común de la lista (por noche vs. por estadía, moneda).

    `ajuste` (máscara booleana, típicamente Train) fija las filas con las que se calculan las
    cercas de Tukey y los cuantiles de winsorización; después se aplican a todas las filas. La
    mediana de la búsqueda no fuga información porque cada búsqueda queda entera de un solo lado
    de la partición.
    """
    out = df.copy()
    ref = out if ajuste is None else out[ajuste.to_numpy()]
    lo_abs, hi_abs = cerca_tukey_log(ref["price_usd"])
    med = out.groupby(col_id)["price_usd"].transform("median")
    rel = out["price_usd"] / med
    absoluto = (out["price_usd"] < lo_abs) | (out["price_usd"] > hi_abs)
    relativo = (rel > K_OUTLIER) | (rel < 1 / K_OUTLIER)
    out["precio_outlier"] = (absoluto | relativo).astype("int8")
    lo, hi = (rel if ajuste is None else rel[ajuste.to_numpy()]).quantile(list(WINSOR))
    out["precio_rel"] = rel.clip(lo, hi)
    out["log_precio_rel"] = np.log(out["precio_rel"])
    out["log_precio"] = np.log(out["price_usd"].clip(lower=1.0))
    return out


def imputar_con_indicador(
    df: pd.DataFrame,
    col: str,
    por: str | list[str] | None = None,
    ajuste: pd.Series | None = None,
) -> pd.DataFrame:
    """Imputa `col` con la mediana (dentro de `por` si se da, con la global de respaldo) y agrega
    `<col>_faltante`. Se usa solo para faltantes MAR; los MNAR estructurales no se imputan.

    `ajuste` (máscara booleana, típicamente Train): las medianas se calculan **solo** con esas
    filas y se aplican a todas. Así Test no aporta información a lo que ve el modelo en Train
    (sin fuga). Un grupo que no aparece en Train recibe la mediana global de Train.
    """
    out = df.copy()
    falta = out[col].isna()
    out[f"{col}_faltante"] = falta.astype("int8")
    ref = out if ajuste is None else out[ajuste.to_numpy()]
    global_ = ref[col].median()
    if por:
        claves = [por] if isinstance(por, str) else list(por)
        medianas = ref.groupby(claves)[col].median().rename("_mediana_grupo")
        relleno = out[claves].join(medianas, on=claves)["_mediana_grupo"]
    else:
        relleno = global_
    out[col] = out[col].fillna(relleno).fillna(global_)
    return out


# ------------------------------------------------------------------------------- faltantes
def contraste_mcar(
    df: pd.DataFrame, col: str, contra: list[str], alpha: float = ALPHA
) -> pd.DataFrame:
    """Bajo MCAR, el indicador de faltante de `col` es independiente de toda variable observada.

    Para cada variable de `contra`: si es binaria, χ² de independencia (tamaño de efecto: V de
    Cramér); si es continua, Mann-Whitney entre filas con y sin faltante (tamaño de efecto:
    diferencia de medias estandarizada). Con N grande casi todo es "significativo", así que la
    lectura se apoya en el tamaño de efecto (|SMD| ≥ 0,1 o V ≥ 0,1 = dependencia apreciable).
    Corrección de Bonferroni por el número de contrastes.
    """
    falta = df[col].isna()
    m = len(contra)
    filas = []
    for v in contra:
        x = df[v]
        ok = x.notna()
        if x.dropna().nunique() <= 2:
            tabla = pd.crosstab(falta[ok], x[ok])
            chi2, p, gl, _ = stats.chi2_contingency(tabla, correction=False)
            efecto, nombre, est = float(np.sqrt(chi2 / tabla.to_numpy().sum())), "V de Cramér", chi2
        else:
            a, b = x[ok & falta].astype(float), x[ok & ~falta].astype(float)
            u = stats.mannwhitneyu(a, b)
            p, est, gl = u.pvalue, u.statistic, None
            sd = np.sqrt((a.var() + b.var()) / 2)
            efecto, nombre = float((a.mean() - b.mean()) / sd), "SMD"
        filas.append(
            {
                "variable_faltante": col,
                "contra": v,
                "pct_faltante": 100 * falta.mean(),
                "estadistico": float(est),
                "gl": gl,
                "p_bonferroni": min(1.0, float(p) * m),
                "efecto": efecto,
                "medida": nombre,
                "dependencia_apreciable": abs(efecto) >= 0.1,
            }
        )
    return pd.DataFrame(filas)


def _em_normal_multivariada(
    x: np.ndarray, max_iter: int = 500, tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Media y covarianza de máxima verosimilitud de una normal multivariada con faltantes (EM)."""
    n, p = x.shape
    obs = ~np.isnan(x)
    patrones = np.unique(obs, axis=0)
    filas_por_patron = [np.all(obs == patron, axis=1) for patron in patrones]
    mu = np.nanmean(x, axis=0)
    sigma = np.diag(np.nanvar(x, axis=0))
    for _ in range(max_iter):
        suma_x = np.zeros(p)
        suma_xx = np.zeros((p, p))
        for o, filas in zip(patrones, filas_por_patron, strict=True):
            m = ~o
            xo = x[filas]
            xhat = np.where(np.isnan(xo), 0.0, xo)
            c = np.zeros((p, p))
            if m.any():
                if o.any():
                    b = sigma[np.ix_(m, o)] @ np.linalg.inv(sigma[np.ix_(o, o)])
                    xhat[:, m] = mu[m] + (xo[:, o] - mu[o]) @ b.T
                    c[np.ix_(m, m)] = sigma[np.ix_(m, m)] - b @ sigma[np.ix_(o, m)]
                else:
                    xhat[:, m] = mu[m]
                    c[np.ix_(m, m)] = sigma[np.ix_(m, m)]
            suma_x += xhat.sum(axis=0)
            suma_xx += xhat.T @ xhat + filas.sum() * c
        mu_nueva = suma_x / n
        sigma_nueva = suma_xx / n - np.outer(mu_nueva, mu_nueva)
        cambio = max(np.abs(mu_nueva - mu).max(), np.abs(sigma_nueva - sigma).max())
        mu, sigma = mu_nueva, sigma_nueva
        if cambio < tol:
            break
    return mu, sigma


def little_mcar(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    r"""Test de Little (1988) de MCAR multivariado.

    Con $\hat\mu$ y $\hat\Sigma$ de máxima verosimilitud (EM) y $J$ patrones de faltantes,
    $d^2 = \sum_j n_j (\bar y_j - \hat\mu_{o_j})^\top \hat\Sigma_{o_j}^{-1} (\bar y_j - \hat\mu_{o_j})$,
    que bajo H0 (MCAR) se distribuye asintóticamente $\chi^2$ con $\sum_j p_j - p$ grados de
    libertad ($\bar y_j$: media de las $p_j$ variables observadas en el patrón $j$). Bajo MCAR la
    media de cada patrón coincide con la global; se rechaza H0 si $d^2$ es grande. Las filas sin
    ninguna variable observada no aportan y se descartan.
    """
    x = df[cols].astype(float).to_numpy()
    x = x[~np.isnan(x).all(axis=1)]
    n, p = x.shape
    mu, sigma = _em_normal_multivariada(x)
    obs = ~np.isnan(x)
    d2, gl, patrones = 0.0, -p, 0
    for patron in np.unique(obs, axis=0):
        filas = np.all(obs == patron, axis=1)
        dif = x[filas][:, patron].mean(axis=0) - mu[patron]
        d2 += filas.sum() * float(dif @ np.linalg.inv(sigma[np.ix_(patron, patron)]) @ dif)
        gl += int(patron.sum())
        patrones += 1
    return {
        "d2": float(d2),
        "gl": int(gl),
        "p_valor": float(stats.chi2.sf(d2, gl)),
        "patrones": patrones,
        "n": int(n),
    }


# ----------------------------------------------------------------------------- competidores
COMPETIDORES_UTILES = (2, 3, 5, 8)  # los demás tienen > 93 % de faltantes en la muestra


def agregar_competidores(df: pd.DataFrame, umbral_diff: float = 100.0) -> pd.DataFrame:
    """Resume los 8 competidores en pocas variables.

    Semántica (fuente primaria, pestaña Data de Kaggle, transcripta por Verónica el 12/11):
    `compN_rate` = +1 si Expedia tiene **menor** precio que el competidor N, 0 si igual, −1 si
    mayor; `compN_rate_percent_diff` es la diferencia porcentual **absoluta** (sin signo).

    * `comp_n_info`: competidores con dato de precio.
    * `comp_n_pos`, `comp_n_neg`: competidores con `rate = +1` (Expedia más barata) y `rate = -1`
      (Expedia más cara).
    * `comp_saldo`: (n_pos − n_neg) / n_info ∈ [−1, 1]; **positivo = Expedia más barata que la
      mayoría de los competidores con dato** (NaN si no hay competidores con dato).
    * `comp_diff_max`: máxima `rate_percent_diff` válida (≤ `umbral_diff` %, el resto se trata
      como error de captura). Es absoluta: no indica quién es más barato.
    """
    out = df.copy()
    rates = out[[f"comp{i}_rate" for i in range(1, 9)]]
    diffs = out[[f"comp{i}_rate_percent_diff" for i in range(1, 9)]]
    out["comp_n_info"] = rates.notna().sum(axis=1).astype("int8")
    out["comp_n_pos"] = (rates == 1).sum(axis=1).astype("int8")
    out["comp_n_neg"] = (rates == -1).sum(axis=1).astype("int8")
    out["comp_saldo"] = (out["comp_n_pos"] - out["comp_n_neg"]) / out["comp_n_info"].replace(
        0, np.nan
    )
    out["comp_diff_max"] = diffs.where(diffs <= umbral_diff).max(axis=1)
    # compN_inv = −1 no está documentado por Kaggle ({+1, 0, NULL}): se trata como faltante y se
    # conserva un indicador (propuesto por Claude Code, aprobado por Verónica el 12/11).
    invs = [f"comp{i}_inv" for i in range(1, 9)]
    no_doc = out[invs] == -1
    out["comp_inv_no_documentado"] = no_doc.any(axis=1).astype("int8")
    out[invs] = out[invs].mask(no_doc)
    return out
