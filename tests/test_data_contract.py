"""Contrato de datos y premisa experimental de `random_bool`.

Dos bloques:
* Tests sintéticos (siempre corren): verifican que las funciones de validación detectan lo que
  deben detectar. Sin esto, un contrato "en verde" no prueba nada.
* Tests sobre los datos reales (marca `data`): se saltean si todavía no se corrió
  `python data/download_data.py`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tpi.aleatorizacion import MARGEN_EQUIVALENCIA, resumir_rho, rho_intra_busqueda
from tpi.config import PROCESSED_DIR, SEED
from tpi.data_contract import (
    COLUMNAS_ESPERADAS,
    N_BUSQUEDAS_TRAIN_REPORTADO,
    N_FILAS_TRAIN_REPORTADO,
    reporte_centinelas,
    validar_contrato,
)
from tpi.sampling import muestrear_busquedas

RUTA_MUESTRA = PROCESSED_DIR / "expedia_muestra.parquet"
RUTA_MANIFIESTO = PROCESSED_DIR / "manifiesto.json"


# --------------------------------------------------------------------------- fixtures sintéticas
def _df_sintetico(n_busquedas: int = 200, n_hoteles: int = 20, aleatorio: bool = True):
    """Listas sintéticas que cumplen el contrato; si `aleatorio=False`, ordenadas por calidad."""
    rng = np.random.default_rng(SEED)
    filas = []
    for s in range(n_busquedas):
        calidad = rng.uniform(1, 5, n_hoteles).round(1)
        orden = rng.permutation(n_hoteles) if aleatorio else np.argsort(-calidad)
        pos = np.empty(n_hoteles, dtype=int)
        pos[orden] = np.arange(1, n_hoteles + 1)
        click = (rng.uniform(size=n_hoteles) < 0.05).astype(int)
        book = click * (rng.uniform(size=n_hoteles) < 0.5)
        for h in range(n_hoteles):
            filas.append(
                {
                    "srch_id": s,
                    "prop_id": h,
                    "position": pos[h],
                    "prop_review_score": calidad[h],
                    "random_bool": int(aleatorio),
                    "click_bool": click[h],
                    "booking_bool": book[h],
                    "gross_bookings_usd": 100.0 if book[h] else np.nan,
                }
            )
    df = pd.DataFrame(filas)
    for c in COLUMNAS_ESPERADAS:  # el resto de columnas, en NaN (permitido por el contrato)
        if c not in df.columns:
            df[c] = np.nan
    return df


# ----------------------------------------------------------------------- contrato (sintético)
def test_contrato_acepta_datos_validos():
    assert validar_contrato(_df_sintetico()) == []


def test_contrato_detecta_columnas_faltantes():
    errores = validar_contrato(_df_sintetico().drop(columns=["random_bool"]))
    assert errores and "Faltan columnas" in errores[0]


def test_contrato_detecta_reserva_sin_clic():
    df = _df_sintetico()
    df.loc[0, ["click_bool", "booking_bool", "gross_bookings_usd"]] = [0, 1, 50.0]
    assert any("reserva sin clic" in e for e in validar_contrato(df))


def test_contrato_detecta_random_bool_no_constante_en_busqueda():
    df = _df_sintetico()
    df.loc[0, "random_bool"] = 0
    assert any("random_bool no constante" in e for e in validar_contrato(df))


def test_contrato_detecta_valores_fuera_de_dominio():
    df = _df_sintetico()
    df.loc[0, "prop_review_score"] = 7.5
    assert any("prop_review_score" in e for e in validar_contrato(df))


def test_reporte_centinelas_cuantifica_ceros():
    df = _df_sintetico()
    df.loc[:9, "prop_review_score"] = 0.0
    rep = reporte_centinelas(df)
    assert rep.loc["prop_review_score", "pct_centinela"] == pytest.approx(100 * 10 / len(df))


# ------------------------------------------------------------------------ muestreo (sintético)
def test_muestreo_es_determinista_y_estratificado():
    busq = pd.DataFrame({"srch_id": np.arange(1000), "random_bool": np.arange(1000) % 3 == 0})
    a = muestrear_busquedas(busq, 50, seed=SEED)
    b = muestrear_busquedas(busq.sample(frac=1, random_state=1), 50, seed=SEED)  # otro orden
    np.testing.assert_array_equal(a, b)
    elegidos = busq.set_index("srch_id").loc[a, "random_bool"]
    assert elegidos.value_counts().to_dict() == {True: 50, False: 50}


def test_muestreo_rechaza_busquedas_duplicadas():
    busq = pd.DataFrame({"srch_id": [1, 1], "random_bool": [0, 1]})
    with pytest.raises(ValueError):
        muestrear_busquedas(busq, 1)


# ------------------------------------------------------------- aleatorización (sintético)
def test_rho_detecta_orden_aleatorio():
    # Equivalencia requiere potencia: con 200 búsquedas el IC 95 % de E[rho] mide ~±0,031 y no
    # cabe en ±0,03 aunque rho_medio ≈ 0 (falla registrada en ia_log, Fase 0). Con 2000, ~±0,01.
    df = _df_sintetico(n_busquedas=2000, aleatorio=True)
    r = resumir_rho(rho_intra_busqueda(df, "prop_review_score"), "q")
    assert r.equivalente_a_cero, r


def test_rho_detecta_orden_por_calidad():
    r = resumir_rho(rho_intra_busqueda(_df_sintetico(aleatorio=False), "prop_review_score"), "q")
    assert r.rho_medio < -0.9 and r.p_valor < 1e-6


# ------------------------------------------------------------------------- datos reales
requiere_datos = pytest.mark.skipif(
    not RUTA_MUESTRA.exists(), reason="Sin datos: correr `python data/download_data.py`"
)


@pytest.fixture(scope="module")
def muestra() -> pd.DataFrame:
    return pd.read_parquet(RUTA_MUESTRA)


@pytest.mark.data
@requiere_datos
def test_datos_reales_cumplen_contrato(muestra):
    assert validar_contrato(muestra) == []


@pytest.mark.data
@requiere_datos
def test_tamano_train_coincide_con_lo_reportado():
    """Refuta o confirma la cifra de Liu et al. (2013) citada en el informe."""
    m = json.loads(RUTA_MANIFIESTO.read_text("utf-8"))
    assert m["train_n_filas"] == N_FILAS_TRAIN_REPORTADO
    assert m["train_n_busquedas"] == N_BUSQUEDAS_TRAIN_REPORTADO
    assert m["train_busquedas_random_bool_inconsistente"] == 0


CALIDAD = ["prop_review_score", "prop_starrating", "prop_location_score2", "price_usd"]


@pytest.mark.data
@requiere_datos
@pytest.mark.parametrize("variable", CALIDAD)
def test_random_bool_1_orden_independiente_de_atributos(muestra, variable):
    """Premisa del benchmark experimental. Si falla, se detiene el diseño (ver ia_log, Fase 0)."""
    sub = muestra[muestra["random_bool"] == 1]
    r = resumir_rho(rho_intra_busqueda(sub, variable), variable)
    assert r.equivalente_a_cero, (
        f"{variable}: E[rho]={r.rho_medio:.4f}, IC95=[{r.ic95_inf:.4f}, {r.ic95_sup:.4f}] "
        f"excede ±{MARGEN_EQUIVALENCIA}"
    )


@pytest.mark.data
@requiere_datos
def test_random_bool_0_orden_depende_de_la_calidad(muestra):
    """Con el orden del algoritmo, al menos un atributo de calidad debe asociarse a la posición.

    Si ninguno se asocia, no hay confusión que corregir y el diseño pierde sentido.
    """
    sub = muestra[muestra["random_bool"] == 0]
    resumenes = [resumir_rho(rho_intra_busqueda(sub, v), v) for v in CALIDAD]
    assert any(abs(r.rho_medio) > MARGEN_EQUIVALENCIA and r.p_valor < 0.001 for r in resumenes), (
        resumenes
    )
