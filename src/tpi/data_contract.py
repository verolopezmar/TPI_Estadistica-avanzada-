"""Contrato de datos del dataset Expedia (ICDM 2013, `train.csv`).

Contrato entre fases: qué columnas esperamos, con qué tipo, en qué dominio y con qué
centinelas. Separamos dos niveles:

* **Reglas duras** (`validar_contrato`): si se violan, el pipeline se detiene. Son condiciones
  lógicas del sistema (p. ej. no puede haber reserva sin clic).
* **Centinelas** (`reporte_centinelas`): códigos que no son el valor real (0 = "sin reseñas",
  NaN = "usuario sin historial"). No son errores: se cuantifican y se tratan en la Fase 1.

Las semánticas marcadas como HIPÓTESIS provienen de la descripción de la competencia tal como
la recordaba el asistente de IA y **no están verificadas en fuente primaria** (docs/ia_log.md,
Fase 0). Las verifican los tests de `tests/test_data_contract.py` sobre el archivo real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COMPETICION = "expedia-personalized-sort"
ARCHIVO_TRAIN = "train.csv"
N_FILAS_TRAIN_REPORTADO = 9_917_530  # Liu et al. (2013), arXiv:1311.7679 (verificado en el texto)
N_BUSQUEDAS_TRAIN_REPORTADO = 399_344  # ídem

CLAVE_BUSQUEDA = "srch_id"

COLUMNAS_COMPETIDORES = [
    f"comp{i}_{sufijo}" for i in range(1, 9) for sufijo in ("rate", "inv", "rate_percent_diff")
]

# Tipo lógico esperado por columna: id, int, bool, float, datetime, cat
COLUMNAS_ESPERADAS: dict[str, str] = {
    "srch_id": "id",
    "date_time": "datetime",
    "site_id": "cat",
    "visitor_location_country_id": "cat",
    "visitor_hist_starrating": "float",
    "visitor_hist_adr_usd": "float",
    "prop_country_id": "cat",
    "prop_id": "id",
    "prop_starrating": "int",
    "prop_review_score": "float",
    "prop_brand_bool": "bool",
    "prop_location_score1": "float",
    "prop_location_score2": "float",
    "prop_log_historical_price": "float",
    "position": "int",
    "price_usd": "float",
    "promotion_flag": "bool",
    "srch_destination_id": "cat",
    "srch_length_of_stay": "int",
    "srch_booking_window": "int",
    "srch_adults_count": "int",
    "srch_children_count": "int",
    "srch_room_count": "int",
    "srch_saturday_night_bool": "bool",
    "srch_query_affinity_score": "float",
    "orig_destination_distance": "float",
    "random_bool": "bool",
    **{c: "float" for c in COLUMNAS_COMPETIDORES},
    "click_bool": "bool",
    "gross_bookings_usd": "float",
    "booking_bool": "bool",
}

# dtypes de lectura: compactos, toleran NaN (float32) y evitan inferencia por chunk
DTYPES_LECTURA: dict[str, str] = {
    c: {
        "id": "int32",
        "int": "int16",
        "bool": "int8",
        "float": "float32",
        "cat": "int32",
    }[t]
    for c, t in COLUMNAS_ESPERADAS.items()
    if t != "datetime"
}
# columnas enteras que pueden traer NaN se leen como float
DTYPES_LECTURA.update({"prop_starrating": "float32"})


@dataclass(frozen=True)
class Rango:
    """Dominio físico/lógico de una columna (extremos inclusivos; NaN se ignora)."""

    minimo: float | None = None
    maximo: float | None = None
    valores: frozenset | None = None


DOMINIOS: dict[str, Rango] = {
    "prop_starrating": Rango(0, 5),
    "prop_review_score": Rango(0, 5),
    "visitor_hist_starrating": Rango(0, 5),
    "prop_brand_bool": Rango(valores=frozenset({0, 1})),
    "promotion_flag": Rango(valores=frozenset({0, 1})),
    "srch_saturday_night_bool": Rango(valores=frozenset({0, 1})),
    "random_bool": Rango(valores=frozenset({0, 1})),
    "click_bool": Rango(valores=frozenset({0, 1})),
    "booking_bool": Rango(valores=frozenset({0, 1})),
    "position": Rango(minimo=1),
    "srch_length_of_stay": Rango(minimo=1),
    "srch_booking_window": Rango(minimo=0),
    "srch_adults_count": Rango(minimo=1),
    "srch_children_count": Rango(minimo=0),
    "srch_room_count": Rango(minimo=1),
    "price_usd": Rango(minimo=0),
    "gross_bookings_usd": Rango(minimo=0),
    "orig_destination_distance": Rango(minimo=0),
    "prop_log_historical_price": Rango(minimo=0),
    "srch_query_affinity_score": Rango(maximo=0),  # log-probabilidad
    **{f"comp{i}_rate": Rango(valores=frozenset({-1, 0, 1})) for i in range(1, 9)},
    **{f"comp{i}_inv": Rango(valores=frozenset({-1, 0, 1})) for i in range(1, 9)},
}

# Centinelas conocidos o sospechados: columna -> (valor, significado)
CENTINELAS: dict[str, tuple[float, str]] = {
    "prop_review_score": (0.0, "HIPÓTESIS: 0 = hotel sin reseñas (NaN = dato no disponible)"),
    "prop_starrating": (0.0, "HIPÓTESIS: 0 = sin categoría informada / no aplica"),
    "prop_log_historical_price": (0.0, "HIPÓTESIS: 0 = hotel no vendido en el período previo"),
}

# Faltantes estructurales esperables (se diagnostica MCAR/MAR/MNAR en la Fase 1)
FALTANTES_ESPERADOS: dict[str, str] = {
    "visitor_hist_starrating": "HIPÓTESIS MNAR: NaN = usuario sin compras previas (no recurrente)",
    "visitor_hist_adr_usd": "HIPÓTESIS MNAR: ídem visitor_hist_starrating",
    "srch_query_affinity_score": "HIPÓTESIS: NaN = hotel no apareció en búsquedas previas",
    "orig_destination_distance": "HIPÓTESIS MAR: no se pudo calcular la distancia",
    "gross_bookings_usd": "Estructural: solo existe si booking_bool = 1",
    **{c: "Competidor sin dato para ese hotel" for c in COLUMNAS_COMPETIDORES},
}


def columnas_faltantes(df: pd.DataFrame) -> list[str]:
    return [c for c in COLUMNAS_ESPERADAS if c not in df.columns]


def validar_contrato(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de violaciones de reglas duras (vacía = el contrato se cumple)."""
    errores: list[str] = []

    faltan = columnas_faltantes(df)
    if faltan:
        errores.append(f"Faltan columnas: {faltan}")
        return errores

    for col, dom in DOMINIOS.items():
        s = df[col].dropna()
        if dom.valores is not None:
            fuera = set(np.unique(s)) - set(dom.valores)
            if fuera:
                errores.append(f"{col}: valores fuera de dominio {sorted(fuera)[:10]}")
        if dom.minimo is not None and (s < dom.minimo).any():
            errores.append(f"{col}: {(s < dom.minimo).sum()} valores < {dom.minimo}")
        if dom.maximo is not None and (s > dom.maximo).any():
            errores.append(f"{col}: {(s > dom.maximo).sum()} valores > {dom.maximo}")

    # Lógica del embudo: reserva => clic
    n = int(((df["booking_bool"] == 1) & (df["click_bool"] == 0)).sum())
    if n:
        errores.append(f"{n} filas con reserva sin clic (booking_bool=1, click_bool=0)")

    # El gasto solo existe si hubo reserva
    n = int(((df["booking_bool"] == 0) & df["gross_bookings_usd"].notna()).sum())
    if n:
        errores.append(f"{n} filas con gross_bookings_usd sin reserva")

    # random_bool es un atributo de la búsqueda, no del hotel
    n = int((df.groupby(CLAVE_BUSQUEDA)["random_bool"].nunique() > 1).sum())
    if n:
        errores.append(f"{n} búsquedas con random_bool no constante")

    # Posiciones únicas dentro de cada búsqueda
    n = int(df.duplicated([CLAVE_BUSQUEDA, "position"]).sum())
    if n:
        errores.append(f"{n} posiciones repetidas dentro de una misma búsqueda")

    # Un hotel aparece a lo sumo una vez por búsqueda
    n = int(df.duplicated([CLAVE_BUSQUEDA, "prop_id"]).sum())
    if n:
        errores.append(f"{n} hoteles repetidos dentro de una misma búsqueda")

    return errores


def reporte_centinelas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla con % de NaN y % de centinela por columna, para la auditoría forense de la Fase 1."""
    filas = []
    for col in COLUMNAS_ESPERADAS:
        if col not in df.columns:
            continue
        s = df[col]
        fila = {
            "columna": col,
            "pct_nan": 100 * s.isna().mean(),
            "centinela": None,
            "pct_centinela": np.nan,
            "nota": FALTANTES_ESPERADOS.get(col, ""),
        }
        if col in CENTINELAS:
            valor, significado = CENTINELAS[col]
            fila.update(
                centinela=valor,
                pct_centinela=100 * (s == valor).mean(),
                nota=significado,
            )
        filas.append(fila)
    return pd.DataFrame(filas).set_index("columna")
