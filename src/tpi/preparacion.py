"""Pipeline de preparación de la Fase 1 en una sola función (la usan las Fases 3-5 y la
regeneración de artefactos). El notebook ejecuta los mismos pasos uno por uno con su
justificación y verifica que coinciden con `preparar()`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tpi.auditoria import (
    agregar_competidores,
    aplicar_centinelas,
    imputar_con_indicador,
    precio_relativo,
)
from tpi.particion import es_train

SLOTS_ESPECIALES = (5, 11, 17, 23)  # Fase 1, sección 1.1 (test_slots_estructurales...)
# Faltantes MAR que se imputan (Fase 1, sección 1.3): (variable, agrupamiento de la mediana)
IMPUTACIONES: tuple[tuple[str, list[str] | None], ...] = (
    ("prop_location_score2", ["prop_country_id"]),
    ("prop_review_score", ["prop_starrating"]),
    ("prop_starrating", None),
    ("prop_log_historical_price", ["prop_country_id"]),
    ("orig_destination_distance", ["visitor_location_country_id", "prop_country_id"]),
)


def preparar(crudo: pd.DataFrame) -> pd.DataFrame:
    """Preparación completa sin fuga: la partición depende solo de `srch_id`, `random_bool` y la
    semilla, así que se define primero, y toda estadística de transformación (cercas de Tukey,
    cuantiles de winsorización, medianas de imputación) se ajusta solo con Train."""
    df, _ = aplicar_centinelas(crudo)
    df["train"] = es_train(df)
    tr = df["train"]
    df["slot_especial"] = df["position"].isin(SLOTS_ESPECIALES).astype("int8")
    df = precio_relativo(df, ajuste=tr)
    df["usuario_recurrente"] = df["visitor_hist_starrating"].notna().astype("int8")
    df["con_afinidad"] = df["srch_query_affinity_score"].notna().astype("int8")
    for col, por in IMPUTACIONES:
        df = imputar_con_indicador(df, col, por=por, ajuste=tr)
    df = agregar_competidores(df)
    df["log_pos"] = np.log(df["position"])
    return df
