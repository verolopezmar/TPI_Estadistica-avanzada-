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


def preparar(crudo: pd.DataFrame) -> pd.DataFrame:
    df, _ = aplicar_centinelas(crudo)
    df["slot_especial"] = df["position"].isin(SLOTS_ESPECIALES).astype("int8")
    df = precio_relativo(df)
    df["usuario_recurrente"] = df["visitor_hist_starrating"].notna().astype("int8")
    df["con_afinidad"] = df["srch_query_affinity_score"].notna().astype("int8")
    df = imputar_con_indicador(df, "prop_location_score2", por=["prop_country_id"])
    df = imputar_con_indicador(df, "prop_review_score", por=["prop_starrating"])
    df = imputar_con_indicador(df, "prop_starrating")
    df = imputar_con_indicador(df, "prop_log_historical_price", por=["prop_country_id"])
    df = imputar_con_indicador(
        df, "orig_destination_distance", por=["visitor_location_country_id", "prop_country_id"]
    )
    df = agregar_competidores(df)
    df["log_pos"] = np.log(df["position"])
    df["train"] = es_train(df)
    return df
