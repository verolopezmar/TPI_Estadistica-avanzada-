"""Contrato causal genérico: lo único que conoce el pipeline (notebook, app, tests de Fases 1-4).

El pipeline no importa nada específico de Expedia: pide `contrato_activo()` y trabaja con los
roles (T, Y, M, Z, clúster, estrato experimental). Cambiar de dataset es cambiar la variable de
entorno `TPI_DATASET` (`expedia` por defecto, `online_shoppers` = plan B semisintético).

Motivo (docs/ia_log.md, 12/11/2026): las reglas de Kaggle prohíben compartir los datos de Expedia
y los "resulting model parameters" con no participantes, mientras que el TP exige repo y app
públicos. Verónica decidió (12/11) seguir con Expedia; el plan B queda listo sin rehacer la estructura.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tpi.config import PROCESSED_DIR, SEED


@dataclass(frozen=True)
class ContratoCausal:
    nombre: str
    fuente: str
    licencia: str
    ruta_muestra: Path
    tratamiento: str  # columna binaria T, creada por `derivar`
    resultado: str  # Y principal
    mediador: str | None
    confusores: dict[str, tuple[str, ...]]  # grupo del DAG -> columnas
    predictores: tuple[str, ...] = ()  # solo afectan Y (no entran al ajuste Backdoor)
    clave_grupo: str | None = None  # clúster para errores robustos y particiones agrupadas
    estrato_experimental: str | None = None  # 1 = asignación aleatoria de T
    tau_verdadero: str | None = None  # columna con el efecto real (solo semisintético)
    derivar: Callable[[pd.DataFrame], pd.DataFrame] = field(default=lambda df: df)

    @property
    def columnas_z(self) -> list[str]:
        return [c for cols in self.confusores.values() for c in cols]

    def cargar(self) -> pd.DataFrame:
        if not self.ruta_muestra.exists():
            raise FileNotFoundError(
                f"No existe {self.ruta_muestra}. Ver data/README.md para generar la muestra."
            )
        df = self.derivar(pd.read_parquet(self.ruta_muestra))
        faltan = [c for c in self.roles_obligatorios() if c not in df.columns]
        if faltan:
            raise KeyError(f"El contrato '{self.nombre}' no se cumple, faltan: {faltan}")
        return df

    def roles_obligatorios(self) -> list[str]:
        extra = [self.mediador, self.clave_grupo, self.estrato_experimental, self.tau_verdadero]
        return [self.tratamiento, self.resultado, *self.columnas_z, *[c for c in extra if c]]


# ------------------------------------------------------------------------------- Expedia
def _derivar_expedia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # T principal (aprobada por Verónica el 12/11): top-5 de la posición EN PANTALLA.
    df["top5"] = (df["position"] <= 5).astype("int8")
    # T de sensibilidad: top-4 del ranking entre los hoteles presentes en el dataset. Los slots 5,
    # 11, 17 y 23 los ocupan listados que Kaggle no publica (Fase 1, error 16 del ia_log).
    if "srch_id" in df.columns:
        rango = df.groupby("srch_id")["position"].rank(method="first")
        df["top4_dataset"] = (rango <= 4).astype("int8")
    df["usuario_recurrente"] = df["visitor_hist_starrating"].notna().astype("int8")
    return df


EXPEDIA = ContratoCausal(
    nombre="expedia",
    fuente="Kaggle, Personalize Expedia Hotel Searches - ICDM 2013 (expedia-personalized-sort)",
    licencia="Reglas de la competencia: datos confidenciales, prohibido compartir fuera del equipo",
    ruta_muestra=PROCESSED_DIR / "expedia_muestra.parquet",
    tratamiento="top5",
    resultado="booking_bool",
    mediador="click_bool",
    confusores={
        "Zh": (
            "prop_starrating",
            "prop_review_score",
            "prop_brand_bool",
            "prop_location_score1",
            "prop_location_score2",
            "price_usd",
            "promotion_flag",
        ),
        "Zp": ("prop_log_historical_price", "srch_query_affinity_score"),
        "Zb": (
            "srch_length_of_stay",
            "srch_booking_window",
            "srch_adults_count",
            "srch_children_count",
            "srch_room_count",
            "srch_saturday_night_bool",
        ),
        "Zu": ("usuario_recurrente",),
    },
    # Zc (comp*_rate/inv/diff): se clasifica (confusor o predictor) en la Fase 1 con el
    # criterio del equipo: confusor si el ranking la usa Y además afecta la reserva.
    clave_grupo="srch_id",
    estrato_experimental="random_bool",
    derivar=_derivar_expedia,
)


# ------------------------------------------------------------------- Online Shoppers (plan B)
def _derivar_online_shoppers(df: pd.DataFrame) -> pd.DataFrame:
    from tpi.plan_b import generar_semisintetico

    return generar_semisintetico(df, seed=SEED)


ONLINE_SHOPPERS = ContratoCausal(
    nombre="online_shoppers",
    fuente="UCI ML Repository #468, Sakar & Kastro (2018), doi:10.24432/C5F88Q",
    licencia="CC BY 4.0",
    ruta_muestra=PROCESSED_DIR / "online_shoppers.parquet",
    tratamiento="expuesto",  # exposición simulada al recomendador (no "T": choca con DataFrame.T)
    resultado="Y",  # compra semisintética
    mediador="M",  # interacción con el ítem recomendado (simulada)
    confusores={
        "Zs": ("ProductRelated", "ProductRelated_Duration", "PageValues", "BounceRates"),
        "Zu": ("returning",),
        "Zt": ("Weekend", "SpecialDay"),
    },
    tau_verdadero="tau_real",
    derivar=_derivar_online_shoppers,
)

CONTRATOS: dict[str, ContratoCausal] = {c.nombre: c for c in (EXPEDIA, ONLINE_SHOPPERS)}


def contrato_activo() -> ContratoCausal:
    nombre = os.environ.get("TPI_DATASET", "expedia")
    if nombre not in CONTRATOS:
        raise ValueError(f"TPI_DATASET={nombre!r} no existe; opciones: {sorted(CONTRATOS)}")
    return CONTRATOS[nombre]


__all__ = ["CONTRATOS", "EXPEDIA", "ONLINE_SHOPPERS", "ContratoCausal", "contrato_activo"]
