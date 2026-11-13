"""Lógica de la app, sin Streamlit (testeable con pytest). Solo lee artefactos agregados."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "models"


@lru_cache(maxsize=1)
def cargar_artefactos(ruta: Path = MODELOS / "artefactos_app.json") -> dict:
    return json.loads(Path(ruta).read_text("utf-8"))


@lru_cache(maxsize=1)
def cargar_modelo(ruta: Path = MODELOS / "rf_app.joblib"):
    return joblib.load(ruta)


def perfil_completo(entradas: dict, art: dict) -> tuple[pd.DataFrame, list[str]]:
    """Completa con medianas de Train las features no ingresadas y recorta al rango [p1, p99] de
    Train (fuera de ese rango el modelo extrapola). Devuelve (fila, avisos)."""
    meta = art["modelo"]
    fila, avisos = {}, []
    for f in meta["features"]:
        v = float(entradas.get(f, meta["medianas_train"][f]))
        lo, hi = meta["p01"][f], meta["p99"][f]
        if v < lo or v > hi:
            avisos.append(f"{f} = {v:g} fuera del rango observado [{lo:g}; {hi:g}]: se recorta.")
            v = min(max(v, lo), hi)
        fila[f] = v
    return pd.DataFrame([fila])[meta["features"]], avisos


def predecir(modelo, x: pd.DataFrame) -> dict:
    """Probabilidad del bosque y dispersión entre árboles (percentiles 5-95 de las predicciones de
    cada árbol: intervalo de variabilidad del ensamble, no un IC frecuentista)."""
    por_arbol = np.array([a.predict_proba(x.to_numpy())[0, 1] for a in modelo.estimators_])
    return {
        "p": float(modelo.predict_proba(x)[0, 1]),
        "p05": float(np.percentile(por_arbol, 5)),
        "p95": float(np.percentile(por_arbol, 95)),
    }


def explicar_local(modelo, x: pd.DataFrame, art: dict, top: int = 8) -> pd.Series:
    """Contribución ceteris paribus: cuánto cambia P(reserva) si cada feature vuelve a su mediana
    de Train (positivo = el valor ingresado sube la probabilidad respecto de la mediana)."""
    p0 = modelo.predict_proba(x)[0, 1]
    contrib = {}
    for f, med in art["modelo"]["medianas_train"].items():
        x2 = x.copy()
        x2[f] = med
        contrib[f] = p0 - modelo.predict_proba(x2)[0, 1]
    s = pd.Series(contrib)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top)


def cate_perfil(art: dict, estrellas: int, log_precio_rel: float, recurrente: int) -> dict:
    """CATE precalculado (CausalForestDML, R = 0) para el punto de grilla más cercano."""
    g = pd.DataFrame(art["causal"]["cate_grilla"])
    d = g[(g.estrellas == estrellas) & (g.usuario_recurrente == recurrente)]
    return d.iloc[(d.log_precio_rel - log_precio_rel).abs().argmin()].to_dict()


def proyectar_pca(art: dict, valores: dict) -> tuple[float, float]:
    """Proyecta un caso en el plano PC1-PC2 con la media, el desvío y los autovectores de Train."""
    p = art["pca"]
    z = (np.array([valores[v] for v in p["variables"]]) - np.array(p["media"])) / np.array(
        p["desvio"]
    )
    comp = np.array(p["componentes"])
    return float(z @ comp[0]), float(z @ comp[1])


def semaforo(refutaciones: list[dict]) -> str:
    """verde: pasan todas; amarillo: falla una; rojo: fallan dos o más."""
    fallas = sum(not r["pasa"] for r in refutaciones)
    return "verde" if fallas == 0 else ("amarillo" if fallas == 1 else "rojo")
