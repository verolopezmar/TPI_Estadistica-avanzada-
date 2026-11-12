"""App (Fase 5): confidencialidad de los artefactos, rangos de entrada y ejecución sin errores.

Los artefactos se versionan en models/, así que estos tests corren también en CI (sin datos).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from tpi.artefactos import IDENTIFICADORES, K_MIN, MIN_HOJA, claves_prohibidas
from tpi.config import MODELS_DIR, ROOT

RUTA_ART = MODELS_DIR / "artefactos_app.json"
RUTA_MOD = MODELS_DIR / "rf_app.joblib"
requiere_artefactos = pytest.mark.skipif(
    not (RUTA_ART.exists() and RUTA_MOD.exists()), reason="Sin artefactos: ejecutar el notebook"
)
sys.path.insert(0, str(ROOT / "app"))


@pytest.fixture(scope="module")
def art():
    return json.loads(RUTA_ART.read_text("utf-8"))


def _conteos(obj):
    """Todas las listas de conteos de los artefactos (histogramas y grillas)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "conteos":
                yield np.asarray(v, dtype=float).ravel()
            else:
                yield from _conteos(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _conteos(v)


@requiere_artefactos
def test_artefactos_sin_identificadores(art):
    assert claves_prohibidas(art) == []
    texto = RUTA_ART.read_text("utf-8")
    assert not any(f'"{i}"' in texto for i in IDENTIFICADORES)


@requiere_artefactos
def test_ninguna_celda_publicada_describe_menos_de_k_filas(art):
    for c in _conteos(art):
        assert not ((c > 0) & (c < K_MIN)).any()


@requiere_artefactos
def test_curvas_por_posicion_respetan_k(art):
    for k in ("R0", "R1"):
        assert min(art["ctr_posicion"][k]["n"]) >= K_MIN


@requiere_artefactos
def test_modelo_de_la_app_tiene_hojas_de_al_menos_k_filas():
    import joblib

    m = joblib.load(RUTA_MOD)
    assert m.min_samples_leaf >= MIN_HOJA
    assert RUTA_MOD.stat().st_size < 50e6  # liviano para el repo y Streamlit Cloud


@requiere_artefactos
def test_prediccion_en_rango_y_entradas_recortadas(art):
    from logica import cargar_modelo, perfil_completo, predecir

    x, avisos = perfil_completo({"position": 500, "prop_starrating": 3}, art)
    assert avisos and x["position"].iloc[0] <= art["modelo"]["p99"]["position"]
    r = predecir(cargar_modelo(), x)
    assert 0 <= r["p05"] <= r["p"] <= 1 and r["p95"] <= 1


@requiere_artefactos
def test_semaforo_y_cate(art):
    from logica import cate_perfil, semaforo

    assert semaforo([{"pasa": True}] * 3) == "verde"
    assert semaforo([{"pasa": True}, {"pasa": False}, {"pasa": False}]) == "rojo"
    c = cate_perfil(art, 3, 0.1, 0)
    assert c["ic_inf"] <= c["cate"] <= c["ic_sup"]


@requiere_artefactos
def test_app_corre_sin_excepciones():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
