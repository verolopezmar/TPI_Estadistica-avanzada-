"""Smoke test del entorno: cada dependencia con binarios compilados debe importar.

Motivación (docs/ia_log.md, Fase 0): pip instaló pyarrow 26 sobre numpy 1.26 sin avisar de la
incompatibilidad, y Smart App Control de Windows bloqueó la DLL de pyzmq 27.3.0. Ambos fallos
aparecían recién al ejecutar el notebook o Quarto; este test los detecta en `pytest`.
"""

import importlib
import io

import pytest

MODULOS_CRITICOS = [
    "numpy",
    "pandas",
    "scipy.stats",
    "statsmodels.api",
    "pingouin",
    "sklearn.ensemble",
    "dowhy",
    "econml.dml",
    "networkx",
    "matplotlib.pyplot",
    "seaborn",
    "streamlit",
    "pyarrow.parquet",
    "zmq",
    "inflate64",
    "ipykernel",
    "nbclient",
]


@pytest.mark.parametrize("modulo", MODULOS_CRITICOS)
def test_modulo_importa(modulo):
    importlib.import_module(modulo)


def test_numpy_es_1x_por_compatibilidad_con_econml():
    import numpy as np

    assert np.__version__.startswith("1."), "econml 0.15.1 / dowhy 0.12 requieren numpy<2"


def test_parquet_ida_y_vuelta():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    buf = io.BytesIO()
    df.to_parquet(buf)
    buf.seek(0)
    pd.testing.assert_frame_equal(pd.read_parquet(buf), df)
