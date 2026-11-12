"""Tests de Fase 0: estructura del repo y reproducibilidad básica."""

import numpy as np

from tpi import config


def test_seed_es_42():
    assert config.SEED == 42


def test_set_global_seed_es_determinista():
    a = config.set_global_seed().normal(size=5)
    b = config.set_global_seed().normal(size=5)
    np.testing.assert_array_equal(a, b)


def test_estructura_de_carpetas():
    for carpeta in ["src", "data", "notebooks", "app", "tests", "docs", "report", "models"]:
        assert (config.ROOT / carpeta).is_dir(), f"Falta la carpeta /{carpeta}"


def test_rutas_son_relativas_a_la_raiz():
    for ruta in [config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.FIGURES_DIR]:
        assert config.ROOT in ruta.parents


def test_archivos_de_gobernanza_existen():
    for archivo in ["README.md", "BITACORA.md", "requirements.txt", "docs/ia_log.md"]:
        assert (config.ROOT / archivo).is_file(), f"Falta {archivo}"
