"""Las cifras del README salen de models/resultados.json (regla: cero números escritos a mano)."""

from __future__ import annotations

import importlib.util

import pytest

from tpi.config import ROOT

RUTA_SCRIPT = ROOT / "scripts" / "cifras_readme.py"
RUTA_RES = ROOT / "models" / "resultados.json"


@pytest.mark.skipif(not RUTA_RES.exists(), reason="Sin resultados: ejecutar el notebook")
def test_bloque_de_cifras_del_readme_coincide_con_resultados():
    spec = importlib.util.spec_from_file_location("cifras_readme", RUTA_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.actualizar(check=True), "README desactualizado: python scripts/cifras_readme.py"


def test_informe_y_slides_no_tienen_cifras_de_resultado_sin_codigo():
    """Las secciones de resultados del informe leen el JSON; ningún placeholder queda en el texto."""
    for nombre in ("informe.qmd", "slides.qmd"):
        texto = (ROOT / "report" / nombre).read_text("utf-8")
        assert "resultados.json" in texto
        for marca in ("TODO", "XXX", "lorem", "PENDIENTE"):
            assert marca not in texto, f"{nombre} contiene {marca}"
