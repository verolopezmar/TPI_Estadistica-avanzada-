"""Confidencialidad de los datos de la competencia (reglas de Kaggle, docs/ia_log.md).

Las reglas prohíben compartir el dataset con no participantes y el repo es público. Este test
es la puerta determinística: falla si git trackea cualquier archivo con formato de datos.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tpi.config import ROOT

EXTENSIONES_PROHIBIDAS = (
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".zip",
    ".gz",
    ".7z",
    ".xlsx",
    ".xls",
    ".arff",
    ".pkl",
    ".pickle",
)
# Excepciones explícitas y revisadas (vacía a propósito: agregar acá exige justificar en el PR)
PERMITIDOS: set[str] = set()


def _archivos_trackeados() -> list[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("No es un repositorio git")
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode("utf-8")
    return [f for f in out.split("\0") if f]


def test_ningun_archivo_de_datos_trackeado():
    prohibidos = [
        f
        for f in _archivos_trackeados()
        if f.lower().endswith(EXTENSIONES_PROHIBIDAS) and f not in PERMITIDOS
    ]
    assert prohibidos == [], f"Archivos de datos trackeados por git: {prohibidos}"


def test_gitignore_cubre_datos_y_credenciales():
    reglas = (ROOT / ".gitignore").read_text("utf-8")
    for regla in ["data/raw/*", "data/processed/*", "kaggle.json", "access_token"]:
        assert regla in reglas, f"Falta '{regla}' en .gitignore"
