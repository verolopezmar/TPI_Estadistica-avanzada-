"""Módulo de inferencia de la Fase 1: contra referencias externas y reglas de reporte."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg
import pytest

from tpi.config import SEED
from tpi.inferencia import (
    anova_un_factor,
    contraste_pareado,
    efecto_minimo_detectable,
    mann_whitney,
    posthoc,
    potencia,
    welch_dos_grupos,
)

rng = np.random.default_rng(SEED)
A = rng.normal(10.5, 2, 400)
B = rng.normal(10.0, 4, 300)


def test_welch_coincide_con_pingouin():
    r = welch_dos_grupos(A, B)
    ref = pg.ttest(A, B, correction=True).iloc[0]
    assert r.valor == pytest.approx(ref["T"])
    assert r.gl == pytest.approx(ref["dof"])
    assert r.p_valor == pytest.approx(ref["p_val"])
    assert r.ic95[0] == pytest.approx(ref["CI95"][0], abs=1e-2)
    assert r.efecto == pytest.approx(pg.compute_effsize(A, B, eftype="hedges"), rel=1e-3)


def test_mann_whitney_r_biserial_coincide_con_pingouin():
    r = mann_whitney(A, B)
    ref = pg.mwu(A, B).iloc[0]
    assert r.p_valor == pytest.approx(ref["p_val"])
    assert r.efecto == pytest.approx(ref["RBC"], abs=1e-9)


def test_pareado_dz_y_wilcoxon():
    d = rng.normal(0.2, 1, 500)
    r = contraste_pareado(d)
    assert r.efecto == pytest.approx(d.mean() / d.std(ddof=1))
    assert r.ic95[0] < d.mean() < r.ic95[1]
    assert 0 <= r.extra["wilcoxon_p"] <= 1


def test_texto_nunca_acepta_h0():
    for r in (welch_dos_grupos(A, A + 1e-9), welch_dos_grupos(A, B), contraste_pareado(B - 10)):
        t = r.texto()
        assert "acepta" not in t
        assert ("no se rechaza H0" in t) != r.rechaza_h0
        assert "p = " in t and "IC 95 %" in t


def test_potencia_valores_clasicos():
    """Cohen (1988): d = 0,5 con n = 64 por grupo da potencia ≈ 0,80 a α = 0,05."""
    assert potencia(0.5, 64, 64) == pytest.approx(0.80, abs=0.01)
    assert efecto_minimo_detectable(64, 64) == pytest.approx(0.5, abs=0.01)


@pytest.fixture(scope="module")
def grupos() -> pd.DataFrame:
    medias, desvios, ns = [0.0, 0.3, 0.6, 0.6], [1.0, 1.0, 3.0, 1.0], [200, 300, 250, 150]
    return pd.DataFrame(
        {
            "y": np.concatenate(
                [rng.normal(m, s, n) for m, s, n in zip(medias, desvios, ns, strict=True)]
            ),
            "g": np.repeat(["a", "b", "c", "d"], ns),
        }
    )


def test_anova_detecta_heterocedasticidad_y_coincide_con_pingouin(grupos):
    r = anova_un_factor(grupos, "y", "g")
    assert r["levene"].rechaza_h0
    ref = pg.welch_anova(data=grupos, dv="y", between="g").iloc[0]
    assert r["welch"].valor == pytest.approx(ref["F"])
    assert r["clasico"].gl == (3, len(grupos) - 4)
    assert 0 <= r["kruskal"].efecto <= 1 and 0 <= r["clasico"].efecto <= 1


def test_posthoc_tiene_todos_los_pares_y_tres_criterios(grupos):
    t = posthoc(grupos, "y", "g")
    assert len(t) == 6
    assert t[["p_games_howell", "p_bonferroni", "p_tukey"]].notna().all().all()
