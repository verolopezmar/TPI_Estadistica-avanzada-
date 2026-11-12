"""Contrato causal genérico y plan B semisintético.

Garantizan que el pipeline puede cambiar de dataset (TPI_DATASET) sin rehacer la estructura y
que el generador semisintético tiene la confusión y el efecto que declara.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tpi.config import SEED
from tpi.contrato import CONTRATOS, EXPEDIA, ONLINE_SHOPPERS, contrato_activo
from tpi.plan_b import COLUMNAS_REALES, generar_semisintetico


def test_contrato_por_defecto_es_expedia(monkeypatch):
    monkeypatch.delenv("TPI_DATASET", raising=False)
    assert contrato_activo() is EXPEDIA


def test_contrato_se_cambia_por_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("TPI_DATASET", "online_shoppers")
    assert contrato_activo() is ONLINE_SHOPPERS
    monkeypatch.setenv("TPI_DATASET", "titanic")
    with pytest.raises(ValueError):
        contrato_activo()


@pytest.mark.parametrize("nombre", sorted(CONTRATOS))
def test_todo_contrato_declara_los_roles_del_enunciado(nombre):
    c = CONTRATOS[nombre]
    assert c.tratamiento and c.resultado and c.mediador and c.columnas_z
    assert c.tratamiento not in c.columnas_z and c.resultado not in c.columnas_z
    assert c.mediador not in c.columnas_z, "Ajustar por el mediador viola Backdoor"


def test_expedia_deriva_tratamiento_top5():
    df = pd.DataFrame(
        {"position": [1, 5, 6, 30], "visitor_hist_starrating": [np.nan, 3, 4, np.nan]}
    )
    out = EXPEDIA.derivar(df)
    assert out["top5"].tolist() == [1, 1, 0, 0]
    assert out["usuario_recurrente"].tolist() == [0, 1, 1, 0]


# ----------------------------------------------------------------------- plan B semisintético
def _online_shoppers_sintetico(n: int = 20_000) -> pd.DataFrame:
    """Imita la estructura de UCI #468 (sin descargar nada)."""
    rng = np.random.default_rng(SEED)
    pv = rng.exponential(5, n) * (rng.uniform(size=n) < 0.25)
    return pd.DataFrame(
        {
            "ProductRelated": rng.poisson(30, n),
            "ProductRelated_Duration": rng.exponential(1000, n),
            "PageValues": pv,
            "BounceRates": rng.uniform(0, 0.2, n),
            "VisitorType": rng.choice(["Returning_Visitor", "New_Visitor"], n, p=[0.85, 0.15]),
            "Weekend": rng.uniform(size=n) < 0.23,
            "SpecialDay": rng.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0], n),
            "Revenue": rng.uniform(size=n) < 0.05 + 0.5 * (pv > 0),
        }
    )


@pytest.fixture(scope="module")
def semi() -> pd.DataFrame:
    return generar_semisintetico(_online_shoppers_sintetico(), seed=SEED)


def test_semisintetico_es_determinista():
    a = generar_semisintetico(_online_shoppers_sintetico(2000), seed=SEED)
    b = generar_semisintetico(_online_shoppers_sintetico(2000), seed=SEED)
    pd.testing.assert_frame_equal(a, b)


def test_semisintetico_rechaza_columnas_faltantes():
    with pytest.raises(KeyError):
        generar_semisintetico(_online_shoppers_sintetico(100).drop(columns=COLUMNAS_REALES[0]))


def test_semisintetico_tiene_confusion_por_z(semi):
    """El 'recomendador' expone más a sesiones con PageValues alto: hay sesgo ingenuo."""
    ate = semi["tau_real"].mean()
    ingenuo = (
        semi.loc[semi["expuesto"] == 1, "Y"].mean() - semi.loc[semi["expuesto"] == 0, "Y"].mean()
    )
    assert 0 < ate < 0.1
    assert ingenuo - ate > 0.02


def test_ipw_con_propension_real_recupera_el_ate(semi):
    """Con e(Z) verdadera, el estimador de Hájek es consistente: valida la verdad declarada."""
    t, y, e = semi["expuesto"], semi["Y"], semi["e_real"]
    hajek = (t * y / e).sum() / (t / e).sum() - ((1 - t) * y / (1 - e)).sum() / (
        (1 - t) / (1 - e)
    ).sum()
    assert hajek == pytest.approx(semi["tau_real"].mean(), abs=0.02)


def test_semisintetico_cumple_su_contrato(semi):
    faltan = [c for c in ONLINE_SHOPPERS.roles_obligatorios() if c not in semi.columns]
    assert faltan == []


@pytest.mark.parametrize("nombre", sorted(CONTRATOS))
def test_roles_no_chocan_con_atributos_de_dataframe(nombre):
    """`df.T` es la transpuesta, no la columna "T" (error 13 del ia_log): un rol con ese nombre
    devuelve resultados silenciosamente erróneos con la notación de punto."""
    c = CONTRATOS[nombre]
    roles = [c.tratamiento, c.resultado, c.mediador, *c.columnas_z]
    assert not [r for r in roles if r and hasattr(pd.DataFrame, r)]
