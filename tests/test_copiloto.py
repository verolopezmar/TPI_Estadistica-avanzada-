"""Copiloto (bonus): herramientas de solo lectura, guardarraíl contra cifras inventadas y loop de
tool use con un cliente simulado (sin llamadas reales a la API)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace as NS

import pytest

from tpi.config import MODELS_DIR, ROOT

sys.path.insert(0, str(ROOT / "app"))
RUTA_ART = MODELS_DIR / "artefactos_app.json"
requiere_artefactos = pytest.mark.skipif(not RUTA_ART.exists(), reason="Sin artefactos")


@pytest.fixture(scope="module")
def art():
    return json.loads(RUTA_ART.read_text("utf-8"))


def test_numeros_en_formato_espanol():
    from copiloto import numeros

    assert numeros("OR 2,11 contra 3,38; 1.234,5 USD y -0,7 %") == [2.11, 3.38, 1234.5, -0.7]


def test_guardarrail_acepta_cifras_redondeadas_y_rechaza_inventadas():
    from copiloto import cifras_no_verificadas

    resultados = [{"salida": {"or_benchmark": 2.1125, "cate_pp": 10.42}}]
    assert cifras_no_verificadas("El OR es 2,11 y el CATE 10,4 p.p. en el top-5.", resultados) == []
    assert cifras_no_verificadas("El OR es 2,11 y el efecto llega a 37 %.", resultados) == [37.0]


def test_guardarrail_acepta_cifras_de_la_pregunta():
    from copiloto import cifras_no_verificadas

    assert (
        cifras_no_verificadas(
            "Para hoteles 20 % más baratos...", [], pregunta="¿y si es 20 % más barato?"
        )
        == []
    )


class _ClienteFalso:
    """Imita client.beta.messages.create con respuestas guionadas."""

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.llamadas.append(kw)
        return self.respuestas.pop(0)


def _uso(nombre, entrada, id_="t1"):
    return NS(type="tool_use", name=nombre, input=entrada, id=id_)


def _texto(t):
    return NS(type="text", text=t)


@requiere_artefactos
def test_loop_usa_herramienta_y_responde_con_cifras_verificadas(art):
    from copiloto import responder

    orb = round(art["causal"]["or_benchmark"], 2)
    cliente = _ClienteFalso(
        [
            NS(stop_reason="tool_use", content=[_uso("efecto_causal_top5", {})]),
            NS(
                stop_reason="end_turn",
                content=[
                    _texto(
                        f"Con orden aleatorio, el OR del top-5 es {orb:.2f}.".replace(
                            ".", ",", 1
                        ).replace(",", ".", 0)
                    )
                ],
            ),
        ]
    )
    r = responder("¿Cuánto vale el top-5?", cliente, art)
    assert not r["bloqueado"], r
    assert r["resultados"][0]["herramienta"] == "efecto_causal_top5"
    # el resultado de la herramienta vuelve al modelo como tool_result con el mismo id
    # mensajes: [pregunta, assistant con tool_use, user con tool_result, ...]
    resultado_herramienta = cliente.llamadas[1]["messages"][2]
    assert resultado_herramienta["role"] == "user"
    assert resultado_herramienta["content"][0]["tool_use_id"] == "t1"
    assert cliente.llamadas[0]["fallbacks"] == "default"


@requiere_artefactos
def test_loop_bloquea_cifras_inventadas(art):
    from copiloto import responder

    cliente = _ClienteFalso(
        [
            NS(stop_reason="tool_use", content=[_uso("refutaciones", {})]),
            NS(
                stop_reason="end_turn",
                content=[
                    _texto(
                        "Aplicar la política a clientes de más de 45 años sumaría 12,7 % de reservas."
                    )
                ],
            ),
        ]
    )
    r = responder("¿Y para clientes de más de 45 años?", cliente, art)
    assert r["bloqueado"] and 12.7 in r["no_verificadas"]


@requiere_artefactos
def test_loop_maneja_rechazo(art):
    from copiloto import responder

    r = responder("...", _ClienteFalso([NS(stop_reason="refusal", content=[])]), art)
    assert r["bloqueado"]


@requiere_artefactos
def test_herramientas_devuelven_solo_agregados(art):
    from copiloto import ejecutar_herramienta
    from logica import cargar_modelo

    for nombre, entrada in [
        ("efecto_causal_top5", {}),
        (
            "cate_por_perfil",
            {"estrellas": 3, "precio_vs_busqueda_pct": -20, "usuario_recurrente": False},
        ),
        ("refutaciones", {}),
        ("escenarios_valuacion", {}),
        (
            "prediccion_reserva",
            {
                "estrellas": 4,
                "precio_noche_usd": 120,
                "precio_vs_busqueda_pct": 0,
                "posicion": 3,
                "en_promocion": True,
                "usuario_recurrente": False,
            },
        ),
    ]:
        salida = ejecutar_herramienta(nombre, entrada, art, cargar_modelo())
        texto = json.dumps(salida)
        assert "srch_id" not in texto and "prop_id" not in texto
