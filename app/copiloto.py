"""Copiloto de decisión (bonus del enunciado): asistente conversacional con tool use sobre el motor
causal y predictivo de la app.

Diseño (propuesto por Claude Code; ver docs/ia_log.md):
* Herramientas de **solo lectura** sobre los artefactos agregados (`models/artefactos_app.json`) y
  el RF compacto: nunca ven datos individuales.
* **Guardarraíl contra cifras inventadas:** toda cifra de la respuesta final tiene que aparecer en
  algún resultado de herramienta (o en la pregunta). Si no, la respuesta se bloquea y se muestran
  los resultados crudos de las herramientas.
* Loop manual de tool use (Messages API) con `fallbacks` del lado del servidor ante rechazos.
* Sin clave de API configurada, la app muestra cómo activarlo; los tests usan un cliente simulado.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

MODELO = "claude-opus-5-5"
MAX_PASOS = 6

SISTEMA = """Sos el copiloto de decisión de un estudio sobre el ranking de hoteles de Expedia (datos del \
desafío ICDM 2013, solo búsquedas con al menos un clic). Respondés en español, en tono sobrio y breve, a \
directivos que deciden cómo asignar el top-5 de la lista de resultados.

Reglas:
1. Toda cifra que escribas tiene que salir textualmente (podés redondear) de un resultado de tus \
herramientas. Si no tenés una herramienta que la produzca, decí que no está disponible. No estimes ni \
extrapoles números por tu cuenta.
2. Distinguí predicción (qué es probable que pase) de efecto causal (qué pasa si intervenimos). Para \
preguntas de intervención ("¿qué pasa si subimos...?") usá las herramientas causales.
3. Si la pregunta usa una variable que el estudio no tiene (por ejemplo, edad de los clientes), decilo y \
ofrecé la más cercana disponible.
4. Mencioná la limitación relevante cuando corresponda: población de búsquedas con al menos un clic, \
aleatorización imperfecta del benchmark y riesgo de confusión no observada en los datos del algoritmo."""

HERRAMIENTAS: list[dict[str, Any]] = [
    {
        "name": "efecto_causal_top5",
        "description": "Efecto causal del top-5 sobre la reserva: OR intra-búsqueda del benchmark con orden aleatorio, del observacional ajustado y del ingenuo, más la robustez (bootstrap, diferencia de riesgos, re-ponderación).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "cate_por_perfil",
        "description": "Efecto heterogéneo (CATE, Causal Forest DML, orden del algoritmo) del top-5 sobre la probabilidad de reserva para un perfil de hotel, con IC 95 %.",
        "input_schema": {
            "type": "object",
            "properties": {
                "estrellas": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "precio_vs_busqueda_pct": {
                    "type": "number",
                    "description": "Precio del hotel respecto de la mediana de su búsqueda, en % (ej. -20 = 20 % más barato).",
                },
                "usuario_recurrente": {"type": "boolean"},
            },
            "required": ["estrellas", "precio_vs_busqueda_pct", "usuario_recurrente"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "prediccion_reserva",
        "description": "Probabilidad predicha de reserva (Random Forest compacto) para un hotel en una lista, con la dispersión entre árboles. Es predictiva, no causal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "estrellas": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "precio_noche_usd": {"type": "number"},
                "precio_vs_busqueda_pct": {"type": "number"},
                "posicion": {"type": "integer"},
                "en_promocion": {"type": "boolean"},
                "usuario_recurrente": {"type": "boolean"},
            },
            "required": [
                "estrellas",
                "precio_noche_usd",
                "precio_vs_busqueda_pct",
                "posicion",
                "en_promocion",
                "usuario_recurrente",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "refutaciones",
        "description": "Resultado de las refutaciones de DoWhy (causa común aleatoria, placebo, subconjunto) y semáforo de confianza.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "escenarios_valuacion",
        "description": "Escenarios (conservador, central, optimista) de reservas y USD incrementales por cada 1.000 exposiciones en el top-5, con sus supuestos.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# ------------------------------------------------------------------------------ herramientas
def ejecutar_herramienta(nombre: str, entrada: dict, art: dict, modelo_rf=None) -> dict:
    """Ejecuta una herramienta de solo lectura sobre los artefactos agregados."""
    from logica import cate_perfil, perfil_completo, predecir, semaforo

    ca = art["causal"]
    if nombre == "efecto_causal_top5":
        return {
            "or_benchmark_orden_aleatorio": round(ca["or_benchmark"], 3),
            "or_observacional_ajustado": round(ca["or_observacional_ajustado"], 3),
            "or_ingenuo": round(ca["or_ingenuo"], 3),
            "cociente_or_observacional_sobre_benchmark": round(ca["robustez"]["cociente_or"], 3),
            "ic95_bootstrap_cociente_or": [round(x, 3) for x in ca["robustez"]["ic_boot_or"]],
            "cociente_efecto_marginal": round(ca["robustez"]["cociente_ame"], 3),
            "ic95_bootstrap_cociente_efecto_marginal": [
                round(x, 3) for x in ca["robustez"]["ic_boot_ame"]
            ],
            "lectura": "OR > 1: el top-5 aumenta las chances de reserva. El observacional ajustado sobreestima el efecto del benchmark.",
        }
    if nombre == "cate_por_perfil":
        log_rel = math.log(1 + entrada["precio_vs_busqueda_pct"] / 100)
        c = cate_perfil(art, int(entrada["estrellas"]), log_rel, int(entrada["usuario_recurrente"]))
        return {
            "cate_puntos_porcentuales": round(100 * c["cate"], 2),
            "ic95_puntos_porcentuales": [round(100 * c["ic_inf"], 2), round(100 * c["ic_sup"], 2)],
            "precio_vs_busqueda_pct_usado": round(100 * (math.exp(c["log_precio_rel"]) - 1), 1),
            "nota": "Grilla precalculada; se usa el punto de precio más cercano.",
        }
    if nombre == "prediccion_reserva":
        entradas = {
            "prop_starrating": entrada["estrellas"],
            "log_precio": math.log(max(entrada["precio_noche_usd"], 1.0)),
            "log_precio_rel": math.log(1 + entrada["precio_vs_busqueda_pct"] / 100),
            "position": entrada["posicion"],
            "promotion_flag": int(entrada["en_promocion"]),
            "usuario_recurrente": int(entrada["usuario_recurrente"]),
        }
        x, avisos = perfil_completo(entradas, art)
        r = predecir(modelo_rf, x)
        return {
            "probabilidad_reserva_pct": round(100 * r["p"], 2),
            "dispersion_entre_arboles_pct": [round(100 * r["p05"], 2), round(100 * r["p95"], 2)],
            "avisos": avisos,
            "nota": "Predicción, no efecto causal.",
        }
    if nombre == "refutaciones":
        return {
            "semaforo": semaforo(ca["refutaciones"]),
            "pruebas": [
                {"prueba": r["prueba"], "pasa": bool(r["pasa"]), "p": round(r["p"], 3)}
                for r in ca["refutaciones"]
            ],
            "nota": "Miden estabilidad, no ausencia de confusión no observada.",
        }
    if nombre == "escenarios_valuacion":
        return {
            e["escenario"]: {
                "reservas_incrementales_por_1000": round(e["reservas incrementales / 1.000"], 2),
                "usd_incrementales_por_1000": round(e["USD incrementales / 1.000"], 0),
                "supuestos": {
                    "fuente_dp": e["fuente Δp"],
                    "fuente_gasto": e["fuente gasto"],
                    "desplazamiento": e["desplazamiento de otra reserva"],
                },
            }
            for e in ca["escenarios"]
        }
    raise KeyError(f"Herramienta desconocida: {nombre}")


# ------------------------------------------------------------------------------ guardarraíl
_NUM = re.compile(r"(?<![\w.,])[-−]?\d{1,3}(?:\.\d{3})+(?:,\d+)?|(?<![\w.,])[-−]?\d+(?:[.,]\d+)?")


def _a_float(t: str) -> float:
    t = t.replace("−", "-")
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?", t):  # 1.234,5 (formato español)
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", ".")
    return float(t)


def numeros(texto: str) -> list[float]:
    return [_a_float(m.group()) for m in _NUM.finditer(texto)]


def _numeros_json(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numeros_json(v)]
    if isinstance(obj, list | tuple):
        return [x for v in obj for x in _numeros_json(v)]
    if isinstance(obj, str):
        return numeros(obj)
    return []


TRIVIALES = {
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    95.0,
    100.0,
    1000.0,
}  # top-5, estrellas, "IC 95 %", "por 1.000"


def cifras_no_verificadas(
    respuesta: str, resultados: list[dict], pregunta: str = ""
) -> list[float]:
    """Cifras de la respuesta que no aparecen (con tolerancia de redondeo) en ningún resultado de
    herramienta ni en la pregunta."""
    permitidas = np.array([x for r in resultados for x in _numeros_json(r)] + numeros(pregunta))
    faltan = []
    for v in numeros(respuesta):
        if abs(v) in TRIVIALES:
            continue
        if permitidas.size and np.any(
            np.isclose(np.abs(permitidas), abs(v), rtol=0.011, atol=0.0051)
        ):
            continue
        faltan.append(v)
    return faltan


# ------------------------------------------------------------------------------ loop
def responder(
    pregunta: str, cliente, art: dict, modelo_rf=None, historial: list | None = None
) -> dict:
    """Loop de tool use. Devuelve {"texto", "resultados", "bloqueado", "no_verificadas", "historial"}."""
    mensajes = list(historial or []) + [{"role": "user", "content": pregunta}]
    resultados: list[dict] = []
    for _ in range(MAX_PASOS):
        r = cliente.beta.messages.create(
            model=MODELO,
            max_tokens=16000,
            system=SISTEMA,
            tools=HERRAMIENTAS,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=mensajes,
        )
        if r.stop_reason == "refusal":
            return {
                "texto": "El modelo no puede responder esta consulta.",
                "resultados": resultados,
                "bloqueado": True,
                "no_verificadas": [],
                "historial": mensajes,
            }
        mensajes.append({"role": "assistant", "content": r.content})
        usos = [b for b in r.content if b.type == "tool_use"]
        if r.stop_reason != "tool_use" or not usos:
            texto = "".join(b.text for b in r.content if b.type == "text").strip()
            faltan = cifras_no_verificadas(texto, resultados, pregunta)
            if faltan:
                return {
                    "texto": "La respuesta incluía cifras que no salen del motor de la app y se bloqueó. Abajo están los resultados de las herramientas consultadas.",
                    "resultados": resultados,
                    "bloqueado": True,
                    "no_verificadas": faltan,
                    "historial": mensajes,
                }
            return {
                "texto": texto,
                "resultados": resultados,
                "bloqueado": False,
                "no_verificadas": [],
                "historial": mensajes,
            }
        devolver = []
        for u in usos:
            try:
                salida = ejecutar_herramienta(u.name, dict(u.input), art, modelo_rf)
                resultados.append(
                    {"herramienta": u.name, "entrada": dict(u.input), "salida": salida}
                )
                devolver.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": u.id,
                        "content": json.dumps(salida, ensure_ascii=False),
                    }
                )
            except Exception as e:  # noqa: BLE001 - el error vuelve al modelo como resultado de herramienta
                devolver.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": u.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    }
                )
        mensajes.append({"role": "user", "content": devolver})
    return {
        "texto": "La consulta necesitó demasiados pasos; reformulala más concreta.",
        "resultados": resultados,
        "bloqueado": True,
        "no_verificadas": [],
        "historial": mensajes,
    }
