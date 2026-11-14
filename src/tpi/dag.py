"""DAG causal (BORRADOR, Fase 0): exposición algorítmica → conversión en Expedia.

Nodos conceptuales (agrupados para legibilidad; `EXPANSION` los lleva a columnas reales para
DoWhy en la Fase 4):

    R   random_bool: mecanismo de asignación del orden (aleatorio vs. algoritmo)
    T   exposición: hotel en el top-5 de la lista (position <= 5)
    M   click_bool: el usuario abre la ficha del hotel (mediador)
    Y   booking_bool / gross_bookings_usd: conversión y gasto
    Zh  calidad y precio del hotel (observados)
    Zp  popularidad previa del hotel (observada, proxy parcial de U)
    Zb  contexto de la búsqueda
    Zu  historial del usuario (usuario recurrente)
    Zc  competidores: comp_n_info (cuántos tienen dato) y comp_saldo (saldo de compN_rate; +1 =
        Expedia más barata que el competidor, definición de Kaggle). Confusor: el ranker la usa
        (interacción Zc × R sobre T) y afecta clic y reserva (Fase 1, sección 1.4)
    S   filtro de inclusión del dataset (≥ 1 clic por búsqueda; en R = 0, además, sobremuestreo de
        compras). Está condicionado en TODOS los datos: es el colisionador natural (Entrada 4)
    U   (latente) desempeño histórico del hotel en Expedia que usa el ranker y no vemos
    I   (latente) intención de compra del usuario en esa sesión
    C   (simulado en la Fase 4) "destacado en el reporte de hoteles más vendidos": T -> C <- Y

Los arcos se justifican uno por uno en la BITÁCORA (Entrada 4).
"""

from __future__ import annotations

import networkx as nx

TRATAMIENTO = "T"
RESULTADO = "Y"
MEDIADOR = "M"
COLISIONADOR = "C"
LATENTES = ("U", "I")
CONFUSORES = ("Zh", "Zp", "Zb", "Zu", "Zc")
SELECCION = "S"

ARCOS: list[tuple[str, str]] = [
    # Mecanismo de asignación del tratamiento
    ("R", "T"),
    ("Zh", "T"),
    ("Zp", "T"),
    ("Zb", "T"),
    ("Zu", "T"),
    ("Zc", "T"),  # Fase 1: interacción Zc × R sobre T significativa (comp_saldo p = 2,7e-18)
    ("U", "T"),
    # Efecto de la exposición: solo a través del clic (no hay reserva sin clic)
    ("T", "M"),
    ("M", "Y"),
    # Causas comunes del resultado
    ("Zh", "M"),
    ("Zh", "Y"),
    ("Zp", "M"),
    ("Zp", "Y"),
    ("Zb", "M"),
    ("Zb", "Y"),
    ("Zu", "M"),
    ("Zu", "Y"),
    ("Zc", "M"),  # comp_saldo afecta el clic con orden aleatorio (OR 1,147)
    ("Zc", "Y"),
    ("U", "M"),
    ("U", "Y"),
    # Intención latente del usuario: causa el clic y la reserva, no la posición
    ("I", "M"),
    ("I", "Y"),
    # Relaciones entre confusores
    ("Zh", "Zp"),
    ("U", "Zp"),
    # Colisionador (simulado)
    ("T", "C"),
    ("Y", "C"),
    # Selección del dataset (colisionador natural): S depende del clic y, en R = 0, de la reserva
    ("M", "S"),
    ("Y", "S"),
]

EXPANSION: dict[str, list[str]] = {
    "R": ["random_bool"],
    "T": ["top5 := position <= 5"],
    "M": ["click_bool"],
    "Y": ["booking_bool", "gross_bookings_usd"],
    "Zh": [
        "prop_starrating",
        "prop_review_score",
        "prop_brand_bool",
        "prop_location_score1",
        "prop_location_score2",
        "price_usd",
        "promotion_flag",
    ],
    "Zp": ["prop_log_historical_price", "srch_query_affinity_score"],
    "Zb": [
        "srch_length_of_stay",
        "srch_booking_window",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_saturday_night_bool",
        "orig_destination_distance",
        "site_id",
    ],
    "Zu": ["visitor_hist_starrating", "visitor_hist_adr_usd", "usuario_recurrente := notna(...)"],
    "Zc": ["comp_n_info", "comp_saldo"],
    "U": ["(no observado)"],
    "I": ["(no observado)"],
    "C": ["(simulado) destacado := f(T, Y, ruido)"],
    "S": ["(implícito) la búsqueda está en el dataset: >= 1 clic"],
}


def construir_dag(subconjunto: str = "algoritmo") -> nx.DiGraph:
    """DAG para un subconjunto de búsquedas.

    * `"algoritmo"` (random_bool = 0): el ranker decide T a partir de Z y de U.
    * `"aleatorio"` (random_bool = 1): la intervención del experimento corta todos los arcos que
      entran a T salvo R → T. Es la cirugía de grafo de do(T) que hizo Expedia por nosotras.
    """
    g = nx.DiGraph(ARCOS)
    if subconjunto == "aleatorio":
        g.remove_edges_from([(p, "T") for p in list(g.predecessors("T")) if p != "R"])
    elif subconjunto != "algoritmo":
        raise ValueError("subconjunto debe ser 'algoritmo' o 'aleatorio'")
    assert nx.is_directed_acyclic_graph(g)
    return g


def cumple_backdoor(g: nx.DiGraph, t: str, y: str, z: set[str]) -> bool:
    """Criterio Backdoor de Pearl para el conjunto de ajuste `z`.

    (i) Ningún nodo de `z` es descendiente de `t` (no condicionar en mediadores ni en
        consecuencias del tratamiento, lo que incluye colisionadores T -> C <- Y).
    (ii) `z` d-separa `t` de `y` en el grafo sin los arcos que salen de `t` (es decir, bloquea
         todos los caminos que entran a `t` por la puerta trasera).
    """
    if z & nx.descendants(g, t):
        return False
    g_bd = g.copy()
    g_bd.remove_edges_from(list(g.out_edges(t)))
    return nx.is_d_separator(g_bd, {t}, {y}, set(z))


POSICIONES = {
    "R": (-2.9, -0.9),
    "U": (0.5, 4.4),
    "Zh": (-0.8, 3.1),
    "Zp": (2.0, 3.1),
    "Zb": (3.7, 2.3),
    "Zu": (-2.9, 2.0),
    "Zc": (4.7, 1.7),
    "T": (-1.0, 0.0),
    "M": (1.0, 0.0),
    "Y": (3.0, 0.0),
    "I": (3.5, -1.5),
    "C": (0.6, -2.3),
    "S": (2.2, -2.3),
}
ETIQUETAS = {
    "R": "R\nrandom_bool",
    "T": "T\ntop-5",
    "M": "M\nclic",
    "Y": "Y\nreserva",
    "Zh": "Zh\ncalidad/precio",
    "Zp": "Zp\npopularidad",
    "Zb": "Zb\nbúsqueda",
    "Zu": "Zu\nusuario",
    "Zc": "Zc\ncompetidores",
    "U": "U (latente)\ndesempeño hist.",
    "I": "I (latente)\nintención",
    "C": "C (simulado)\ndestacado",
    "S": "S (implícito)\nselección",
}
COLORES = {"T": "#4C72B0", "Y": "#55A868", "M": "#DD8452", "C": "#C44E52", "R": "#8172B3"}


def dibujar_dag(
    g: nx.DiGraph,
    ax,
    titulo: str = "",
    node_size: int = 2600,
    font_size: float = 7,
    arrowsize: int = 14,
    titulo_size: float = 10,
) -> None:
    """Dibuja el DAG con posiciones fijas (sin depender del binario de Graphviz)."""
    colores = [COLORES.get(n, "#BBBBBB" if n in LATENTES else "#DADDE3") for n in g.nodes]
    nx.draw_networkx_nodes(g, POSICIONES, ax=ax, node_color=colores, node_size=node_size)
    nx.draw_networkx_labels(g, POSICIONES, labels=ETIQUETAS, ax=ax, font_size=font_size)
    latentes = [e for e in g.edges if e[0] in LATENTES]
    otros = [e for e in g.edges if e[0] not in LATENTES]
    kw = {"ax": ax, "node_size": node_size, "arrowsize": arrowsize, "width": 0.6 + node_size / 5000}
    nx.draw_networkx_edges(g, POSICIONES, edgelist=otros, **kw)
    nx.draw_networkx_edges(g, POSICIONES, edgelist=latentes, style="dashed", alpha=0.6, **kw)
    ax.set_title(titulo, fontsize=titulo_size)
    ax.axis("off")


def a_dot(g: nx.DiGraph) -> str:
    """Serializa el DAG a DOT (formato que acepta dowhy.CausalModel(graph=...))."""
    lineas = ["digraph {"]
    for n in g.nodes:
        attrs = ' [style="dashed"]' if n in LATENTES else ""
        lineas.append(f"  {n}{attrs};")
    lineas += [f"  {a} -> {b};" for a, b in g.edges]
    lineas.append("}")
    return "\n".join(lineas)
