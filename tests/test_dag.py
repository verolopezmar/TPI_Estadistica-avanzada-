"""Propiedades del DAG borrador verificadas por d-separación (demostración analítica en código)."""

import networkx as nx
import pytest

from tpi.dag import CONFUSORES, a_dot, construir_dag, cumple_backdoor

Z = set(CONFUSORES)


@pytest.fixture
def g_alg():
    return construir_dag("algoritmo")


@pytest.fixture
def g_rand():
    return construir_dag("aleatorio")


def test_ambos_grafos_son_aciclicos(g_alg, g_rand):
    assert nx.is_directed_acyclic_graph(g_alg)
    assert nx.is_directed_acyclic_graph(g_rand)


def test_aleatorio_identifica_sin_ajuste(g_rand):
    """Con orden aleatorio, la diferencia de medias simple ya es causal."""
    assert cumple_backdoor(g_rand, "T", "Y", set())


def test_algoritmo_sin_ajuste_esta_confundido(g_alg):
    assert not cumple_backdoor(g_alg, "T", "Y", set())


def test_algoritmo_ajustar_z_observado_no_alcanza_si_existe_u(g_alg):
    """Supuesto a contrastar: con U (latente) en el grafo, Z observado NO cumple Backdoor."""
    assert not cumple_backdoor(g_alg, "T", "Y", Z)


def test_algoritmo_z_mas_u_cumpliria_backdoor(g_alg):
    """Si pudiéramos medir U, el ajuste sería válido: U es exactamente lo que falta."""
    assert cumple_backdoor(g_alg, "T", "Y", Z | {"U"})


def test_condicionar_en_el_mediador_viola_backdoor(g_rand):
    assert not cumple_backdoor(g_rand, "T", "Y", {"M"})


def test_el_clic_es_ademas_colisionador_con_la_intencion(g_rand):
    """T -> M <- I: condicionar en el clic asocia exposición e intención de compra."""
    assert nx.is_d_separator(g_rand, {"T"}, {"I"}, set())
    assert not nx.is_d_separator(g_rand, {"T"}, {"I"}, {"M"})


def test_condicionar_en_colisionador_induce_berkson(g_rand):
    """T -> C <- Y <- Zh: condicionar en C asocia la exposición con la calidad del hotel."""
    assert nx.is_d_separator(g_rand, {"T"}, {"Zh"}, set())
    assert not nx.is_d_separator(g_rand, {"T"}, {"Zh"}, {"C"})
    assert not cumple_backdoor(g_rand, "T", "Y", {"C"})


def test_dot_contiene_todos_los_arcos(g_alg):
    dot = a_dot(g_alg)
    assert all(f"{a} -> {b};" in dot for a, b in g_alg.edges)
