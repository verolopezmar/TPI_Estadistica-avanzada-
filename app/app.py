"""Retorno causal de un algoritmo de recomendación — app de decisión (TPI EAyMR 2026, UCA).

Ejecutar desde la raíz del repo:  streamlit run app/app.py
La app solo lee artefactos agregados de models/ (sin datos individuales; ver tpi.artefactos).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logica import (  # noqa: E402
    cargar_artefactos,
    cargar_modelo,
    cate_perfil,
    explicar_local,
    perfil_completo,
    predecir,
    proyectar_pca,
    semaforo,
)

AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
st.set_page_config(page_title="Retorno causal del top-5", page_icon="🏨", layout="wide")
art = cargar_artefactos()
meta = art["meta"]


def fig_base(fig: go.Figure, alto: int = 360) -> go.Figure:
    fig.update_layout(
        height=alto,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        template="plotly_white",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


st.title("¿El top-5 vende, o solo muestra lo que se vendería igual?")
st.caption(
    f"Expedia ICDM 2013 · muestra de {meta['busquedas']:,} búsquedas ({meta['filas']:,} hoteles mostrados) · "
    f"prevalencia de reserva {100 * meta['prevalencia_reserva']:.2f} % · solo agregados: celdas con menos de "
    f"{meta['k_min_supresion']} filas suprimidas"
)
with st.sidebar:
    st.header("Cómo leer esta app")
    st.markdown(
        "- **Diagnóstico:** cómo son los datos y qué separa a lo reservado.\n"
        "- **What-Is:** qué *predice* el modelo para un hotel (asociación).\n"
        "- **What-If:** qué *causa* exponer un hotel en el top-5 (intervención).\n\n"
        "Una variable puede predecir muy bien y no ser una palanca: la posición predice la reserva "
        "en parte porque el algoritmo pone arriba a los hoteles que se venderían igual."
    )
    st.info("Datos confidenciales (reglas de Kaggle): la app no contiene registros individuales.")

tab1, tab2, tab3 = st.tabs(
    ["1 · Diagnóstico y exploración", "2 · Predictivo (What-Is)", "3 · Simulación causal (What-If)"]
)

# ------------------------------------------------------------------------------- Diagnóstico
with tab1:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    variables = art["variables"]
    var = c1.selectbox("Variable", list(variables), format_func=lambda v: variables[v]["etiqueta"])
    estrato = c2.radio(
        "Orden de la lista", ["Algoritmo (R = 0)", "Aleatorio (R = 1)", "Ambos"], horizontal=False
    )
    normalizar = c3.toggle("Mostrar como proporción", value=True)
    claves = {"Algoritmo (R = 0)": ["R0"], "Aleatorio (R = 1)": ["R1"], "Ambos": ["R0", "R1"]}[
        estrato
    ]
    fig = go.Figure()
    for b, color, nombre in [(0, AZUL, "Sin reserva"), (1, NARANJA, "Reservado")]:
        conteos = np.sum(
            [variables[var][k][f"reserva_{b}"]["conteos"] for k in claves], axis=0
        ).astype(float)
        bordes = np.array(variables[var][claves[0]][f"reserva_{b}"]["bordes"])
        y = conteos / conteos.sum() if normalizar and conteos.sum() else conteos
        fig.add_bar(
            x=(bordes[:-1] + bordes[1:]) / 2, y=y, name=nombre, marker_color=color, opacity=0.75
        )
    fig.update_layout(
        barmode="overlay",
        title=variables[var]["etiqueta"],
        xaxis_title=variables[var]["etiqueta"],
        yaxis_title="proporción" if normalizar else "filas",
    )
    st.plotly_chart(fig_base(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for k, color, nombre in [
        ("R0", AZUL, "Orden del algoritmo"),
        ("R1", NARANJA, "Orden aleatorio"),
    ]:
        d = art["ctr_posicion"][k]
        fig.add_scatter(
            x=d["posicion"],
            y=d["ctr"],
            mode="lines+markers",
            name=nombre,
            line={"color": color, "width": 2},
            marker={"size": 5},
        )
    fig.add_vrect(
        x0=0.5,
        x1=5.5,
        fillcolor="#e4e3df",
        opacity=0.5,
        line_width=0,
        annotation_text="top-5",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Tasa de clic por posición", xaxis_title="posición", yaxis_title="tasa de clic"
    )
    c1.plotly_chart(fig_base(fig), use_container_width=True)
    c1.caption(
        "Con orden aleatorio la caída es sesgo de posición puro; con el algoritmo se le suma la calidad de lo que pone arriba. Los pozos en 5, 11, 17 y 23 son slots ocupados por listados que no están en los datos."
    )

    p = art["pca"]
    g = p["grilla_test"]
    reserva = c2.toggle("Plano PCA: ver solo hoteles reservados", value=False)
    gr_ = g["reserva_1" if reserva else "reserva_0"]
    bx, by = np.array(gr_["bordes_x"]), np.array(gr_["bordes_y"])
    z = np.array(gr_["conteos"], dtype=float).T
    z[z == 0] = np.nan
    fig = go.Figure(
        go.Heatmap(
            x=(bx[:-1] + bx[1:]) / 2,
            y=(by[:-1] + by[1:]) / 2,
            z=np.log10(z),
            colorscale="Blues",
            colorbar={"title": "log10(n)"},
        )
    )
    with c2.expander("Proyectar un hotel en el plano PC1-PC2", expanded=False):
        vals = {}
        cols = st.columns(2)
        defaults = dict(zip(p["variables"], p["media"], strict=True))
        for i, v in enumerate(p["variables"]):
            vals[v] = cols[i % 2].number_input(
                v, value=float(round(defaults[v], 2)), step=0.1, key=f"pca_{v}"
            )
    pc1, pc2 = proyectar_pca(art, vals)
    fig.add_scatter(
        x=[pc1],
        y=[pc2],
        mode="markers",
        marker={"size": 14, "color": NARANJA, "symbol": "x"},
        name="hotel ingresado",
    )
    fig.update_layout(
        title=f"Test en el plano PCA de Train (PC1 {p['pct_var'][0]:.1f} %, PC2 {p['pct_var'][1]:.1f} %)",
        xaxis_title="PC1: categoría y precio",
        yaxis_title="PC2: ubicación y promoción",
    )
    c2.plotly_chart(fig_base(fig), use_container_width=True)
    cargas = pd.DataFrame(
        p["componentes"],
        index=[f"PC{i + 1}" for i in range(len(p["componentes"]))],
        columns=p["variables"],
    ).T.round(2)
    with c2.expander("Cargas de las componentes"):
        st.dataframe(cargas, use_container_width=True)

# ------------------------------------------------------------------------------ What-Is
with tab2:
    modelo = cargar_modelo()
    st.subheader("Probabilidad de reserva de un hotel en una lista")
    st.caption(
        "Random Forest compacto (hojas de ≥ 100 filas). Las variables no ingresadas toman la mediana de Train."
    )
    c = st.columns(4)
    e = {
        "prop_starrating": c[0].slider("Estrellas", 1, 5, 3),
        "prop_review_score": c[0].slider("Reseñas", 1.0, 5.0, 4.0, 0.5),
        "prop_location_score1": c[0].slider("Ubicación 1", 0.0, 7.0, 3.0, 0.1),
        "prop_location_score2": c[0].slider("Ubicación 2", 0.0, 1.0, 0.1, 0.01),
        "log_precio": np.log(c[1].number_input("Precio por noche (USD)", 10.0, 1500.0, 120.0, 5.0)),
        "log_precio_rel": np.log(
            c[1].slider("Precio / mediana de la búsqueda", 0.3, 3.0, 1.0, 0.05)
        ),
        "promotion_flag": int(c[1].toggle("En promoción")),
        "prop_brand_bool": int(c[1].toggle("Hotel de cadena", value=True)),
        "position": c[2].slider("Posición en la lista", 1, 38, 10),
        "random_bool": int(c[2].toggle("Lista en orden aleatorio")),
        "srch_length_of_stay": c[2].slider("Noches", 1, 14, 2),
        "srch_booking_window": c[2].slider("Anticipación (días)", 0, 365, 20),
        "srch_adults_count": c[3].slider("Adultos", 1, 6, 2),
        "srch_children_count": c[3].slider("Niños", 0, 4, 0),
        "srch_room_count": c[3].slider("Habitaciones", 1, 4, 1),
        "usuario_recurrente": int(c[3].toggle("Usuario con compras previas")),
    }
    x, avisos = perfil_completo(e, art)
    for a in avisos:
        st.warning(a)
    r = predecir(modelo, x)
    prev = meta["prevalencia_reserva"]
    k1, k2, k3 = st.columns(3)
    k1.metric("P(reserva)", f"{100 * r['p']:.2f} %", f"{r['p'] / prev:.1f} × la tasa base")
    k2.metric("Dispersión entre árboles (p5-p95)", f"{100 * r['p05']:.2f} - {100 * r['p95']:.2f} %")
    m = art["modelo"]["metricas_test_app"]
    k3.metric(
        "Desempeño en Test (PR-AUC)",
        f"{m['PR_AUC']:.3f}",
        f"ROC-AUC {m['ROC_AUC']:.3f}",
        delta_color="off",
    )
    c1, c2 = st.columns(2)
    contrib = explicar_local(modelo, x, art)
    fig = go.Figure(
        go.Bar(
            x=contrib.values * 100,
            y=contrib.index,
            orientation="h",
            marker_color=[NARANJA if v > 0 else AZUL for v in contrib.values],
        )
    )
    fig.update_layout(
        title="Por qué: cambio en P(reserva) si cada variable volviera a su mediana (p.p.)",
        xaxis_title="puntos porcentuales",
        yaxis={"autorange": "reversed"},
    )
    c1.plotly_chart(fig_base(fig), use_container_width=True)
    cal = art["modelo"]["calibracion_app"]
    fig = go.Figure()
    fig.add_scatter(
        x=cal["prob_media"],
        y=cal["frec_obs"],
        mode="lines+markers",
        name="modelo",
        line={"color": AZUL},
    )
    lim = max(cal["prob_media"] + cal["frec_obs"])
    fig.add_scatter(
        x=[0, lim],
        y=[0, lim],
        mode="lines",
        name="calibración perfecta",
        line={"color": GRIS, "dash": "dash"},
    )
    fig.update_layout(
        title="Calibración en Test (10 bins por cuantil)",
        xaxis_title="probabilidad predicha",
        yaxis_title="frecuencia observada",
    )
    c2.plotly_chart(fig_base(fig), use_container_width=True)
    st.warning(
        "Esta es una predicción, no un efecto causal: subir la posición en este simulador no dice cuánto vendería el hotel si lo subiéramos de verdad. Para eso, el módulo 3."
    )

# ------------------------------------------------------------------------------ What-If
with tab3:
    ca = art["causal"]
    st.subheader("do(T = top-5): ¿cuánto vende de más un hotel por estar en el top-5?")
    luz = semaforo(ca["refutaciones"])
    icono = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}[luz]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("OR causal (benchmark, orden aleatorio)", f"{ca['or_benchmark']:.2f}")
    k2.metric(
        "OR observacional ajustado (algoritmo)",
        f"{ca['or_observacional_ajustado']:.2f}",
        f"+{100 * (ca['or_observacional_ajustado'] / ca['or_benchmark'] - 1):.0f} % vs. benchmark",
        delta_color="inverse",
    )
    k3.metric("OR ingenuo (algoritmo, sin ajuste)", f"{ca['or_ingenuo']:.2f}")
    k4.metric("Refutaciones DoWhy", f"{icono} {luz}")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("**DAG causal** (orden del algoritmo; U e I son latentes)")
        st.graphviz_chart(ca["dag_dot"], use_container_width=True)
    with c2:
        st.markdown("**Semáforo de refutaciones** (regresión backdoor, R = 0)")
        for rr in ca["refutaciones"]:
            st.markdown(
                f"{'🟢' if rr['pasa'] else '🔴'} **{rr['prueba']}**: ATE original {rr['ATE original']:.4f} → {rr['ATE refutado']:.4f} (p = {rr['p']:.3g})"
            )
        st.markdown("**Efecto en escala absoluta (DoWhy)**")
        ate = pd.DataFrame(ca["ate_dowhy"]).set_index("estrato")[
            ["ingenuo (Y ~ T)", "regresión backdoor", "IPW", "tasa base P(Y|T=0)"]
        ]
        st.dataframe(ate.style.format("{:.4f}"), use_container_width=True)
        st.caption(
            "Las tasas absolutas no son comparables entre estratos: el dataset sobrerrepresenta búsquedas con compra cuando el orden lo decide el algoritmo. La comparación válida es el OR dentro de la búsqueda."
        )
    st.markdown(
        "### Heterogeneidad: ¿para qué hoteles rinde más el slot? (CATE, Causal Forest DML)"
    )
    c1, c2, c3 = st.columns(3)
    est_ = c1.select_slider("Estrellas", [1, 2, 3, 4, 5], 3)
    pr = c2.select_slider(
        "Precio vs. su búsqueda",
        options=[-0.6, -0.3, 0.0, 0.3, 0.6],
        value=0.0,
        format_func=lambda v: f"{100 * (np.exp(v) - 1):+.0f} %",
    )
    rec = int(c3.toggle("Usuario recurrente", key="cate_rec"))
    cp = cate_perfil(art, est_, pr, rec)
    st.metric(
        "CATE del top-5 para este perfil (Δ P(reserva), orden del algoritmo)",
        f"{100 * cp['cate']:.2f} p.p.",
        f"IC 95 % [{100 * cp['ic_inf']:.2f}; {100 * cp['ic_sup']:.2f}] p.p.",
        delta_color="off",
    )
    cols = st.columns(4)
    nombres = {
        "prop_starrating": "estrellas",
        "log_precio_rel": "log precio relativo",
        "prop_location_score2": "ubicación 2",
        "usuario_recurrente": "usuario recurrente",
    }
    for col, (v, cur) in zip(cols, ca["cate_curvas"].items(), strict=True):
        fig = go.Figure()
        fig.add_scatter(
            x=cur["valor"], y=cur["IC_sup"], mode="lines", line={"width": 0}, showlegend=False
        )
        fig.add_scatter(
            x=cur["valor"],
            y=cur["IC_inf"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(42,120,214,0.2)",
            line={"width": 0},
            showlegend=False,
        )
        fig.add_scatter(
            x=cur["valor"],
            y=cur["CATE"],
            mode="lines+markers",
            line={"color": AZUL},
            showlegend=False,
        )
        fig.update_layout(
            title=f"CATE según {nombres[v]}", xaxis_title=nombres[v], yaxis_title="Δ P(reserva)"
        )
        col.plotly_chart(fig_base(fig, 280), use_container_width=True)
    with st.expander("Sensibilidad a un confusor no observado y trampa del mediador"):
        st.dataframe(pd.DataFrame(ca["sensibilidad_u"]), use_container_width=True)
        st.dataframe(pd.DataFrame(ca["mediador"]), use_container_width=True)
