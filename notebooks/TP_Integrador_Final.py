# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Retorno causal de un algoritmo de recomendación
#
# **TPI Decision Intelligence Capstone — Estadística Aplicada y Modelos de Regresión (EAyMR 2026, UCA)**
#
# | Integrante | Rol Scrum |
# |---|---|
# | Verónica López Marsico | Project Lead & Scrum Master |
# | Lucía Báez | Backend & ML / Causal Engineer |
# | Antonella Robiollio | Frontend & UI/UX + QA & Validation Engineer |
#
# Notebook maestro (anexo del informe Quarto). Semilla global `SEED = 42`; nivel de significancia
# fijado a priori α = 0,05. Uso de IA: `docs/ia_log.md`.
#
# **Confidencialidad.** Las reglas de la competencia de Kaggle declaran confidenciales los datos.
# Este notebook **nunca muestra registros individuales**: solo agregados, distribuciones y
# resultados de contrastes. Los datos se regeneran con `python data/download_data.py`.

# %%
import sys
import warnings
from pathlib import Path

# Raíz del repo por ruta relativa (funciona desde notebooks/, desde la raíz y en CI)
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(RAIZ / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

from tpi import graficos as gr
from tpi.aleatorizacion import resumir_rho, rho_intra_busqueda
from tpi.auditoria import (
    agregar_competidores,
    aplicar_centinelas,
    cerca_tukey_log,
    contraste_mcar,
    diagnostico_escala_precio,
    imputar_con_indicador,
    precio_relativo,
    resumen_distribucion,
)
from tpi.config import SEED, set_global_seed
from tpi.contrato import contrato_activo
from tpi.inferencia import (
    anova_un_factor,
    contraste_pareado,
    describir_grupos,
    efecto_minimo_detectable,
    mann_whitney,
    posthoc,
    potencia,
    welch_dos_grupos,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)  # interno de DoWhy
warnings.filterwarnings(
    "ignore", message="Some inputs do not have OOB scores"
)  # RF con B chico (3.4)
rng = set_global_seed(SEED)
gr.estilo()
pd.set_option("display.precision", 4)
CONTRATO = contrato_activo()
crudo = CONTRATO.cargar()
print(
    f"Contrato: {CONTRATO.nombre} | filas: {len(crudo):,} | búsquedas: {crudo['srch_id'].nunique():,}"
)

# %% [markdown]
# ---
# # Fase 1 — Inferencia estadística clásica, diagnósticos y ANOVA
#
# ## 1.1 Diseño muestral y plan experimental
#
# **Unidad de observación:** el par búsqueda × hotel $(s, i)$. **Unidad de muestreo:** la
# búsqueda $s$. Los hoteles de una lista comparten usuario, fecha y destino, así que sus filas
# no son independientes: muestrear filas sueltas rompería las listas y, en la Fase 3, filtraría
# información entre Train y Test. Muestreamos 15.000 búsquedas completas por estrato de
# `random_bool` (R), con semilla fija.
#
# **Unidades independientes para los contrastes.** Todo contraste se hace sobre unidades
# independientes: búsquedas (contrastes pareados dentro de la búsqueda) o reservas (a lo sumo
# una por búsqueda). Nunca sobre filas, que darían errores estándar artificialmente chicos.
#
# **Nivel de significancia y potencia.** Fijamos α = 0,05 antes de mirar los datos. Con miles de
# unidades, cualquier diferencia minúscula es "significativa": por eso cada contraste reporta el
# tamaño de efecto y el **efecto mínimo detectable** con potencia $1-\beta = 0{,}80$, que dice qué
# puede y qué no puede ver la muestra.

# %%
busq = crudo.groupby("srch_id").agg(
    R=("random_bool", "first"),
    n_hoteles=("prop_id", "size"),
    reservas=("booking_bool", "sum"),
    clics=("click_bool", "sum"),
)
tabla_diseno = busq.groupby("R").agg(
    busquedas=("n_hoteles", "size"),
    filas=("n_hoteles", "sum"),
    hoteles_por_lista=("n_hoteles", "mean"),
    pct_con_reserva=("reservas", lambda s: 100 * (s >= 1).mean()),
    pct_con_clic=("clics", lambda s: 100 * (s >= 1).mean()),
)
tabla_diseno

# %% [markdown]
# **Slots estructurales.** La cobertura media de las listas (hoteles observados / posición máxima)
# es ≈ 0,87. Antes de atribuirlo a una remoción de filas, miramos en qué fracción de las búsquedas
# aparece cada posición.

# %%
presencia = (
    crudo[crudo.position <= 30].groupby(["random_bool", "position"]).size().unstack(0)
    / crudo.groupby("random_bool")["srch_id"].nunique()
).rename(columns={0: "R = 0", 1: "R = 1"})
SLOTS_ESPECIALES = presencia.index[presencia.max(axis=1) < 0.2].tolist()
print("Posiciones presentes en < 20 % de las búsquedas:", SLOTS_ESPECIALES)
print(
    f"Acuerdo top5 (pantalla) vs. top4_dataset (sensibilidad): {100 * (crudo.top5 == crudo.top4_dataset).mean():.2f} % de las filas"
)
presencia.loc[[p for p in range(1, 13)]].T.round(3)

# %% [markdown]
# Las posiciones 5, 11, 17 y 23 (un slot cada seis) aparecen en apenas 6-8 % de las búsquedas con
# R = 0 y 1,4-1,7 % con R = 1, contra ≈ 99 % de sus vecinas. Son **slots fijos ocupados por listados
# que no están en el dataset** (por ejemplo, anuncios u ofertas especiales), y explican casi todo el
# 13 % de posiciones faltantes. Consecuencias:
#
# * El hueco es **estructural**, no una remoción de hoteles según sus atributos. Por lo tanto **no
#   explica** la asociación residual posición–atributo observada con R = 1 (`docs/ia_log.md`,
#   error 10), que sigue abierta.
# * $T = 1\{\text{position} \le 5\}$ se mantiene definido sobre la posición **en pantalla** (así la
#   define Kaggle), pero en la práctica el top-5 contiene casi siempre 4 hoteles del dataset
#   (posiciones 1-4): el quinto slot lo ocupa otro listado.
# * Las filas que sí ocupan esos slots son pocas y atípicas (tasa de clic muy baja, ver 1.5). Se
#   conservan, pero se señalan con un indicador (`slot_especial`).
# * **T de sensibilidad** (decisión de Verónica, 12/11): `top4_dataset` = los 4 primeros hoteles
#   del dataset en el ranking de la búsqueda. Coincide con `top5` en el 99,89 % de las filas (output
#   de `tpi.contrato`); la Fase 4 reporta ambas.

# %% [markdown]
# ## 1.2 Auditoría forense de datos
#
# Auditamos la génesis de cada anomalía y cuantificamos la distorsión antes y después del
# tratamiento. Cuatro familias: **centinelas**, **desplazamiento de escala del precio**,
# **valores extremos** y **faltantes** (sección 1.3).
#
# ### Centinelas
#
# Tres variables usan el 0 como código de "no hay dato": `prop_review_score` (hotel sin reseñas),
# `prop_starrating` (sin categoría) y `prop_log_historical_price` (sin ventas en el período
# previo). Tratarlas como valores reales sesga medias y desvíos. Las pasamos a NaN y conservamos un
# indicador binario, porque la ausencia de información es en sí misma una señal para el ranker y
# para el usuario.

# %%
df, tabla_centinelas = aplicar_centinelas(crudo)
df["slot_especial"] = df["position"].isin(SLOTS_ESPECIALES).astype("int8")  # ver 1.1
tabla_centinelas

# %% [markdown]
# ### Desplazamiento de escala del precio
#
# `price_usd` debería ser el precio por noche y por habitación. Lo verificamos con la única
# fuente externa disponible, el gasto real de las reservas:
#
# $$r^{\text{noche}} = \frac{\text{gross\_bookings\_usd}}{\text{price} \cdot \text{noches} \cdot \text{habitaciones}}, \qquad
#   r^{\text{estadía}} = \frac{\text{gross\_bookings\_usd}}{\text{price} \cdot \text{habitaciones}}.$$
#
# Si el precio es por noche, $r^{\text{noche}} \approx 1{,}1$ (impuestos y tasas). Si es por
# estadía, $r^{\text{estadía}} \approx 1{,}1$. Con una noche ambas coinciden, así que las usamos
# para calibrar la banda de tolerancia, y clasificamos solo las estadías de dos o más noches.

# %%
reservas_1n = df[
    (df.booking_bool == 1) & (df.gross_bookings_usd > 0) & (df.srch_length_of_stay == 1)
]
r1 = reservas_1n.gross_bookings_usd / (reservas_1n.price_usd * reservas_1n.srch_room_count)
print(
    "Calibración (1 noche), cuantiles 5/25/50/75/95 % de la razón:",
    r1.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist(),
)
escala = diagnostico_escala_precio(df)
tabla_escala = escala["escala"].value_counts().to_frame("reservas")
tabla_escala["pct"] = 100 * tabla_escala["reservas"] / tabla_escala["reservas"].sum()
por_sitio = escala.groupby("site_id").agg(
    n=("escala", "size"), pct_por_estadia=("escala", lambda s: 100 * (s == "por_estadia").mean())
)
print(
    "Sitios con ≥ 100 reservas: % por estadía entre",
    por_sitio.query("n >= 100").pct_por_estadia.min().round(2),
    "y",
    por_sitio.query("n >= 100").pct_por_estadia.max().round(2),
)
tabla_escala

# %% [markdown]
# ### Valores extremos de precio
#
# Dos reglas complementarias:
#
# * **Absoluta:** cercas extremas de Tukey sobre $\log(\text{precio})$, $[Q_1 - 3\,\mathrm{IQR},\; Q_3 + 3\,\mathrm{IQR}]$.
#   Detecta búsquedas enteras con precios imposibles, que la regla relativa no ve porque la
#   mediana de la búsqueda también es absurda.
# * **Relativa:** precio mayor que 10 veces (o menor que 1/10 de) la mediana de su búsqueda.
#
# **Variable de trabajo:** precio relativo a la mediana de la búsqueda (winsorizado en los
# percentiles 1 y 99) y su logaritmo. Es invariante a cualquier escala común de la lista (por
# noche vs. por estadía, moneda) y mide lo que el usuario ve: si el hotel es caro *para esa
# búsqueda*.

# %%
df = precio_relativo(df)
lo_abs, hi_abs = cerca_tukey_log(df["price_usd"])
print(
    f"Cercas absolutas: [{lo_abs:.2f}; {hi_abs:.2f}] USD | outliers: {df.precio_outlier.sum():,} filas ({100 * df.precio_outlier.mean():.3f} %)"
)
distorsion = pd.DataFrame(
    {
        "price_usd (crudo)": resumen_distribucion(df["price_usd"]),
        "price_usd sin outliers": resumen_distribucion(df.loc[df.precio_outlier == 0, "price_usd"]),
        "log precio relativo": resumen_distribucion(df["log_precio_rel"]),
    }
).T
distorsion

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
lp = np.log10(df["price_usd"].clip(lower=0.01))
ax[0].hist(lp, bins=120, color=gr.AZUL)
for c in (lo_abs, hi_abs):
    ax[0].axvline(np.log10(c), color=gr.TEXTO_2, ls="--", lw=1)
ax[0].set(
    yscale="log",
    xlabel="log10(price_usd)",
    ylabel="Filas (escala log)",
    title="Precio crudo y cercas de Tukey",
)
ax[0].annotate(
    "cercas extremas\n(Q ± 3·IQR en log)",
    xy=(np.log10(hi_abs), 1e4),
    xytext=(4.2, 3e4),
    fontsize=8,
    color=gr.TEXTO_2,
    arrowprops={"arrowstyle": "->", "color": gr.TEXTO_2},
)
razones = escala["razon_noche"].clip(0, 3)
ax[1].hist(razones, bins=90, color=gr.AZUL)
ax[1].axvspan(0.85, 1.6, color=gr.AQUA, alpha=0.15, lw=0)
ax[1].set(
    xlabel="gasto / (precio · noches · habitaciones)",
    ylabel="Reservas (≥ 2 noches)",
    title="Escala del precio: razón gasto/precio",
)
ax[1].annotate(
    "banda compatible\ncon 'por noche'",
    xy=(0.85, ax[1].get_ylim()[1] * 0.6),
    xytext=(0.15, ax[1].get_ylim()[1] * 0.75),
    fontsize=8,
    color=gr.TEXTO_2,
    arrowprops={"arrowstyle": "->", "color": gr.TEXTO_2},
)
sns.kdeplot(df["log_precio_rel"], ax=ax[2], color=gr.AZUL, fill=True, alpha=0.25, bw_adjust=1.5)
ax[2].set(
    xlabel="log(precio / mediana de la búsqueda)", ylabel="Densidad", title="Variable tratada"
)
fig.tight_layout()
gr.guardar(fig, "f1_precio_auditoria")
plt.show()

# %% [markdown]
# ## 1.3 Mecanismo de pérdida de datos (MCAR, MAR, MNAR)
#
# Sea $R_j = 1\{X_j \text{ falta}\}$. **MCAR**: $P(R_j \mid X_{\text{obs}}, X_{\text{mis}}) = P(R_j)$.
# **MAR**: $P(R_j \mid X_{\text{obs}}, X_{\text{mis}}) = P(R_j \mid X_{\text{obs}})$. **MNAR**: depende del propio valor
# faltante. MCAR implica que $R_j$ es independiente de **toda** variable observada, y eso es
# contrastable: comparamos $R_j$ contra variables observadas (χ² de independencia para binarias,
# Mann-Whitney para continuas, Bonferroni por familia). MAR vs. MNAR **no** es contrastable con
# los datos: se argumenta por el mecanismo de generación.
#
# Con cientos de miles de filas, casi cualquier dependencia da p ≈ 0. Por eso la lectura se apoya
# en el tamaño de efecto: $|\text{SMD}| \ge 0{,}1$ o $V \ge 0{,}1$ es dependencia apreciable.

# %%
faltantes = (
    df[[c for c in df.columns if df[c].isna().any() and not c.startswith("comp")]]
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .to_frame("pct_faltante")
)
faltantes

# %%
CONTRA = [
    "booking_bool",
    "click_bool",
    "random_bool",
    "prop_starrating",
    "log_precio_rel",
    "prop_location_score1",
    "srch_booking_window",
]
mcar = pd.concat(
    [
        contraste_mcar(df, c, [v for v in CONTRA if v != c])
        for c in [
            "prop_location_score2",
            "orig_destination_distance",
            "prop_review_score",
            "visitor_hist_starrating",
            "srch_query_affinity_score",
        ]
    ]
)
resumen_mcar = mcar.pivot(index="variable_faltante", columns="contra", values="efecto").round(3)
resumen_mcar["max_|efecto|"] = resumen_mcar.abs().max(axis=1)
resumen_mcar

# %% [markdown]
# **Lectura (efectos de la tabla; ver `contraste_mcar` para p-valores con Bonferroni):**
#
# | Variable | % faltante | Dependencia observada | Mecanismo | Tratamiento |
# |---|---|---|---|---|
# | `prop_location_score2` | 22,0 | SMD = −0,506 con `location_score1`; −0,257 con estrellas; V = 0,043 con reserva | **MAR**: el faltante se explica por variables observadas del hotel (hoteles peor ubicados y de menos estrellas) | Mediana por país + indicador `_faltante` |
# | `orig_destination_distance` | 32,8 | SMD = 0,304 con estrellas; 0,229 con `location_score1`; 0,196 con anticipación | **MAR**: falla de geolocalización asociada a perfiles observables | Mediana por par (país del usuario, país del hotel) + indicador |
# | `prop_review_score` (incl. centinela 0) | 5,3 | SMD = −0,238 con `location_score1`; 0,146 con anticipación | **MNAR estructural**: el hotel no tiene reseñas; el valor no existe | Mediana por estrellas + indicador `sin_resenas`, que conserva la información |
# | `visitor_hist_starrating` / `_adr_usd` | 95,5 | Todas las dependencias < 0,1 (máx. 0,091) | **MNAR estructural**: el usuario no tiene compras previas; no hay valor que imputar | **No se imputa**: se usa `usuario_recurrente` |
# | `srch_query_affinity_score` | 94,3 | SMD = 0,180 con estrellas | **MNAR estructural**: el hotel no apareció en búsquedas previas | **No se imputa**: indicador `con_afinidad` |
# | `gross_bookings_usd` | 97,8 | Existe solo si `booking_bool = 1` | Estructural por diseño | Solo se analiza entre reservas |
#
# **MCAR se rechaza para las cinco variables contrastadas** (en todas hay al menos una dependencia
# con p ajustado < 0,05), aunque para `visitor_hist_starrating` ninguna dependencia es apreciable.
# La imputación por mediana dentro de grupos es coherente con MAR condicional al grupo, y el
# indicador evita que el modelo trate el valor imputado como observado. Un faltante MNAR
# estructural **no se imputa**: imputar un historial que no existe inventaría información.

# %%
df["usuario_recurrente"] = df["visitor_hist_starrating"].notna().astype("int8")
df["con_afinidad"] = df["srch_query_affinity_score"].notna().astype("int8")
df = imputar_con_indicador(df, "prop_location_score2", por=["prop_country_id"])
df = imputar_con_indicador(df, "prop_review_score", por=["prop_starrating"])
df = imputar_con_indicador(df, "prop_starrating")
df = imputar_con_indicador(df, "prop_log_historical_price", por=["prop_country_id"])
df = imputar_con_indicador(
    df, "orig_destination_distance", por=["visitor_location_country_id", "prop_country_id"]
)
print(
    "NaN restantes en variables de ajuste:",
    int(
        df[
            [
                "prop_location_score2",
                "prop_review_score",
                "prop_starrating",
                "prop_log_historical_price",
                "orig_destination_distance",
            ]
        ]
        .isna()
        .sum()
        .sum()
    ),
)

# %% [markdown]
# ## 1.4 Variables de competidores ($Z_c$): ¿confusor o predictor?
#
# **Definiciones (fuente primaria: pestaña Data de la competencia, transcriptas y verificadas por
# Verónica el 12/11):**
#
# * `compN_rate`: "+1 if Expedia has a lower price than competitor N for the hotel; 0 if the same;
#   -1 if Expedia's price is higher than competitor N; NULL signifies there is no competitive data".
#   **+1 = Expedia más barata.**
# * `compN_inv`: "+1 if competitor N does not have availability in the hotel; 0 if both Expedia and
#   competitor N have availability; NULL signifies there is no competitive data". En los datos
#   aparece también **−1**, un valor no documentado (ver la tabla siguiente); no lo interpretamos.
# * `compN_rate_percent_diff`: diferencia porcentual **absoluta** (sin signo; la dirección sale de
#   `compN_rate`), con el precio de Expedia como denominador.
#
# **Criterio de clasificación (fijado por Verónica antes del análisis):** $Z_c$ es **confusor** si el
# ranking la usa ($Z_c \to T$) **y** además afecta la reserva ($Z_c \to Y$). Si solo afecta $Y$, es
# **predictor**: puede incluirse por precisión, pero no integra el conjunto de ajuste mínimo de
# Backdoor. Las dos flechas se contrastan por separado:
#
# * **$Z_c \to T$:** el orden aleatorio (R = 1) corta los arcos que entran a T. Si el ranker usa
#   $Z_c$, la asociación entre $Z_c$ y T con R = 0 debe **debilitarse** con R = 1. Se mide de tres
#   formas: (a) $\rho$ de Spearman intra-búsqueda posición–$Z_c$; (b) diferencia pareada por
#   búsqueda $D_s = \bar Z_{c,\text{top-5}} - \bar Z_{c,\text{resto}}$, en cada estrato; (c) logit de
#   T sobre $Z_c$ y $Z_h$ con interacción $Z_c \times R$ (errores por clúster de búsqueda). (c) es
#   el contraste formal: $H_0$: el efecto de $Z_c$ sobre T, condicional a $Z_h$, es igual con y sin
#   el ranker.
# * **$Z_c \to Y$:** con R = 1, logit de la reserva (y del clic, su mediador obligado) sobre $Z_c$,
#   controlando por $\log(\text{posición})$ y por calidad y precio, con errores por clúster.
#
# Resumen de los 8 competidores (entre 56 % y 98 % de faltantes): `comp_n_info` = competidores con
# dato de precio; `comp_saldo` = (n con rate = +1 − n con rate = −1) / n con dato, en $[-1, 1]$:
# **positivo = Expedia más barata que la mayoría de los competidores con dato**; `comp_diff_max` =
# máxima diferencia porcentual absoluta válida (≤ 100 %).

# %%
inv = pd.concat([df[f"comp{i}_inv"].value_counts() for i in range(1, 9)], axis=1).sum(axis=1)
print(
    "compN_inv (8 competidores, valores no nulos):",
    inv.astype(int).to_dict(),
    f"| −1 no documentado: {100 * inv.get(-1.0, 0) / inv.sum():.2f} %",
)
diff_min = min(df[f"comp{i}_rate_percent_diff"].min() for i in range(1, 9))
print(f"Mínimo de compN_rate_percent_diff en los 8 competidores: {diff_min}")

# %%
df = agregar_competidores(df)
df["log_pos"] = np.log(df["position"])
ZH = "prop_starrating + prop_review_score + prop_location_score2 + log_precio_rel + promotion_flag"
filas_zc = []
for v in ["comp_n_info", "comp_saldo", "comp_diff_max"]:
    fila = {"variable": v}
    # (a) rho intra-búsqueda posición–Zc, por estrato
    for r in (0, 1):
        fila[f"rho_R{r}"] = resumir_rho(rho_intra_busqueda(df[df.random_bool == r], v), v).rho_medio
    # (b) diferencia pareada top-5 − resto, por estrato
    for r in (0, 1):
        d = df[df.random_bool == r].dropna(subset=[v])
        g = d.groupby(["srch_id", "top5"])[v].mean().unstack().dropna()
        c = contraste_pareado(g[1] - g[0])
        fila[f"D_R{r}"], fila[f"p_D_R{r}"] = c.diferencia, c.p_valor
    # (c) interacción Zc × R en el logit de T, condicional a Zh
    d = df.dropna(subset=[v])
    f = smf.logit(f"top5 ~ {v}*random_bool + ({ZH})*random_bool", d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["srch_id"]}
    )
    fila["OR_T_R0"] = float(np.exp(f.params[v]))
    fila["razon_OR_T_R1/R0"] = float(np.exp(f.params[f"{v}:random_bool"]))
    fila["p_interaccion"] = float(f.pvalues[f"{v}:random_bool"])
    # Zc -> Y (y -> M) con R = 1
    d1 = df[df.random_bool == 1].dropna(subset=[v])
    for y in ["click_bool", "booking_bool"]:
        f = smf.logit(
            f"{y} ~ {v} + log_pos + prop_starrating + prop_review_score + log_precio_rel", d1
        ).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d1["srch_id"]})
        fila[f"OR_{y}"] = float(np.exp(f.params[v]))
        fila[f"p_{y}"] = float(f.pvalues[v])
    filas_zc.append(fila)
tabla_zc = pd.DataFrame(filas_zc).set_index("variable")
tabla_zc.T

# %% [markdown]
# **Clasificación** (criterio de Verónica; aplicación de Claude Code; **pendiente de revisión por
# Verónica**; incluye el contraste explícito de $Z_c \to T$ que ella pidió el 12/11):
#
# | Variable | $Z_c \to T$ (ranker) | $Z_c \to Y$ (R = 1) | Clase |
# |---|---|---|---|
# | `comp_saldo` | Con R = 0 el top-5 tiene más saldo (D = +0,049, p = 1,1·10⁻⁵⁹); con R = 1 desaparece (D = −0,006, p = 0,073). Interacción: razón de OR 0,842, p = 2,7·10⁻¹⁸ | Reserva OR = 1,137 (p = 0,041); clic OR = 1,147 (p = 2,1·10⁻⁸) | **Confusor** |
# | `comp_n_info` | D = +0,100 con R = 0 vs. +0,027 con R = 1 (cae 73 %). Condicional a $Z_h$, OR(T) = 0,948 con R = 0; interacción: razón 1,019, p = 2,8·10⁻⁵ | Reserva OR = 1,118 (p = 2,9·10⁻¹⁷); clic sin efecto (p = 0,20) | **Confusor (débil en T)** |
# | `comp_diff_max` | Interacción p = 0,054: no se rechaza H0 | Reserva p = 0,53; clic p = 0,79 | **Ni confusor ni predictor**: se excluye |
#
# **Lectura con el signo correcto.** `comp_saldo` > 0 significa que Expedia es más barata que la
# mayoría de los competidores con dato. Bajo el orden del algoritmo, los hoteles con precio
# competitivo suben al top-5 (OR(T) = 1,173 por unidad de saldo, condicional a calidad y precio
# relativo). Con la posición aleatoria, ese mismo precio competitivo aumenta el clic (+14,7 % en las
# chances) y la reserva (+13,7 %). Como el ranker usa una variable que también causa la reserva por
# su cuenta, omitirla sesgaría el efecto del top-5 hacia arriba.
#
# **`comp_n_info`** pasa el criterio, pero su arco a T es chico (≈ 2 % en las chances, condicional a
# $Z_h$), y la asociación residual con R = 1 (D = +0,027, p = 1,9·10⁻⁹) es del mismo tipo que la
# aleatorización imperfecta de `prop_starrating` (limitación declarada, Entrada 4). Incluirla en el
# ajuste es seguro: es pre-tratamiento, no es mediador ni colisionador.
#
# **DAG actualizado** (`src/tpi/dag.py`): se agregan $Z_c \to T$ y $Z_c \to M$, y $Z_c$ pasa al conjunto de
# confusores. `tests/test_dag.py::test_sin_zc_el_ajuste_no_alcanza_aunque_se_observara_u` demuestra que
# omitir $Z_c$ deja abierto $T \leftarrow Z_c \to Y$.

# %% [markdown]
# ## 1.5 Análisis exploratorio univariado y bivariado
#
# La figura central del problema: la tasa de clic por posición con el orden del algoritmo
# (R = 0) y con el orden aleatorio (R = 1). Bajo R = 1 la caída con la posición es **sesgo de
# posición** puro (el arco $T \to M$). Bajo R = 0 se le suma la calidad de los hoteles que el
# algoritmo pone arriba (confusión). La distancia entre ambas curvas es la primera evidencia
# visual del sesgo que la Fase 4 tiene que corregir.

# %%
ctr = (
    df[df.position <= 35]
    .groupby(["random_bool", "position"])["click_bool"]
    .agg(["mean", "count"])
    .reset_index()
)
ctr["ee"] = np.sqrt(ctr["mean"] * (1 - ctr["mean"]) / ctr["count"])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for r, color, nombre in [
    (0, gr.AZUL, "Orden del algoritmo (R = 0)"),
    (1, gr.NARANJA, "Orden aleatorio (R = 1)"),
]:
    d = ctr[ctr.random_bool == r]
    ax[0].plot(d.position, d["mean"], color=color, label=nombre)
    ax[0].fill_between(
        d.position, d["mean"] - 1.96 * d.ee, d["mean"] + 1.96 * d.ee, color=color, alpha=0.2, lw=0
    )
    ax[0].annotate(
        nombre,
        xy=(d.position.iloc[-1], d["mean"].iloc[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=gr.TEXTO_2,
        va="center",
    )
ax[0].axvspan(0.5, 5.5, color=gr.GRILLA, alpha=0.6, lw=0)
ax[0].text(3, 0.004, "T = 1\n(top-5)", ha="center", fontsize=8, color=gr.TEXTO_2)
ax[0].set(
    xlabel="Posición en la lista",
    ylabel="Tasa de clic (IC 95 %)",
    title="Sesgo de posición vs. selección del algoritmo",
    xlim=(0.5, 41),
)
ax[0].legend(loc="upper right")
grupos_est = df.dropna(subset=["prop_starrating"]).assign(
    estrellas=lambda d: d.prop_starrating.astype(int).astype(str)
)
sns.boxplot(
    data=grupos_est,
    x="estrellas",
    y="log_precio_rel",
    order=["1", "2", "3", "4", "5"],
    color=gr.AZUL,
    ax=ax[1],
    fliersize=0,
    width=0.5,
    boxprops={"alpha": 0.35},
)
ax[1].set(
    xlabel="Estrellas del hotel",
    ylabel="log(precio / mediana de la búsqueda)",
    title="Precio relativo según categoría",
    ylim=(-1.6, 1.6),
)
fig.tight_layout()
gr.guardar(fig, "f1_ctr_posicion")
plt.show()

# %%
NUM = [
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "log_precio_rel",
    "promotion_flag",
    "comp_n_info",
    "position",
    "click_bool",
    "booking_bool",
]
corr_s = df[NUM].corr(method="spearman")
corr_p = df[NUM].corr(method="pearson")
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
mascara = np.triu(np.ones_like(corr_s, dtype=bool), k=1)
for a, c, t in [(ax[0], corr_s, "Spearman (ρ)"), (ax[1], corr_p, "Pearson (r)")]:
    sns.heatmap(
        c,
        mask=mascara,
        cmap=gr.DIVERGENTE,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=1,
        linecolor=gr.SUPERFICIE,
        cbar=a is ax[1],
        ax=a,
    )
    a.set_title(t)
    a.grid(False)
fig.tight_layout()
gr.guardar(fig, "f1_correlaciones")
plt.show()
print(
    "Mayor |ρ_Spearman − r_Pearson|:",
    (corr_s - corr_p).abs().stack().sort_values(ascending=False).head(3).round(3).to_dict(),
)
pd.DataFrame(
    {"ρ con position": corr_s["position"], "ρ con booking_bool": corr_s["booking_bool"]}
).drop(["position", "booking_bool"]).round(3)

# %% [markdown]
# **Lectura de la exploración:**
#
# * **Sesgo de posición.** Con orden aleatorio, la tasa de clic cae con la posición (el arco
#   $T \to M$ existe). Con el orden del algoritmo, la curva arranca más alta en el top-5 y cae más
#   abajo al final de la lista: el algoritmo concentra arriba a los hoteles que se clickean igual.
#   La brecha entre curvas es la confusión que la Fase 4 debe eliminar. Los "pozos" en 5, 11, 17 y
#   23 son los slots estructurales (1.1).
# * **Calidad y precio.** El precio relativo sube con las estrellas: calidad y precio están
#   entrelazados, lo que anticipa multicolinealidad en la Fase 2.
# * **Correlaciones.** Spearman y Pearson difieren a lo sumo en 0,043 (entre los dos scores de
#   ubicación), así que las relaciones son aproximadamente monótonas y lineales en estas escalas. La
#   variable más asociada a la posición es `prop_location_score2` (ρ = −0,172: mejor ubicación, más
#   arriba), seguida por el clic (−0,160), las estrellas (−0,082) y la promoción (−0,082). Con la
#   reserva, la asociación más fuerte es el clic (ρ = 0,686, el mediador), seguida por la posición
#   (ρ = −0,14 en la matriz). Las demás son débiles (`location_score2`: 0,065).
# * **Colinealidad anticipada.** `prop_log_historical_price` correlaciona 0,62 con las estrellas, y
#   el precio relativo, 0,45 con estrellas y con reseñas: bloque calidad–precio a auditar con VIF en
#   la Fase 2.
# * Son correlaciones a nivel fila, sin separar R = 0 de R = 1: descriptivas, no causales.

# %% [markdown]
# ## 1.6 Contraste 1 — Sesgo de posición en el clic (pareado, R = 1)
#
# **Pregunta de negocio:** si el orden es aleatorio, ¿los hoteles del top-5 reciben más clics que
# el resto *de la misma lista*? Es el primer eslabón del DAG ($T \to M$): si no existiera, el slot
# no tendría valor causal.
#
# Para cada búsqueda $s$ con R = 1: $D_s = \overline{\text{CTR}}_{s,\text{top-5}} - \overline{\text{CTR}}_{s,\text{resto}}$.
#
# $$H_0: \mathbb{E}[D_s] = 0 \qquad H_1: \mathbb{E}[D_s] \neq 0$$
#
# **Estadístico:** t de Student para una muestra sobre las diferencias pareadas,
# $t = \bar D / (s_D/\sqrt{n})$ con $n-1$ gl. Es pareado porque las dos tasas comparten búsqueda
# (usuario, destino, fecha), lo que elimina todo confusor a nivel búsqueda. La validez descansa en
# el TCL ($n$ en miles); como resguardo no paramétrico, Wilcoxon de rangos con signo. Tamaño de
# efecto: $d_z = \bar D / s_D$.
#
# **Selección:** el dataset solo trae búsquedas con al menos un clic (BITÁCORA, Entrada 4). Eso
# infla ambas tasas, pero dentro de la búsqueda el cociente se conserva aproximadamente, así que
# también reportamos el cociente de tasas medias.

# %%
r1_df = df[df.random_bool == 1]
por_busq = (
    r1_df.assign(T=(r1_df.position <= 5)).groupby(["srch_id", "T"])["click_bool"].mean().unstack()
)
por_busq = por_busq.dropna()
D = por_busq[True] - por_busq[False]
c1 = contraste_pareado(D, nombre="Contraste 1 (CTR top-5 − resto, R = 1)")
print(c1.texto())
print(f"Wilcoxon: W = {c1.extra['wilcoxon_W']:.4g}, p = {c1.extra['wilcoxon_p']:.3g}")
print(
    f"CTR medio top-5 = {por_busq[True].mean():.4f} | resto = {por_busq[False].mean():.4f} | cociente = {por_busq[True].mean() / por_busq[False].mean():.3f}"
)
print(
    f"Efecto mínimo detectable (d_z, 1−β = 0,80): {efecto_minimo_detectable(len(D)):.4f} | potencia para el d_z observado: {potencia(c1.efecto, len(D)):.4f}"
)
print(f"Asimetría de D: {stats.skew(D):.3f} | curtosis en exceso: {stats.kurtosis(D):.3f}")

# %% [markdown]
# **Decisión.** Se rechaza $H_0$: t(14982) = 37,47, p = 7,74·10⁻²⁹⁴. Con orden aleatorio, los hoteles
# del top-5 tienen una tasa de clic 5,6 puntos porcentuales mayor que el resto de su propia lista
# (IC 95 % [5,31; 5,90] p.p.), y $d_z = 0{,}306$ (efecto pequeño a mediano en la escala de Cohen). El
# Wilcoxon de resguardo coincide (p = 7,7·10⁻¹³¹), así que la conclusión no depende de la
# normalidad: $D_s$ tiene asimetría −0,540 y curtosis en exceso 3,608, que el TCL absorbe con
# n = 14.983.
#
# **Escala relativa.** La tasa de clic media del top-5 es 0,1043 contra 0,0483 del resto: un
# cociente de **2,16**. Por la selección del dataset (solo búsquedas con clic) las tasas absolutas
# están infladas, pero el cociente dentro de la búsqueda es la escala que se conserva (Entrada 4).
#
# **Potencia.** Con n = 14.983 el efecto mínimo detectable es $d_z$ = 0,023: la muestra detectaría
# efectos 13 veces menores que el observado.
#
# **Negocio.** Sin mirar la calidad del hotel, estar en el top-5 duplica la probabilidad de clic.
# Ese es el valor *bruto* del slot sobre la atención; falta saber cuánto se traduce en reservas
# (Fase 4).

# %% [markdown]
# ## 1.7 Contraste 2 — Gasto por reserva: usuarios recurrentes vs. nuevos
#
# **Pregunta de negocio:** ¿los usuarios con historial de compras en Expedia (recurrentes) gastan
# más por reserva que los nuevos? Es la "autoselección de usuarios recurrentes" del enunciado: si
# gastan distinto, el historial ($Z_u$) es candidato a confusor de cualquier efecto sobre el gasto.
#
# **Unidad:** la reserva (a lo sumo una por búsqueda, así que son independientes). Excluimos
# reservas con gasto nulo y con precio outlier. **Variable:** $\log(\text{gasto})$, porque el
# gasto tiene cola pesada a derecha; la diferencia de medias en log se lee como **cociente de
# medias geométricas** $e^{\bar y_1 - \bar y_2}$.
#
# $$H_0: \mu_{\log,\text{recurrentes}} = \mu_{\log,\text{nuevos}} \qquad H_1: \mu_{\log,\text{recurrentes}} \neq \mu_{\log,\text{nuevos}}$$
#
# **Estadístico:** t de Welch, porque los grupos tienen tamaños muy distintos y no hay razón para
# suponer varianzas iguales (Levene abajo). Gl de Welch-Satterthwaite. Resguardo: U de Mann-Whitney
# ($H_0$: $P(X_1 > X_2) = 1/2$). Tamaños de efecto: g de Hedges y r biserial de rangos.
# Contraste **descriptivo**, no causal: los recurrentes difieren en destino y duración.

# %%
res = df[(df.booking_bool == 1) & (df.gross_bookings_usd > 0) & (df.precio_outlier == 0)].copy()
res["log_gasto"] = np.log(res["gross_bookings_usd"])
rec, nue = (
    res.loc[res.usuario_recurrente == 1, "log_gasto"],
    res.loc[res.usuario_recurrente == 0, "log_gasto"],
)
lev = stats.levene(rec, nue, center="median")
print(f"Levene (Brown-Forsythe): F(1; {len(res) - 2}) = {lev.statistic:.4g}, p = {lev.pvalue:.3g}")
c2 = welch_dos_grupos(rec, nue, nombre="Contraste 2 (log gasto, recurrentes − nuevos)")
c2b = mann_whitney(np.exp(rec), np.exp(nue), nombre="Contraste 2b (gasto, Mann-Whitney)")
print(c2.texto())
print(c2b.texto())
print(
    f"Cociente de medias geométricas = {np.exp(c2.diferencia):.4f}, IC 95 % [{np.exp(c2.ic95[0]):.4f}; {np.exp(c2.ic95[1]):.4f}]"
)
print(f"Hodges-Lehmann (diferencia de gasto en USD): {c2b.diferencia:.2f}")
print(
    f"n recurrentes = {len(rec)}, n nuevos = {len(nue)} | efecto mínimo detectable (g): {efecto_minimo_detectable(len(rec), len(nue)):.4f}"
)
print(f"Asimetría log gasto: recurrentes {stats.skew(rec):.3f}, nuevos {stats.skew(nue):.3f}")
res.groupby("usuario_recurrente")["gross_bookings_usd"].describe()

# %% [markdown]
# **Decisión.** No se rechaza $H_0$: t de Welch(1158,6) = 1,303, p = 0,193; IC 95 % para la diferencia
# de medias en log [−0,019; 0,096]. Equivale a un cociente de medias geométricas de 1,039 (IC 95 %
# [0,981; 1,101]): los recurrentes gastarían entre 1,9 % menos y 10,1 % más por reserva. El
# Mann-Whitney de resguardo coincide (p = 0,093; r biserial = 0,031).
#
# **Supuestos.** Levene no rechaza la igualdad de varianzas (F(1; 16062) = 3,19, p = 0,074), pero
# Welch se mantiene porque los grupos están muy desbalanceados (1.025 vs. 15.039), y en ese caso un
# error de tipo II de Levene sería costoso. La asimetría del log gasto (0,076 y 0,507) está dentro
# de $[-1, 1]$.
#
# **Potencia.** No rechazar no prueba igualdad. Con estos tamaños, el efecto mínimo detectable con
# potencia 0,80 es g = 0,090; el observado (g = 0,043) es la mitad. **Diferencias de gasto menores
# que ≈ 0,09 desvíos no son detectables con esta muestra**: el contraste es compatible tanto con
# "no hay diferencia" como con una diferencia chica.
#
# **Negocio.** No hay evidencia de que el historial del usuario cambie el ticket medio. La
# "autoselección de recurrentes" del enunciado, si existe, operaría sobre la conversión y no sobre
# el gasto por reserva.

# %% [markdown]
# ## 1.8 ANOVA de un factor — Precio por noche del hotel reservado según categoría
#
# **Pregunta:** ¿el precio esperado por noche del hotel reservado varía entre categorías de
# estrellas? Es la base del supuesto de que la calidad ($Z_h$) y el precio están entrelazados en
# $Z$ (relevante para la multicolinealidad de la Fase 2).
#
# **Modelo:** $Y_{gj} = \mu + \alpha_g + \varepsilon_{gj}$, con $\sum \alpha_g = 0$, $Y$ = log del
# precio por noche de la reserva $j$ en la categoría $g \in \{1\text{–}2, 3, 4, 5\}$ (1 y 2
# estrellas se unen por el bajo $n$ de 1 estrella).
#
# $$H_0: \mu_{1\text{–}2} = \mu_3 = \mu_4 = \mu_5 \qquad H_1: \exists\, g \ne g' : \mu_g \ne \mu_{g'}$$
#
# **Supuestos y resguardos:**
# * *Normalidad:* con miles de observaciones, Shapiro-Wilk rechaza ante desvíos microscópicos
#   irrelevantes para el TCL. La evaluamos con asimetría (criterio $|\text{asimetría}| \le 1$),
#   curtosis y gráficos Q-Q.
# * *Homocedasticidad:* Levene centrado en la mediana (Brown-Forsythe). Si se rechaza, el F
#   clásico deja de ser válido y usamos **ANOVA de Welch** (no asume varianzas iguales) y
#   **Kruskal-Wallis** (no paramétrico).
# * *Post-hoc:* Games-Howell (no asume varianzas iguales), comparado con Tukey HSD (que sí las
#   asume) y con Welch por pares + Bonferroni (conservador).

# %%
anova_df = res.dropna(subset=["prop_starrating"]).copy()
anova_df = anova_df[anova_df.precio_outlier == 0]
anova_df["categoria"] = pd.cut(
    anova_df.prop_starrating, [0, 2, 3, 4, 5], labels=["1-2", "3", "4", "5"]
)
anova_df["log_precio_noche"] = np.log(anova_df.price_usd)
desc = describir_grupos(anova_df, "log_precio_noche", "categoria")
sw = {
    g: stats.shapiro(d.sample(min(len(d), 5000), random_state=SEED)).pvalue
    for g, d in anova_df.groupby("categoria", observed=True)["log_precio_noche"]
}
desc["shapiro_p (n≤5000)"] = pd.Series(sw)
desc

# %%
an = anova_un_factor(anova_df, "log_precio_noche", "categoria")
for k in ["levene", "clasico", "welch", "kruskal"]:
    print(an[k].texto())
print(f"ω² (ANOVA clásico) = {an['clasico'].extra['omega2']:.4f}")
ph = posthoc(anova_df, "log_precio_noche", "categoria")
ph["cociente_precios"] = np.exp(ph["diferencia"])
ph

# %%
fig, ax = plt.subplots(1, 5, figsize=(14, 3.2))
for a, (g, d) in zip(
    ax[:4], anova_df.groupby("categoria", observed=True)["log_precio_noche"], strict=True
):
    stats.probplot(d, dist="norm", plot=a)
    a.get_lines()[0].set(markerfacecolor=gr.AZUL, markeredgecolor=gr.AZUL, markersize=2, alpha=0.4)
    a.get_lines()[1].set(color=gr.NARANJA, lw=1.5)
    a.set(
        title=f"Q-Q · {g} estrellas",
        xlabel="Cuantil teórico N(0,1)",
        ylabel="Cuantil muestral" if g == "1-2" else "",
    )
medias = anova_df.groupby("categoria", observed=True)["log_precio_noche"].agg(["mean", "sem"])
ax[4].errorbar(
    range(len(medias)),
    np.exp(medias["mean"]),
    yerr=[
        np.exp(medias["mean"]) - np.exp(medias["mean"] - 1.96 * medias["sem"]),
        np.exp(medias["mean"] + 1.96 * medias["sem"]) - np.exp(medias["mean"]),
    ],
    fmt="o",
    color=gr.AZUL,
    capsize=4,
    ms=8,
)
ax[4].set(
    xticks=range(len(medias)),
    xticklabels=medias.index,
    xlabel="Estrellas",
    ylabel="USD por noche (media geom., IC 95 %)",
    title="Precio por categoría",
)
fig.tight_layout()
gr.guardar(fig, "f1_anova")
plt.show()

# %% [markdown]
# **Supuestos.**
# * *Normalidad:* Shapiro-Wilk rechaza en los cuatro grupos (p entre 2,4·10⁻²² y 3,4·10⁻⁵ con n ≤ 5.000),
#   como anticipa el enunciado para muestras grandes. Lo relevante: asimetrías entre −0,213 y 0,314
#   (dentro de $[-1, 1]$) y curtosis en exceso entre 0,49 y 1,61. Los Q-Q muestran colas algo más
#   pesadas que la normal, sobre todo la izquierda en 1-2 y 3 estrellas. Con n ≥ 1.252 por grupo, el
#   TCL garantiza la validez aproximada de las medias.
# * *Homocedasticidad:* **Levene rechaza** (F(3; 16060) = 56,57, p = 2,27·10⁻³⁶; razón de varianzas
#   máx/mín = 1,59). El F clásico pierde validez, así que la inferencia principal es el **ANOVA de
#   Welch**.
#
# **Decisión.** Se rechaza $H_0$ con las tres pruebas: Welch F(3; 4432,0) = 1807; F clásico(3; 16060)
# = 1820; Kruskal-Wallis H(3) = 4150; en las tres, p < 10⁻³⁰⁰ (subdesborde de doble precisión). El
# tamaño de efecto es grande: η² = 0,254, ω² = 0,254 y ε² = 0,258. La categoría explica ≈ 25 % de
# la varianza del log precio.
#
# **Post-hoc.** Las seis comparaciones son significativas con los tres criterios (Games-Howell,
# Tukey HSD y Bonferroni), así que la conclusión no depende del supuesto de homocedasticidad. En
# cocientes de medias geométricas, un hotel de 5 estrellas cuesta 2,65 veces uno de 1-2
# (1/0,3773), 1,92 veces uno de 3 y 1,44 veces uno de 4. Los p-valores de Games-Howell que aparecen
# como ≈ 10⁻¹³ o 0 son el límite numérico de la distribución de rango estudentizado en pingouin, no
# valores exactos.
#
# **Por qué Welch y no Kruskal-Wallis como principal:** la pregunta es sobre el **valor esperado**
# del log precio (medias), y Welch responde exactamente eso sin suponer varianzas iguales.
# Kruskal-Wallis contrasta otra hipótesis (dominancia estocástica) y queda como resguardo.

# %% [markdown]
# ## 1.9 Síntesis de la Fase 1
#
# | Hallazgo | Evidencia | Consecuencia para las fases siguientes |
# |---|---|---|
# | Hay sesgo de posición puro sobre el clic | Contraste 1: CTR ×2,16 en el top-5 con orden aleatorio | El arco $T \to M$ existe; el efecto sobre la reserva se estima en la Fase 4 |
# | El algoritmo selecciona hoteles que se clickean igual | Curvas de CTR R = 0 vs. R = 1 | Confusión a corregir (Backdoor) |
# | La categoría explica ≈ 25 % del log precio | ANOVA de Welch, η² = 0,254 | Multicolinealidad calidad–precio (VIF, Fase 2) |
# | El gasto por reserva no difiere entre recurrentes y nuevos (no detectable bajo g ≈ 0,09) | Contraste 2 | $Z_u$ entra como confusor de la conversión, no del gasto |
# | $Z_c$ = {`comp_n_info`, `comp_saldo`} son confusores | Tabla 1.4 | Entran al ajuste en la Fase 4 |
# | Slots estructurales 5/11/17/23 fuera del dataset | Presencia por posición | T = top-5 en pantalla ≈ 4 hoteles del dataset |
# | Faltantes MAR y MNAR estructurales, ninguno MCAR | Tabla 1.3 | Imputación por grupo + indicadores; los MNAR no se imputan |

# %% [markdown]
# ---
# # Fase 2 — Diagnóstico de Gauss-Markov, puntos influyentes y reducción PCA
#
# ## 2.0 Partición Train/Test (única para las Fases 2 a 4)
#
# 80/20 **agrupada por búsqueda** (ninguna lista queda partida entre Train y Test) y
# **estratificada por `random_bool`**, con `SEED = 42` (`tpi.particion`). Toda la Fase 2 se estima
# **solo con Train**; Test queda intacto para el torneo de la Fase 3. La partición es aleatoria y no
# cronológica porque el período es corto (8 meses) y el objetivo es generalizar a búsquedas nuevas
# del mismo período (ver Durbin-Watson en 2.2).

# %%
import statsmodels.api as sm
from sklearn.linear_model import HuberRegressor, RANSACRegressor

from tpi.particion import es_train
from tpi.regresion import ajustar_pca, diagnostico_gauss_markov, influencia, resumen_pca, tabla_vif

df["train"] = es_train(df)
print(
    df.groupby("train").agg(
        busquedas=("srch_id", "nunique"), filas=("srch_id", "size"), pct_R1=("random_bool", "mean")
    )
)

# %% [markdown]
# ## 2.1 Modelo lineal de referencia (MCO): precio hedónico del hotel reservado
#
# **Unidad:** la reserva (una por búsqueda, así que las observaciones son independientes), en Train
# y sin precios outlier. **Respuesta:** $Y = \log(\text{precio por noche})$. Las pendientes se leen
# como semielasticidades: $100\,(e^{\beta_j}-1)$ % de cambio en el precio por unidad de $X_j$.
#
# $$\log(\text{precio})_i = \beta_0 + \sum_{j=1}^{p} \beta_j X_{ij} + \varepsilon_i, \qquad
#   \mathbb{E}[\varepsilon \mid X] = 0,\; \operatorname{Var}(\varepsilon \mid X) = \sigma^2 I$$
#
# Bajo Gauss-Markov (linealidad, exogeneidad, homocedasticidad, no autocorrelación, rango completo)
# el estimador MCO $\hat\beta = (X^\top X)^{-1}X^\top y$ es el de mínima varianza entre los lineales
# insesgados. La normalidad de $\varepsilon$ solo hace falta para que las pruebas t y F sean exactas
# en muestras chicas; con $n$ en miles valen asintóticamente.
#
# **Contrastes:** $F$ global, $H_0: \beta_1 = \dots = \beta_p = 0$; $t$ individual,
# $H_0: \beta_j = 0$.
#
# **Experimento previo (fallido, Entrada 3).** Primero ajustamos la especificación ingenua: precio
# crudo en USD, sin log y sin quitar outliers.

# %%
X_MCO = [
    "prop_starrating", "prop_review_score", "sin_resenas", "prop_location_score1",
    "prop_location_score2", "prop_log_historical_price", "sin_historial_precio", "prop_brand_bool",
    "promotion_flag", "srch_length_of_stay", "srch_booking_window", "srch_adults_count",
    "srch_children_count", "srch_room_count", "srch_saturday_night_bool", "usuario_recurrente",
    "random_bool",
]  # fmt: skip
reservas_tr = df[df.train & (df.booking_bool == 1)].sort_values("date_time").copy()
ingenuo = smf.ols("price_usd ~ " + " + ".join(X_MCO), reservas_tr).fit()
dg_ing = diagnostico_gauss_markov(ingenuo)
inf_ing = influencia(ingenuo)
print(
    f"INGENUO (precio crudo, con outliers): n = {int(ingenuo.nobs)}, R² = {ingenuo.rsquared:.4f}, "
    f"JB = {dg_ing.jb:.4g}, curtosis = {dg_ing.curtosis:.1f}, Cook máx = {inf_ing.cook.max():.3f}, "
    f"obs. con D > 0,5: {int(inf_ing.influyente_05.sum())}"
)

rtr = reservas_tr[reservas_tr.precio_outlier == 0].copy()
rtr["log_precio_noche"] = np.log(rtr["price_usd"])
FORMULA = "log_precio_noche ~ " + " + ".join(X_MCO)
mco = smf.ols(FORMULA, rtr).fit()
print(mco.summary().tables[0])
from tpi.inferencia import _p as formato_p

print(
    f"F global: F({int(mco.df_model)}; {int(mco.df_resid)}) = {mco.fvalue:.2f}, {formato_p(mco.f_pvalue)}"
)
ic_mco = mco.conf_int()
pd.DataFrame(
    {
        "coef": mco.params,
        "ee": mco.bse,
        "t": mco.tvalues,
        "p": mco.pvalues,
        "IC95_inf": ic_mco[0],
        "IC95_sup": ic_mco[1],
        "efecto_%": 100 * (np.exp(mco.params) - 1),
    }
).round(4)

# %% [markdown]
# **Experimento fallido (Entrada 3).** El modelo ingenuo sobre el precio crudo explica el 0,28 % de la
# varianza (R² = 0,0028, n = 12.877). Sus residuos tienen curtosis 12.192,8 (JB = 7,975·10¹⁰) y una
# sola observación tiene distancia de Cook 1,215 (> 0,5): un precio imposible domina la estimación.
# Las dos correcciones de la Fase 1 (log y outliers) lo resuelven.
#
# **Modelo corregido** (n = 12.858 reservas de Train): R² = 0,604, R² ajustado = 0,603. El F global
# rechaza $H_0$ de que todas las pendientes son nulas: F(17; 12840) = 1151 (p-valor exacto en la
# salida). Lectura de los coeficientes significativos, como cambio porcentual del precio por noche:
#
# * **Precio histórico del hotel** ($\beta$ = 0,766, t = 91,46): es casi una elasticidad. Un 1 % más
#   de precio histórico se asocia con un 0,77 % más de precio actual; es el regresor dominante.
# * **Estrellas** (+6,2 % por estrella, t = 13,36) y **reseñas** (+5,6 % por punto, t = 9,06),
#   netos del precio histórico.
# * **Promoción:** −14,9 % (t = −24,91), el descuento visible.
# * **Usuario recurrente:** +4,7 % (t = 3,94). Los recurrentes reservan hoteles algo más caros por
#   noche (compatible con el contraste 2 de la Fase 1, que no encontró diferencias en el gasto
#   total).
# * **No se rechaza $H_0$** para `random_bool` (p = 0,661): el precio del hotel reservado no
#   difiere entre orden aleatorio y orden del algoritmo, condicional a sus atributos. Tampoco para
#   marca (p = 0,705), ausencia de reseñas (p = 0,066) ni `location_score1` (p = 0,082).
#
# Es un modelo **descriptivo** del precio, no causal: sirve de referencia para auditar los supuestos
# de Gauss-Markov sobre datos reales.

# %% [markdown]
# ## 2.2 Auditoría de los supuestos de Gauss-Markov
#
# * **Homocedasticidad:** Breusch-Pagan ($H_0$: $\operatorname{Var}(\varepsilon\mid X) = \sigma^2$;
#   LM $= nR^2$ de la regresión auxiliar de $\hat\varepsilon^2$ sobre $X$, $\chi^2_p$) y White
#   (agrega cuadrados y productos cruzados).
# * **Normalidad:** Jarque-Bera, $JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right) \sim \chi^2_2$, y Q-Q.
# * **Independencia:** las reservas están ordenadas por fecha, así que el Durbin-Watson
#   $d = \sum (\hat\varepsilon_t - \hat\varepsilon_{t-1})^2 / \sum \hat\varepsilon_t^2 \approx 2(1-\hat\rho)$
#   mide autocorrelación temporal de primer orden.

# %%
dg = diagnostico_gauss_markov(mco)
print(f"Breusch-Pagan: LM = {dg.bp_lm:.2f}, gl = {dg.bp_gl}, {formato_p(dg.bp_p)}")
print(f"White: LM = {dg.white_lm:.2f}, {formato_p(dg.white_p)}")
print(
    f"Jarque-Bera: JB = {dg.jb:.2f}, {formato_p(dg.jb_p)} | asimetría = {dg.asimetria:.3f}, "
    f"curtosis = {dg.curtosis:.3f}"
)
print(f"Durbin-Watson (orden temporal): d = {dg.durbin_watson:.4f}")
mco_hc3 = smf.ols(FORMULA, rtr).fit(cov_type="HC3")
comp_ee = pd.DataFrame(
    {"coef": mco.params, "ee_MCO": mco.bse, "ee_HC3": mco_hc3.bse, "p_HC3": mco_hc3.pvalues}
)
comp_ee["razon_ee"] = comp_ee["ee_HC3"] / comp_ee["ee_MCO"]
comp_ee.round(4)

# %%
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
hb = ax[0].hexbin(
    mco.fittedvalues, mco.resid, gridsize=60, cmap=gr.SECUENCIAL, mincnt=1, bins="log"
)
ax[0].axhline(0, color=gr.NARANJA, lw=1.5)
ax[0].set(xlabel="Valor ajustado ŷ (log USD)", ylabel="Residuo", title="Residuos vs. ajustados")
fig.colorbar(hb, ax=ax[0], label="observaciones (log)")
sm.qqplot(
    mco.resid,
    line="s",
    ax=ax[1],
    markersize=2,
    alpha=0.4,
    markerfacecolor=gr.AZUL,
    markeredgecolor=gr.AZUL,
)
ax[1].get_lines()[1].set(color=gr.NARANJA, lw=1.5)
ax[1].set(title="Q-Q de residuos", xlabel="Cuantil teórico N(0,1)", ylabel="Cuantil muestral")
cuantiles = pd.qcut(mco.fittedvalues, 20)
disp = mco.resid.groupby(cuantiles, observed=True).std()
ax[2].plot(
    mco.fittedvalues.groupby(cuantiles, observed=True).mean(), disp, "o-", color=gr.AZUL, ms=5
)
ax[2].set(
    xlabel="ŷ (media del ventil)",
    ylabel="Desvío de los residuos",
    title="Dispersión por ventil de ŷ",
)
fig.tight_layout()
gr.guardar(fig, "f2_residuos")
plt.show()

# %% [markdown]
# **Homocedasticidad: se rechaza.** Breusch-Pagan LM = 516,99 (gl = 17, p = 4,96·10⁻⁹⁹) y White
# LM = 989,87 (p = 1,71·10⁻¹¹⁸). El panel derecho lo muestra: el desvío de los residuos va de 0,26 a
# 0,37 según el nivel de precio ajustado. (El aviso de "matriz de rango incompleto" del test de White
# viene de elevar al cuadrado regresores binarios, que son iguales a sí mismos: statsmodels lo
# resuelve con pseudoinversa y no afecta el estadístico.)
# **Remediación:** errores estándar robustos **HC3**. Los EE suben hasta un 31 %
# (`sin_historial_precio`) y un 21 % (`sin_resenas`, cuyo p pasa de 0,066 a 0,129). Las
# conclusiones sustantivas no cambian; los coeficientes MCO siguen siendo insesgados, solo era
# incorrecta su varianza.
#
# **Normalidad: se rechaza** (JB = 4209,26; p < 10⁻³⁰⁰), pero con asimetría 0,082 y **curtosis
# 5,798**: colas pesadas y simétricas, visibles en los extremos del Q-Q. Con n = 12.858, los
# contrastes t y F siguen siendo válidos asintóticamente (TCL); las colas pesadas motivan los
# estimadores robustos de 2.3.
#
# **Independencia:** Durbin-Watson = 1,9515 con las reservas en orden temporal, compatible con
# ausencia de autocorrelación de primer orden ($\hat\rho \approx 1 - d/2 = 0{,}024$). Respalda la
# partición aleatoria (no cronológica) de 2.0.

# %% [markdown]
# ## 2.3 Observaciones extremas y puntos influyentes
#
# * Residuo estudentizado externo: $r_i = \hat\varepsilon_i / (\hat\sigma_{(i)}\sqrt{1-h_{ii}})$; outlier de respuesta si $|r_i| > 3$.
# * Leverage $h_{ii}$ (diagonal de $H = X(X^\top X)^{-1}X^\top$): extremo en $X$ si $h_{ii} > 2(p+1)/n$.
# * Cook: $D_i = \frac{\hat\varepsilon_i^2}{(p+1)\,\mathrm{MSE}}\left[\frac{h_{ii}}{(1-h_{ii})^2}\right]$; influyente si $D_i > 4/n$ o $D_i > 0{,}5$.
#
# **¿Error de medición o caso real?** Cruzamos los influyentes con el diagnóstico de escala del precio
# de la Fase 1 (reservas cuyo gasto es compatible con precio por noche, por estadía o con ninguno).

# %%
inf = influencia(mco)
n_obs, k = int(mco.nobs), mco.model.exog.shape[1]
print(f"n = {n_obs}, p + 1 = {k} | umbrales: h > {2 * k / n_obs:.5f}, D > {4 / n_obs:.6f}")
print(
    inf[["outlier_y", "alto_leverage", "influyente_4n", "influyente_05"]].sum().to_dict(),
    f"| D máx = {inf.cook.max():.4f}",
)
esc_tr = diagnostico_escala_precio(rtr)[["escala"]]
cruce = (
    rtr[[]]
    .join(esc_tr)
    .fillna({"escala": "1 noche (sin ambigüedad)"})
    .join(inf[["influyente_4n", "outlier_y"]])
)
cruce.groupby("escala")[["influyente_4n", "outlier_y"]].agg(["mean", "size"]).round(4)

# %%
y_r, X_r = rtr["log_precio_noche"], rtr[X_MCO].astype(float)
huber = HuberRegressor(epsilon=1.35, max_iter=1000).fit(X_r, y_r)
# RANSAC con el umbral por defecto (MAD de y, no de los residuos): experimento fallido (Entrada 3)
ransac = RANSACRegressor(random_state=SEED).fit(X_r, y_r)
# RANSAC corregido: umbral = 3 · σ robusto de los residuos MCO (1,4826 · MAD)
sigma_rob = 1.4826 * np.median(np.abs(mco.resid - np.median(mco.resid)))
ransac3 = RANSACRegressor(residual_threshold=3 * sigma_rob, random_state=SEED).fit(X_r, y_r)
sin_inf = smf.ols(FORMULA, rtr[~inf["influyente_4n"].to_numpy()]).fit()
robustez = pd.DataFrame(
    {
        "MCO": mco.params[X_MCO],
        "MCO sin D>4/n": sin_inf.params[X_MCO],
        "Huber": pd.Series(huber.coef_, index=X_MCO),
        "RANSAC (defecto)": pd.Series(ransac.estimator_.coef_, index=X_MCO),
        "RANSAC (3σ rob.)": pd.Series(ransac3.estimator_.coef_, index=X_MCO),
    }
)
print(
    f"RANSAC defecto: {ransac.inlier_mask_.mean():.2%} de inliers | RANSAC 3σ: "
    f"{ransac3.inlier_mask_.mean():.2%} | Huber: {np.mean(huber.outliers_):.2%} marcados como outliers"
)
print(
    "Máx. |coef − MCO sin influyentes|:",
    (robustez.drop(columns="MCO sin D>4/n").sub(robustez["MCO sin D>4/n"], axis=0))
    .abs()
    .max()
    .round(4)
    .to_dict(),
)
print(f"R² MCO sin influyentes: {sin_inf.rsquared:.4f} (n = {int(sin_inf.nobs)})")
robustez.round(4)

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.2))
tam = 4 + 400 * inf["cook"] / inf["cook"].max()
ax.scatter(inf["leverage"], inf["r_ext"], s=tam, color=gr.AZUL, alpha=0.35, lw=0)
ax.axhline(3, color=gr.TEXTO_2, ls="--", lw=1)
ax.axhline(-3, color=gr.TEXTO_2, ls="--", lw=1)
ax.axvline(2 * k / n_obs, color=gr.NARANJA, ls="--", lw=1)
ax.annotate("|r| = 3", xy=(inf.leverage.max() * 0.6, 3.3), fontsize=8, color=gr.TEXTO_2)
ax.annotate(
    "h = 2(p+1)/n",
    xy=(2 * k / n_obs, inf.r_ext.max() * 0.97),
    xytext=(6, 0),
    textcoords="offset points",
    fontsize=8,
    color=gr.TEXTO_2,
)
ax.set(
    xscale="log",
    xlabel="Leverage h_ii (escala log)",
    ylabel="Residuo estudentizado externo",
    title="Influencia (tamaño ∝ distancia de Cook)",
)
fig.tight_layout()
gr.guardar(fig, "f2_influencia")
plt.show()

# %% [markdown]
# **Diagnóstico** (n = 12.858, p + 1 = 18): 119 outliers de respuesta (|r| > 3), 843 puntos de alto
# leverage (h > 0,0028) y 746 influyentes según D > 4/n = 0,000311. **Ningún punto supera
# D = 0,5** (D máx = 0,0117). No hay un punto que, por sí solo, mueva la recta; hay un grupo de
# puntos moderadamente influyentes.
#
# **¿Error de medición o caso real?** Cruzado con el diagnóstico de escala del precio (Fase 1), la
# tasa de influyentes es del 4,85 % en reservas de 1 noche y del 6,37 % en las compatibles con
# precio por noche, pero sube al **7,91 %** en las de precio por estadía y al **9,51 %** en las
# incompatibles con ambas escalas. Lo mismo ocurre con los outliers de respuesta: 0,82-0,83 %
# contra 1,86 % y 2,62 %. **Parte de la influencia es error de medición** (precio informado en la
# escala equivocada). El resto son casos reales de colas pesadas.
#
# **Regresión robusta:**
# * Quitar los 746 influyentes sube el R² a 0,6857 y cambia los coeficientes a lo sumo en 0,023
#   (estrellas: 0,060 → 0,045).
# * **Huber** ($\epsilon$ = 1,35) llega casi a los mismos coeficientes que MCO sin influyentes
#   (diferencia máxima 0,012) sin descartar datos: es la especificación de resguardo.
# * **RANSAC con el umbral por defecto: experimento fallido (Entrada 3).** El umbral por defecto es
#   la MAD de *y*, no de los residuos, y con 17 regresores deja solo el 55,02 % de inliers. Los
#   coeficientes se vuelven inestables: estrellas −0,046, habitaciones +0,312; diferencia máxima con
#   MCO sin influyentes, 0,341.
# * Con el umbral corregido (3 σ̂ robustos de los residuos MCO) retiene el 89,03 % y la diferencia
#   máxima baja a 0,058.
#
# **Decisión:** MCO con errores HC3 como especificación principal, y Huber como resguardo. Los
# influyentes no se eliminan: la mayoría son casos reales, y la parte que es error de medición ya
# está neutralizada en las fases siguientes por el precio relativo a la búsqueda.

# %% [markdown]
# ## 2.4 Multicolinealidad (VIF)
#
# $\mathrm{VIF}_j = 1/(1-R_j^2)$, donde $R_j^2$ es el de la regresión de $X_j$ sobre los demás
# regresores. Mide cuánto se infla $\operatorname{Var}(\hat\beta_j)$ respecto de regresores
# ortogonales. Umbrales: 5 (atención) y 10 (severa). Lo calculamos para el modelo de 2.1 y para el
# conjunto ampliado que usará el torneo de la Fase 3 (agrega precio, precio relativo, competidores y
# posición).

# %%
X_AMPLIADO = X_MCO + [
    "log_precio",
    "log_precio_rel",
    "comp_n_info",
    "position",
    "prop_location_score2_faltante",
]
vif = pd.concat(
    {
        "VIF modelo 2.1": tabla_vif(rtr[X_MCO]),
        "VIF conjunto ampliado (Fase 3)": tabla_vif(df.loc[df.train, X_AMPLIADO]),
    },
    axis=1,
)
vif.sort_values("VIF conjunto ampliado (Fase 3)", ascending=False).round(2).head(10)

# %% [markdown]
# **No hay multicolinealidad relevante.** En el modelo de 2.1 el VIF máximo es 1,74 (precio
# histórico) y 1,70 (estrellas). En el conjunto ampliado de la Fase 3 el máximo es 2,98
# (`log_precio`), seguido por 2,63 (precio histórico) y 1,97 (estrellas). Ninguna variable se acerca a
# 5. **Corrige una anticipación de la Fase 1:** las correlaciones de a pares de hasta 0,62 entre
# estrellas y precio histórico hacían esperar multicolinealidad, pero un $R^2_j$ auxiliar de a lo sumo
# 1 − 1/2,98 = 0,66 no infla las varianzas en forma preocupante (se registra en la Entrada 3).
# **Decisión:** no se eliminan ni combinan regresores por colinealidad. La regularización Ridge/Lasso
# de la Fase 3 se justifica por selección y estabilidad predictiva, no por un VIF alto.

# %% [markdown]
# ## 2.5 Reducción de dimensionalidad con PCA (sin fuga de datos)
#
# **Regla estricta:** la estandarización (media y desvío) y la descomposición espectral de la matriz
# de correlaciones se calculan **solo con las filas de Train**; Test se proyecta con esa misma
# rotación (`pipe.transform`), sin reajustar. `tests/test_fase2.py::test_pca_se_ajusta_solo_con_train`
# lo verifica.
#
# Variables: atributos continuos del hotel en la fila búsqueda × hotel (la unidad del torneo de la
# Fase 3). Criterios de retención: codo del scree plot y Kaiser ($\lambda_k > 1$ sobre variables
# estandarizadas).

# %%
X_PCA = [
    "prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2",
    "prop_log_historical_price", "log_precio", "log_precio_rel", "promotion_flag",
    "prop_brand_bool", "comp_n_info",
]  # fmt: skip
pca_pipe = ajustar_pca(df[df.train], X_PCA, seed=SEED)
var_pca, cargas = resumen_pca(pca_pipe, X_PCA)
n_kaiser = int((var_pca["autovalor"] > 1).sum())
print(
    f"Kaiser (λ > 1): {n_kaiser} componentes, {var_pca['pct_acum'].iloc[n_kaiser - 1]:.1f} % de la varianza"
)
Z_test = pca_pipe.transform(df.loc[~df.train, X_PCA].astype(float))
print(f"Test proyectado con la rotación de Train: {Z_test.shape}")
var_pca.round(3)

# %%
cargas.iloc[:, :4].round(3)

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
pcs = np.arange(1, len(var_pca) + 1)
ax[0].plot(pcs, var_pca["autovalor"], "o-", color=gr.AZUL, ms=6)
ax[0].axhline(1, color=gr.NARANJA, ls="--", lw=1)
ax[0].annotate("Kaiser: λ = 1", xy=(len(pcs) - 2.5, 1.06), fontsize=8, color=gr.TEXTO_2)
ax[0].set(xlabel="Componente", ylabel="Autovalor λ_k", title="Scree plot (Train)", xticks=pcs)
ax[1].bar(pcs, var_pca["pct_var"], color=gr.AZUL, alpha=0.35, width=0.6)
ax[1].plot(pcs, var_pca["pct_acum"], "o-", color=gr.AZUL, ms=5)
ax[1].set(
    xlabel="Componente",
    ylabel="% de varianza (barras) y acumulado (línea)",
    title="Varianza explicada",
    xticks=pcs,
    ylim=(0, 105),
)
sns.heatmap(
    cargas.iloc[:, :4],
    cmap=gr.DIVERGENTE,
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8},
    linewidths=1,
    linecolor=gr.SUPERFICIE,
    ax=ax[2],
)
ax[2].set_title("Cargas (autovectores) PC1-PC4")
ax[2].grid(False)
fig.tight_layout()
gr.guardar(fig, "f2_pca")
plt.show()

# %%
proy = pd.DataFrame(Z_test[:, :2], columns=["PC1", "PC2"]).assign(
    reserva=df.loc[~df.train, "booking_bool"].to_numpy()
)
muestra_proy = pd.concat(
    [proy[proy.reserva == 0].sample(4000, random_state=SEED), proy[proy.reserva == 1]]
)
fig, ax = plt.subplots(figsize=(7, 4.6))
for r, color, nombre in [
    (0, gr.AZUL, "Sin reserva (muestra de 4.000)"),
    (1, gr.NARANJA, "Reservado"),
]:
    d = muestra_proy[muestra_proy.reserva == r]
    ax.scatter(d.PC1, d.PC2, s=8, color=color, alpha=0.35 if r == 0 else 0.6, lw=0, label=nombre)
ax.set(
    xlabel=f"PC1 ({var_pca.pct_var.iloc[0]:.1f} %)",
    ylabel=f"PC2 ({var_pca.pct_var.iloc[1]:.1f} %)",
    title="Test proyectado en el plano PC1-PC2 de Train",
)
ax.legend(loc="upper right", markerscale=2)
fig.tight_layout()
gr.guardar(fig, "f2_pca_plano")
plt.show()
proy.groupby("reserva")[["PC1", "PC2"]].mean()

# %% [markdown]
# **Retención.** Kaiser retiene **4 componentes** (λ = 3,264; 1,515; 1,184; 1,002), que explican el
# 69,7 % de la varianza. El cuarto está en el límite (λ = 1,002). El codo del scree plot está en la
# 2.ª componente (de 3,26 a 1,52, y después la caída se suaviza). **Decisión:** 2 componentes para la
# visualización (plano PC1-PC2 de la app, módulo de diagnóstico) y 4 para el conjunto alternativo de
# features que se pone a prueba en la Fase 3.
#
# **Interpretación de negocio de las cargas:**
# * **PC1 (32,6 %): "categoría y precio".** Cargas positivas y parecidas en estrellas (0,44), precio
#   histórico (0,47), precio (0,47), precio relativo (0,39) y reseñas (0,35). Ordena los hoteles de
#   económicos a premium.
# * **PC2 (15,2 %): "buena ubicación a buen precio, independiente".** Ubicación (0,55 y 0,48) y
#   promoción (0,38), con signo opuesto al de cadena (−0,41) y al precio relativo (−0,30).
# * PC3 (11,8 %): cobertura de competidores (0,77) y pertenencia a cadena (0,58).
#   PC4 (10,0 %): promoción (0,75) frente a `location_score2` (−0,54).
#
# **Proyección ciega de Test.** Los hoteles reservados tienen PC2 medio = 0,270, contra −0,0035 de
# los no reservados. En PC1 la diferencia es mínima (0,0070 vs. −0,0063). **Lo que se reserva no es
# lo más caro ni lo de mayor categoría, sino lo bien ubicado, en promoción y barato para su
# búsqueda.** Es un hallazgo predictivo, no causal (Fase 4).

# %% [markdown]
# ## 2.6 Síntesis de la Fase 2
#
# | Supuesto / análisis | Resultado | Remediación |
# |---|---|---|
# | Especificación | Precio crudo: R² = 0,0028, Cook máx 1,215 (fallido) | log(precio) sin outliers: R² = 0,604 |
# | Homocedasticidad | Se rechaza (BP p = 4,96·10⁻⁹⁹; White p = 1,71·10⁻¹¹⁸) | Errores HC3 (EE hasta +31 %) |
# | Normalidad | Se rechaza (JB), curtosis 5,80, asimetría 0,08 | TCL (n = 12.858) + estimador robusto de Huber |
# | Independencia | DW = 1,95 | Ninguna; respalda la partición aleatoria |
# | Influyentes | 746 con D > 4/n, ninguno con D > 0,5; más frecuentes con escala de precio errónea | Huber como resguardo; RANSAC con umbral 3σ̂ (el de defecto falla) |
# | Multicolinealidad | VIF máx 1,74 (modelo) y 2,98 (conjunto Fase 3) | Ninguna |
# | PCA (solo Train) | 4 comp. por Kaiser (69,7 %), codo en 2; PC1 = categoría-precio, PC2 = ubicación/promoción | 2 PCs para la app, 4 como features alternativas en la Fase 3 |

# %% [markdown]
# ---
# # Fase 3 — Torneo predictivo, calibración y explicabilidad
#
# **Pregunta predictiva:** $P(\text{reserva} \mid X)$ para cada hotel de una lista. Es la pregunta
# observacional del enunciado ($P(Y \mid X)$), distinta de la causal de la Fase 4
# ($P(Y \mid do(T))$).
#
# **Diseño (propuesto por Claude Code):**
# * Unidad: fila búsqueda × hotel. $Y$ = `booking_bool`, con prevalencia ≈ 2 %: clasificación muy
#   desbalanceada.
# * Partición 80/20 de la Fase 2 (agrupada por búsqueda y estratificada por `random_bool`). La
#   validación cruzada es `StratifiedGroupKFold(5)` por `srch_id` dentro de Train: estratificada como
#   pide el enunciado y agrupada para que ninguna lista se parta entre pliegues.
# * Métrica de selección: **PR-AUC** (precisión promedio). Con 2 % de positivos, la exactitud y el
#   ROC-AUC son optimistas; la PR-AUC mide directamente cuánto se concentran las reservas arriba.
# * **Sin fuga:** se excluyen `click_bool` (mediador posterior a la exposición, que ocurre en la
#   misma sesión que la reserva), el gasto y la propia respuesta (`tests/test_modelos.py`).
#   `position` sí entra: se conoce al mostrar la lista.
# * **Cómputo:** las búsquedas de hiperparámetros se hacen sobre una submuestra del 25 % de las
#   búsquedas de Train (búsquedas completas, misma semilla). Los modelos finales se reentrenan con
#   todo Train y se evalúan una sola vez en Test.

# %%
import time

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from tpi.modelos import (
    FEATURES,
    OBJETIVO,
    acierto_top_k,
    cv_agrupado,
    matriz,
    metricas,
    submuestra_busquedas,
    umbral_f1,
)
from tpi.preparacion import preparar

# Verificación: los pasos expositivos de la Fase 1 coinciden con el pipeline reutilizable
pd.testing.assert_frame_equal(matriz(df), matriz(preparar(crudo)), check_dtype=False)
tr, te = df[df.train], df[~df.train]
X_tr, y_tr, g_tr = matriz(tr), tr[OBJETIVO].to_numpy(), tr["srch_id"].to_numpy()
X_te, y_te = matriz(te), te[OBJETIVO].to_numpy()
sub = submuestra_busquedas(tr, 0.25)
X_sub, y_sub, g_sub = matriz(sub), sub[OBJETIVO].to_numpy(), sub["srch_id"].to_numpy()
CV = cv_agrupado()
print(
    f"Train: {len(tr):,} filas ({tr.srch_id.nunique():,} búsquedas), prevalencia {y_tr.mean():.4f}"
)
print(
    f"Test:  {len(te):,} filas ({te.srch_id.nunique():,} búsquedas), prevalencia {y_te.mean():.4f}"
)
print(
    f"Submuestra de búsqueda de hiperparámetros: {len(sub):,} filas ({sub.srch_id.nunique():,} búsquedas)"
)
TIEMPOS = {}

# %% [markdown]
# ## 3.1 Familia 1 — Regresión logística regularizada
#
# $\operatorname{logit} P(Y=1\mid X) = \beta_0 + X^\top\beta$, estimada maximizando la verosimilitud
# penalizada $\ell(\beta) - \lambda\,\Omega(\beta)$ con $\Omega = \lVert\beta\rVert_1$ (Lasso) o
# $\tfrac12\lVert\beta\rVert_2^2$ (Ridge), sobre variables estandarizadas ($C = 1/\lambda$). La
# búsqueda en grilla elige penalización y $C$ por PR-AUC con CV agrupada.
#
# Para interpretar, los **odds ratios** $\mathrm{OR}_j = e^{\beta_j}$ con IC 95 % salen de una
# logística **sin penalizar** sobre Train con errores robustos por clúster de búsqueda (la
# penalización sesga los coeficientes y no tiene IC exactos).

# %%
t0 = time.time()
grid_log = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(solver="saga", max_iter=3000)),
    {
        "logisticregression__penalty": ["l1", "l2"],
        "logisticregression__C": [0.001, 0.01, 0.1, 1, 10],
    },
    scoring="average_precision",
    cv=CV,
    n_jobs=-1,
)
grid_log.fit(X_sub, y_sub, groups=g_sub)
logit = clone(grid_log.best_estimator_).fit(X_tr, y_tr)
TIEMPOS["Logística"] = time.time() - t0
print("Mejor:", grid_log.best_params_, f"| PR-AUC CV = {grid_log.best_score_:.4f}")
coef = pd.Series(logit[-1].coef_[0], index=FEATURES)
print(f"Coeficientes anulados por la penalización: {int((coef == 0).sum())} de {len(coef)}")

# %%
X_or = sm.add_constant(X_tr)
logit_sm = sm.Logit(y_tr, X_or).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": g_tr}, maxiter=200
)
ic_or = np.exp(logit_sm.conf_int())
tabla_or = pd.DataFrame(
    {
        "OR": np.exp(logit_sm.params),
        "IC95_inf": ic_or[0],
        "IC95_sup": ic_or[1],
        "p": logit_sm.pvalues,
    }
).drop("const")
tabla_or.sort_values("p").round(4)

# %% [markdown]
# **Lectura.**
# - La grilla elige **L1 con C = 0,1** (PR-AUC en CV de 0,1415), pero ningún coeficiente se anula:
#   con 150.000 filas, todas las variables aportan algo.
# - Odds ratios (logit sin penalizar, errores por clúster de búsqueda; todos *ceteris paribus* y
#   **asociativos, no causales**):
#   - Cada posición más abajo multiplica las chances de reserva por 0,899 (IC 95 % [0,896; 0,902]).
#   - `location_score2` tiene el OR más alto (4,38 por unidad de score).
#   - Ser más caro que la mediana de la búsqueda las baja: OR 0,40 por unidad de log precio relativo.
#   - Las estrellas (1,18) y las reseñas (1,30) las suben; también que Expedia sea más barata que sus
#     competidores (`comp_saldo`, OR 1,12).
#   - El OR de `random_bool` (0,146) **no es un efecto**: refleja que el proveedor sobremuestreó
#     compras solo en el estrato del algoritmo (Fase 0).

# %% [markdown]
# ## 3.2 Familia 2 — Naive Bayes gaussiano
#
# $P(Y\mid X) \propto P(Y)\prod_j P(X_j \mid Y)$, con $X_j \mid Y \sim N(\mu_{jy}, \sigma^2_{jy})$.
# Elegimos `GaussianNB` porque la mayoría de las variables son continuas o de conteo; las
# binarias quedan aproximadas por una normal, lo que es parte del problema que se evalúa.
# **Supuesto clave:** independencia condicional de las $X_j$ dada la clase. Lo auditamos con las
# correlaciones **dentro de cada clase**: si fueran independientes, serían ≈ 0.

# %%
t0 = time.time()
nb = make_pipeline(StandardScaler(), GaussianNB()).fit(X_tr, y_tr)
TIEMPOS["Naive Bayes"] = time.time() - t0
corr_clase = {
    c: X_tr[y_tr == c].corr().abs().where(~np.eye(len(FEATURES), dtype=bool)) for c in (0, 1)
}
for c in (0, 1):
    v = corr_clase[c].stack()
    print(
        f"Clase {c}: |r| máx = {v.max():.3f} ({v.idxmax()}), pares con |r| > 0,3: {int((v > 0.3).sum() / 2)}"
    )

# %% [markdown]
# **Violación del supuesto.** Dentro de cada clase hay pares fuertemente correlacionados: hasta |r| =
# 0,721 entre el precio y el precio histórico, y 15 pares con |r| > 0,3 en la clase 0. Naive Bayes
# multiplica verosimilitudes como si fueran evidencia independiente, así que **cuenta varias veces la
# misma información** y empuja las probabilidades a los extremos. La consecuencia se ve en 3.5: es el
# modelo peor calibrado (Brier 0,0697 contra ≈ 0,020 de los demás).

# %% [markdown]
# ## 3.3 Familia 3 — Árbol CART y poda por complejidad de costo
#
# Primero un árbol **sin restricciones** (hasta hojas puras) para mostrar el sobreajuste. Después,
# la poda de complejidad de costo minimiza $R_\alpha(T) = R(T) + \alpha\,|T|$, donde $R(T)$ es la
# impureza total de las hojas y $|T|$ su número. `cost_complexity_pruning_path` da la sucesión de
# $\alpha$ en la que se van podando ramas; elegimos $\alpha_{\text{opt}}$ por PR-AUC con CV agrupada
# sobre una grilla de cuantiles de esa sucesión.

# %%
t0 = time.time()
arbol_libre = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)
p_libre_tr, p_libre_te = (
    arbol_libre.predict_proba(X_tr)[:, 1],
    arbol_libre.predict_proba(X_te)[:, 1],
)
print(
    f"Árbol sin restricciones: {arbol_libre.get_n_leaves():,} hojas, profundidad {arbol_libre.get_depth()} | "
    f"PR-AUC Train = {average_precision_score(y_tr, p_libre_tr):.4f}, Test = {average_precision_score(y_te, p_libre_te):.4f} | "
    f"ROC-AUC Train = {roc_auc_score(y_tr, p_libre_tr):.4f}, Test = {roc_auc_score(y_te, p_libre_te):.4f}"
)
camino = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X_sub, y_sub)
alfas = np.unique(
    np.quantile(camino.ccp_alphas[camino.ccp_alphas > 0], np.linspace(0.5, 0.999, 25))
)
grid_arbol = GridSearchCV(
    DecisionTreeClassifier(random_state=SEED),
    {"ccp_alpha": alfas},
    scoring="average_precision",
    cv=CV,
    n_jobs=-1,
    return_train_score=True,
)
grid_arbol.fit(X_sub, y_sub, groups=g_sub)
alfa_opt = grid_arbol.best_params_["ccp_alpha"]
arbol = DecisionTreeClassifier(ccp_alpha=alfa_opt, random_state=SEED).fit(X_tr, y_tr)
TIEMPOS["CART podado"] = time.time() - t0
print(
    f"Camino de poda: {len(camino.ccp_alphas):,} valores de α; α_opt = {alfa_opt:.3e} "
    f"(PR-AUC CV = {grid_arbol.best_score_:.4f}) | árbol podado: {arbol.get_n_leaves()} hojas, profundidad {arbol.get_depth()}"
)

# %%
cvr = pd.DataFrame(grid_arbol.cv_results_)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].plot(
    cvr["param_ccp_alpha"].astype(float),
    cvr["mean_train_score"],
    "o-",
    color=gr.AZUL,
    ms=4,
    label="Train (CV)",
)
ax[0].plot(
    cvr["param_ccp_alpha"].astype(float),
    cvr["mean_test_score"],
    "o-",
    color=gr.NARANJA,
    ms=4,
    label="Validación (CV)",
)
ax[0].fill_between(
    cvr["param_ccp_alpha"].astype(float),
    cvr["mean_test_score"] - cvr["std_test_score"],
    cvr["mean_test_score"] + cvr["std_test_score"],
    color=gr.NARANJA,
    alpha=0.2,
    lw=0,
)
ax[0].axvline(alfa_opt, color=gr.TEXTO_2, ls="--", lw=1)
ax[0].annotate(
    "α_opt",
    xy=(alfa_opt, ax[0].get_ylim()[0]),
    xytext=(4, 8),
    textcoords="offset points",
    fontsize=8,
    color=gr.TEXTO_2,
)
ax[0].set(
    xscale="log",
    xlabel="ccp_alpha (escala log)",
    ylabel="PR-AUC",
    title="Poda por complejidad de costo",
)
ax[0].legend()
plot_tree(
    arbol,
    max_depth=2,
    feature_names=FEATURES,
    class_names=["no reserva", "reserva"],
    filled=False,
    impurity=False,
    proportion=True,
    fontsize=7,
    ax=ax[1],
)
ax[1].set_title("Árbol podado: primeros niveles")
fig.tight_layout()
gr.guardar(fig, "f3_cart")
plt.show()
print(export_text(arbol, feature_names=FEATURES, max_depth=1))

# %% [markdown]
# **Sobreajuste.** El árbol sin restricciones crece hasta 18.896 hojas y profundidad 44. Memoriza Train
# (PR-AUC = 1,000) y fracasa en Test (PR-AUC = 0,031; ROC-AUC = 0,549, casi azar). La poda elige
# $\alpha_{\text{opt}}$ = 2,28·10⁻⁵ entre 1.174 valores del camino y deja **14 hojas** (profundidad 6),
# con PR-AUC de 0,120 en CV.
#
# **Regla del nodo raíz:** `position ≤ 4,5`. El primer corte separa los hoteles de las posiciones 1-4
# del resto: la posición es lo que más reduce la impureza. El segundo corte en ambas ramas es
# `random_bool`, que separa los estratos con tasas base muy distintas. Ninguno de los dos cortes es
# una palanca causal: es el punto de partida del ensayo de la Fase 4.

# %% [markdown]
# ## 3.4 Familia 4 — Random Forest con búsqueda de hiperparámetros
#
# Ensamble de $B$ árboles sobre muestras bootstrap y subconjuntos aleatorios de variables en cada
# corte (`max_features`), lo que decorrelaciona los árboles y reduce la varianza del promedio.
# `RandomizedSearchCV` (20 combinaciones, CV agrupada de 5 pliegues, `n_jobs=-1`) explora
# `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` y `max_features`
# (`sqrt`, `log2`, fracción). El error **out-of-bag** (OOB) usa, para cada fila, solo los árboles
# que no la vieron: es una validación "gratis" que comparamos con la CV.

# %%
t0 = time.time()
busqueda_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=1),
    {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, 24, None],
        "min_samples_split": [2, 10, 50, 200],
        "min_samples_leaf": [1, 5, 20, 50, 100],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    n_iter=20,
    scoring="average_precision",
    cv=CV,
    n_jobs=-1,
    random_state=SEED,
)
busqueda_rf.fit(X_sub, y_sub, groups=g_sub)
print("Mejor:", busqueda_rf.best_params_, f"| PR-AUC CV = {busqueda_rf.best_score_:.4f}")
rf = RandomForestClassifier(
    **busqueda_rf.best_params_, oob_score=True, random_state=SEED, n_jobs=-1
).fit(X_tr, y_tr)
TIEMPOS["Random Forest"] = time.time() - t0
p_oob = rf.oob_decision_function_[:, 1]
ok = ~np.isnan(p_oob)
print(
    f"OOB (Train completo): PR-AUC = {average_precision_score(y_tr[ok], p_oob[ok]):.4f}, ROC-AUC = {roc_auc_score(y_tr[ok], p_oob[ok]):.4f}, error de clasificación = {1 - rf.oob_score_:.4f}"
)

# %%
# Convergencia: OOB y CV en función del número de árboles (submuestra, mismos hiperparámetros)
param_conv = {k: v for k, v in busqueda_rf.best_params_.items() if k != "n_estimators"}
conv = []
for b in [10, 25, 50, 100, 200, 300]:
    m = RandomForestClassifier(
        n_estimators=b, oob_score=True, random_state=SEED, n_jobs=-1, **param_conv
    ).fit(X_sub, y_sub)
    po = m.oob_decision_function_[:, 1]
    okb = ~np.isnan(po)
    cv_b = cross_val_predict(
        clone(m).set_params(oob_score=False),
        X_sub,
        y_sub,
        groups=g_sub,
        cv=CV,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    conv.append(
        {
            "arboles": b,
            "PR_AUC_OOB": average_precision_score(y_sub[okb], po[okb]),
            "PR_AUC_CV": average_precision_score(y_sub, cv_b),
        }
    )
conv = pd.DataFrame(conv)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(conv.arboles, conv.PR_AUC_OOB, "o-", color=gr.AZUL, label="OOB")
ax.plot(conv.arboles, conv.PR_AUC_CV, "o-", color=gr.NARANJA, label="CV agrupada (5 pliegues)")
ax.set(xlabel="Número de árboles B", ylabel="PR-AUC", title="Convergencia del Random Forest")
ax.legend()
fig.tight_layout()
gr.guardar(fig, "f3_rf_convergencia")
plt.show()
conv.round(4)

# %% [markdown]
# **Búsqueda.** La mejor combinación es 200 árboles, `max_depth` = 24, `min_samples_leaf` = 20,
# `min_samples_split` = 2 y `max_features` = 0,3, con PR-AUC de 0,1574 en CV. **OOB vs. CV:** en Train
# completo, el OOB da PR-AUC 0,1512 y ROC-AUC 0,8458, en línea con la CV. En la curva de
# convergencia, ambas crecen con el número de árboles y se estabilizan desde ≈ 200 (OOB: 0,1504 →
# 0,1512 de 200 a 300 árboles; CV: 0,1548 → 0,1551). El OOB queda algo por debajo de la CV con pocos
# árboles porque cada fila se evalúa solo con el ≈ 37 % de los árboles que no la vieron. La
# diferencia se cierra al crecer B: es la convergencia que pide el enunciado.

# %% [markdown]
# ## 3.5 Evaluación en Test, calibración y umbral
#
# El umbral de clasificación se elige **en Train** (el que maximiza F1 en predicciones fuera de
# pliegue de la submuestra) y se aplica tal cual en Test. Con 2 % de positivos, el umbral 0,5 por
# defecto clasificaría casi todo como "no reserva".

# %%
MODELOS = {"Logística": logit, "Naive Bayes": nb, "CART podado": arbol, "Random Forest": rf}
prob_te, filas_test = {}, []
for nombre, m in MODELOS.items():
    p_oof = cross_val_predict(
        clone(m), X_sub, y_sub, groups=g_sub, cv=CV, method="predict_proba", n_jobs=-1
    )[:, 1]
    u = umbral_f1(y_sub, p_oof)
    p = m.predict_proba(X_te)[:, 1]
    prob_te[nombre] = p
    fila = {
        "modelo": nombre,
        **metricas(y_te, p, u),
        "F1 con umbral 0,5": f1_score(y_te, (p >= 0.5).astype(int)),
        "seg. entrenamiento": TIEMPOS[nombre],
    }
    filas_test.append(fila)
prob_te["Posición de Expedia (referencia)"] = 1 / te["position"].to_numpy()
tabla_test = pd.DataFrame(filas_test).set_index("modelo")
tabla_test.round(4)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
colores = {
    "Logística": gr.AZUL,
    "Naive Bayes": gr.NARANJA,
    "CART podado": gr.AQUA,
    "Random Forest": gr.AMARILLO,
}
for nombre, color in colores.items():
    p = prob_te[nombre]
    fpr, tpr, _ = roc_curve(y_te, p)
    ax[0].plot(fpr, tpr, color=color, label=f"{nombre} ({roc_auc_score(y_te, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p)
    ax[1].plot(rc, pr, color=color, label=f"{nombre} ({average_precision_score(y_te, p):.3f})")
    fr, mp = calibration_curve(y_te, p, n_bins=15, strategy="quantile")
    ax[2].plot(mp, fr, "o-", color=color, ms=4, label=nombre)
ax[0].plot([0, 1], [0, 1], color=gr.TEXTO_2, ls="--", lw=1)
ax[0].set(
    xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="ROC (AUC)"
)
ax[1].axhline(y_te.mean(), color=gr.TEXTO_2, ls="--", lw=1)
ax[1].annotate(
    "prevalencia",
    xy=(0.75, y_te.mean()),
    xytext=(0, 4),
    textcoords="offset points",
    fontsize=8,
    color=gr.TEXTO_2,
)
ax[1].set(xlabel="Recall", ylabel="Precisión", title="Precisión-Recall (PR-AUC)")
lim = max(max(prob_te[n].max() for n in colores), 0.5)
ax[2].plot([0, lim], [0, lim], color=gr.TEXTO_2, ls="--", lw=1)
ax[2].set(
    xscale="log",
    yscale="log",
    xlabel="Probabilidad predicha (media del bin, log)",
    ylabel="Frecuencia observada (log)",
    title="Reliability diagram (15 bins por cuantil)",
)
for a in ax:
    a.legend(fontsize=7.5, loc="best")
fig.tight_layout()
gr.guardar(fig, "f3_evaluacion")
plt.show()

# %% [markdown]
# **Ganador: Random Forest**, con PR-AUC 0,154 (unas 7 veces la prevalencia de 0,022), ROC-AUC 0,849 y
# el menor Brier (0,0196). Le siguen la logística (PR-AUC 0,135) y el CART podado (0,120). Naive
# Bayes queda último (0,074) y con un Brier de 0,0697: la violación de la independencia condicional
# **destruye la calibración**. Su umbral elegido en Train (0,9998) no se traslada a Test (recall
# 0,003).
#
# **Calibración (reliability diagram).** Logística, CART y RF quedan cerca de la diagonal; Naive Bayes
# se aleja en los extremos.
#
# **Umbral.** Con el umbral por defecto de 0,5, el F1 es 0 en logística y CART, y 0,0006 en RF: con
# 2 % de positivos nada supera 0,5. Con el umbral elegido en Train, el F1 de RF es 0,233 (precisión
# 0,187, recall 0,306). Es un experimento fallido documentado en la Entrada 3.

# %% [markdown]
# ## 3.6 Explicabilidad: MDI vs. importancia por permutación
#
# * **MDI** (disminución media de la impureza de Gini): suma, sobre todos los cortes que usan
#   $X_j$, la reducción de impureza ponderada por las filas que pasan por el nodo. Se calcula en
#   **Train**.
# * **Permutación:** cuánto cae la PR-AUC en **Test** cuando se permuta $X_j$ (rompiendo su
#   relación con $Y$ y con las demás variables), promediado en 5 repeticiones. Por cómputo, se usa
#   una submuestra del 25 % de las búsquedas de Test.

# %%
te_sub = submuestra_busquedas(te, 0.25)
perm = permutation_importance(
    rf,
    matriz(te_sub),
    te_sub[OBJETIVO].to_numpy(),
    scoring="average_precision",
    n_repeats=5,
    random_state=SEED,
    n_jobs=-1,
)
imp = pd.DataFrame(
    {
        "MDI": rf.feature_importances_,
        "Permutación (ΔPR-AUC)": perm.importances_mean,
        "Permutación_de": perm.importances_std,
    },
    index=FEATURES,
)
imp["cardinalidad"] = [tr[f].nunique() for f in FEATURES]
imp["rango_MDI"] = imp["MDI"].rank(ascending=False).astype(int)
imp["rango_perm"] = imp["Permutación (ΔPR-AUC)"].rank(ascending=False).astype(int)
rho_card = stats.spearmanr(imp["cardinalidad"], imp["MDI"])
print(
    f"Spearman cardinalidad vs. MDI: ρ = {rho_card.statistic:.3f} (p = {rho_card.pvalue:.3g}); vs. permutación: ρ = {stats.spearmanr(imp['cardinalidad'], imp['Permutación (ΔPR-AUC)']).statistic:.3f}"
)
imp.sort_values("MDI", ascending=False).round(4)

# %%
top = imp.sort_values("Permutación (ΔPR-AUC)", ascending=False).head(12).index[::-1]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
ax[0].barh(top, imp.loc[top, "MDI"], color=gr.AZUL, height=0.6)
ax[0].set(xlabel="MDI (Train)", title="Importancia por impureza (MDI)")
ax[1].barh(
    top,
    imp.loc[top, "Permutación (ΔPR-AUC)"],
    xerr=imp.loc[top, "Permutación_de"],
    color=gr.NARANJA,
    height=0.6,
    error_kw={"ecolor": gr.TEXTO_2, "lw": 1},
)
ax[1].set(xlabel="Caída de PR-AUC al permutar (Test)", title="Importancia por permutación")
fig.tight_layout()
gr.guardar(fig, "f3_importancias")
plt.show()

# %% [markdown]
# **El dilema MDI vs. permutación, con los datos.**
# - El MDI correlaciona fuertemente con la cardinalidad de las variables (ρ de Spearman = 0,749,
#   p = 7·10⁻⁶); la permutación, bastante menos (0,567).
# - Ejemplos:
#   - `orig_destination_distance` (178.299 valores distintos) es 8.ª en MDI y 10.ª por permutación,
#     con una caída de PR-AUC de apenas 0,0012.
#   - `prop_log_historical_price` es 7.ª en MDI y 11.ª por permutación.
#   - `prop_starrating` (5 valores) sube del 13.° puesto en MDI al 8.° por permutación.
#
# **Por qué:** el MDI suma reducciones de impureza **en Train**. Una variable continua ofrece miles de
# cortes posibles, y el árbol encuentra entre ellos alguno que reduce impureza por azar (sobre todo
# en nodos profundos con pocas filas). Una dummy ofrece un único corte. La permutación mide la caída
# de desempeño **en Test** cuando se rompe la variable: los cortes que solo servían para memorizar no
# generalizan y no cuentan.
#
# **Las dos más importantes por ambos criterios:** `position` (caída de PR-AUC de 0,103) y
# `random_bool` (0,079). La Fase 4 discute por qué ninguna de las dos es una palanca en la magnitud
# que sugiere su importancia.

# %% [markdown]
# ## 3.7 Sensibilidades propias
#
# 1. **Holdout cronológico** (pedido por Verónica): el mejor modelo se reentrena con las búsquedas
#    del primer 80 % del período y se evalúa en el último 20 % (por `date_time`), para ver si el
#    desempeño se sostiene hacia adelante.
# 2. **¿Ordena mejor que Expedia?** Entre las búsquedas de Test con reserva, en qué fracción el
#    hotel reservado está en el top-5 del modelo, comparado con el top-5 de Expedia y separando
#    orden del algoritmo (R = 0) y orden aleatorio (R = 1).
# 3. **PCA como features** (Fase 2): la logística con las 4 componentes de Train en lugar de las
#    10 variables de hotel originales.

# %%
inicio_busq = df.groupby("srch_id")["date_time"].min().sort_values()
corte = inicio_busq.quantile(0.8)
ids_futuro = set(inicio_busq[inicio_busq > corte].index)
pasado, futuro = df[~df.srch_id.isin(ids_futuro)], df[df.srch_id.isin(ids_futuro)]
mejor = max(MODELOS, key=lambda n: tabla_test.loc[n, "PR_AUC"])
m_crono = clone(MODELOS[mejor]).fit(matriz(pasado), pasado[OBJETIVO])
p_fut = m_crono.predict_proba(matriz(futuro))[:, 1]
print(f"Mejor modelo: {mejor} | corte cronológico: {corte:%Y-%m-%d %H:%M}")
print(
    f"Holdout aleatorio: PR-AUC = {tabla_test.loc[mejor, 'PR_AUC']:.4f}, ROC-AUC = {tabla_test.loc[mejor, 'ROC_AUC']:.4f} | "
    f"Holdout cronológico ({futuro.srch_id.nunique():,} búsquedas): PR-AUC = {average_precision_score(futuro[OBJETIVO], p_fut):.4f}, "
    f"ROC-AUC = {roc_auc_score(futuro[OBJETIVO], p_fut):.4f}, prevalencia {futuro[OBJETIVO].mean():.4f}"
)

# %%
filas_rank = []
for r in (0, 1):
    d = te[te.random_bool == r]
    filas_rank.append(
        {
            "estrato": f"R = {r}",
            "búsquedas con reserva": int(d.groupby("srch_id")[OBJETIVO].max().sum()),
            "top-5 de Expedia": acierto_top_k(d, -d["position"], 5),
            f"top-5 del modelo ({mejor})": acierto_top_k(
                d, pd.Series(MODELOS[mejor].predict_proba(matriz(d))[:, 1]), 5
            ),
        }
    )
pd.DataFrame(filas_rank).set_index("estrato").round(4)

# %%
X_HOTEL_PCA = X_PCA  # las 10 variables de hotel de la Fase 2
otras = [f for f in FEATURES if f not in X_HOTEL_PCA]
Z_tr = pd.DataFrame(
    pca_pipe.transform(tr[X_HOTEL_PCA].astype(float))[:, :4],
    columns=[f"PC{i}" for i in range(1, 5)],
    index=tr.index,
)
Z_te = pd.DataFrame(
    pca_pipe.transform(te[X_HOTEL_PCA].astype(float))[:, :4], columns=Z_tr.columns, index=te.index
)
logit_pca = clone(grid_log.best_estimator_).fit(pd.concat([Z_tr, matriz(tr, otras)], axis=1), y_tr)
p_pca = logit_pca.predict_proba(pd.concat([Z_te, matriz(te, otras)], axis=1))[:, 1]
print(
    f"Logística con 4 PCs + resto: PR-AUC Test = {average_precision_score(y_te, p_pca):.4f} vs. {tabla_test.loc['Logística', 'PR_AUC']:.4f} con las variables originales"
)

# %% [markdown]
# 1. **Holdout cronológico.** Entrenado con el 80 % inicial del período y evaluado en las búsquedas
#    posteriores al 21/05/2013, el RF da PR-AUC 0,146 y ROC-AUC 0,841, contra 0,154 y 0,849 en el
#    holdout aleatorio. La pérdida es chica: el modelo no depende de patrones que no se repiten en el
#    tiempo.
# 2. **¿Ordena mejor que Expedia?** En las búsquedas de Test con reserva, el hotel reservado está en
#    el top-5 del RF en el 62,8 % de los casos con R = 0, contra 60,5 % del top-5 de Expedia, y en el
#    53,5 % con R = 1, contra 49,4 %.
#    - Con orden aleatorio, el top-5 de Expedia **no** debería anticipar nada. Que acierte el 49 % es
#      el efecto causal de la posición: el usuario reserva más lo que ve arriba.
#    - El modelo, que usa la posición, hereda ese efecto y lo mejora apenas.
# 3. **PCA como features.** Reemplazar las 10 variables de hotel por las 4 componentes de Train baja
#    la PR-AUC de la logística de 0,135 a 0,119: la compresión pierde información predictiva. Se
#    mantienen las variables originales.

# %% [markdown]
# ## 3.8 Síntesis de la Fase 3
#
# | Modelo | PR-AUC (Test) | ROC-AUC | Brier | Lectura |
# |---|---|---|---|---|
# | Random Forest | 0,154 | 0,849 | 0,0196 | Ganador; OOB y CV convergen |
# | Logística L1 | 0,135 | 0,844 | 0,0198 | Interpretable (OR con IC); bien calibrada |
# | CART podado | 0,120 | 0,823 | 0,0198 | 14 hojas; regla raíz `position ≤ 4,5` |
# | Naive Bayes | 0,074 | 0,796 | 0,0697 | Viola la independencia condicional; mal calibrado |
#
# - La posición es la variable más predictiva por los dos criterios de importancia.
# - El modelo se sostiene en el holdout cronológico.
# - La pregunta que la Fase 3 **no puede responder** es cuánto de esa importancia es causal: Fase 4.

# %% [markdown]
# ---
# # Fase 4 — Inferencia causal, Causal ML y el choque causal vs. predictivo
#
# ## 4.1 DAG estructural y criterio Backdoor
#
# **Pregunta causal:** $\tau = \mathbb{E}[Y(1) - Y(0)]$, el efecto de que el recomendador exponga un
# hotel en el top-5 ($T$ = `top5`) sobre su reserva ($Y$ = `booking_bool`). El DAG (`tpi.dag`;
# justificación arco por arco en la BITÁCORA, Entrada 4) tiene:
#
# * **Confusores observados** $Z = \{Z_h, Z_p, Z_b, Z_u, Z_c\}$: calidad y precio, popularidad previa,
#   contexto de la búsqueda, historial del usuario y competidores (Fase 1, 1.4).
# * **Latentes:** $U$ (desempeño histórico del hotel que usa el ranker) e $I$ (intención de compra de
#   la sesión).
# * **Mediador** $M$ = clic. **Colisionadores:** $M$ mismo ($T \to M \leftarrow I$), la selección del
#   dataset $S$ (≥ 1 clic por búsqueda) y un colisionador simulado $C$ ("destacado en el reporte",
#   $T \to C \leftarrow Y$).
# * **Experimento natural:** con `random_bool` = 1 la intervención corta todos los arcos que entran a
#   $T$ salvo $R \to T$.
#
# **Demostración analítica** (d-separación sobre el grafo; `tests/test_dag.py`): con el orden del
# algoritmo, $Z$ bloquea todos los caminos de puerta trasera $T \leftarrow Z \to Y$ pero **no**
# $T \leftarrow U \to Y$. Backdoor se cumpliría solo si $U$ fuera observable. Con orden aleatorio, el
# conjunto vacío alcanza.

# %%
import logging

from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

from tpi.causal import (
    efecto_marginal_real,
    logit_condicional,
    lpm_efectos_fijos,
    modelo_dowhy,
    simular_colisionador,
    simular_listas,
)
from tpi.dag import CONFUSORES, a_dot, construir_dag, cumple_backdoor, dibujar_dag

g_alg, g_rand = construir_dag("algoritmo"), construir_dag("aleatorio")
Zs = set(CONFUSORES)
pruebas_bd = pd.DataFrame(
    [
        ("Algoritmo (R=0)", "∅ (ingenuo)", cumple_backdoor(g_alg, "T", "Y", set())),
        ("Algoritmo (R=0)", "Z observado", cumple_backdoor(g_alg, "T", "Y", Zs)),
        ("Algoritmo (R=0)", "Z ∪ {U} (hipotético)", cumple_backdoor(g_alg, "T", "Y", Zs | {"U"})),
        (
            "Algoritmo (R=0)",
            "Z ∪ {M} (trampa del mediador)",
            cumple_backdoor(g_alg, "T", "Y", Zs | {"M", "U"}),
        ),
        (
            "Algoritmo (R=0)",
            "Z ∪ {U} sin Zc",
            cumple_backdoor(g_alg, "T", "Y", (Zs - {"Zc"}) | {"U"}),
        ),
        ("Aleatorio (R=1)", "∅", cumple_backdoor(g_rand, "T", "Y", set())),
        (
            "Aleatorio (R=1)",
            "{S} (selección del dataset)",
            cumple_backdoor(g_rand, "T", "Y", {"S"}),
        ),
        (
            "Aleatorio (R=1)",
            "{C} (colisionador simulado)",
            cumple_backdoor(g_rand, "T", "Y", {"C"}),
        ),
    ],
    columns=["grafo", "conjunto de ajuste", "cumple Backdoor"],
)
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
dibujar_dag(g_alg, ax[0], "Orden del algoritmo (R = 0)")
dibujar_dag(g_rand, ax[1], "Orden aleatorio (R = 1): do(T)")
fig.tight_layout()
gr.guardar(fig, "f4_dag")
plt.show()
pruebas_bd

# %% [markdown]
# **Variables del ajuste** (expansión de los nodos de $Z$ a columnas). En las estimaciones **dentro
# de la búsqueda** (efectos fijos, logit condicional) las variables de búsqueda y usuario ($Z_b$,
# $Z_u$) son constantes en cada lista: quedan absorbidas por el efecto fijo, junto con todo confusor
# de nivel búsqueda, observado o no (intención de la sesión, destino, fecha).

# %%
Z_HOTEL = [
    "prop_starrating", "prop_review_score", "sin_resenas", "prop_location_score1", "prop_location_score2",
    "prop_location_score2_faltante", "prop_brand_bool", "log_precio_rel", "promotion_flag",
    "prop_log_historical_price", "sin_historial_precio", "con_afinidad", "comp_n_info", "comp_saldo",
]  # fmt: skip
Z_BUSQ = [
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "usuario_recurrente",
]
# log_precio solo en el ajuste agrupado: dentro de la búsqueda es casi colineal con log_precio_rel
# (difieren en la mediana de la lista, que absorbe el efecto fijo; ia_log, error 27)
Z_TODAS = Z_HOTEL + ["log_precio"] + Z_BUSQ
dc = df.copy()
dc["comp_saldo"] = dc["comp_saldo"].fillna(0.0)
R0, R1 = dc[dc.random_bool == 0], dc[dc.random_bool == 1]
print(
    f"R = 0: {len(R0):,} filas, {R0.srch_id.nunique():,} búsquedas | R = 1: {len(R1):,} filas, {R1.srch_id.nunique():,} búsquedas"
)

# %% [markdown]
# ## 4.2 Flujo DoWhy en 4 pasos (orden del algoritmo, R = 0)
#
# **Model** (DAG con columnas reales y el clic como mediador) → **Identify** (Backdoor) →
# **Estimate** (regresión lineal con ajuste y ponderación por propensión inversa) → **Refute**
# (sección 4.6). El efecto es una diferencia de probabilidades (escala absoluta); su
# comparabilidad entre estratos está limitada por la selección del dataset (Entrada 4, amenaza A).

# %%
modelo0, estimando0 = modelo_dowhy(R0, "top5", "booking_bool", Z_TODAS, "click_bool")
print(estimando0)

# %%
ate = {}
for nombre, d in [("R = 0 (algoritmo)", R0), ("R = 1 (aleatorio)", R1)]:
    mod, est = (
        (modelo0, estimando0)
        if d is R0
        else modelo_dowhy(d, "top5", "booking_bool", Z_TODAS, "click_bool")
    )
    ing = smf.ols("booking_bool ~ top5", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["srch_id"]}
    )
    lin = mod.estimate_effect(est, "backdoor.linear_regression", confidence_intervals=True)
    # Modelo de propensión escalado: el logit por defecto de DoWhy no converge con covariables en
    # escalas muy distintas (ia_log, error 29). Se exige convergencia.
    with warnings.catch_warnings(record=True) as avisos_ps:
        warnings.simplefilter("always", ConvergenceWarning)
        ipw = mod.estimate_effect(
            est,
            "backdoor.propensity_score_weighting",
            method_params={
                "weighting_scheme": "ips_weight",
                "propensity_score_model": make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=3000)
                ),
            },
        )
    assert not any(issubclass(a.category, ConvergenceWarning) for a in avisos_ps), (
        "propensión sin converger"
    )
    ate[nombre] = {
        "ingenuo (Y ~ T)": ing.params["top5"],
        "ingenuo IC95": tuple(ing.conf_int().loc["top5"].round(4)),
        "regresión backdoor": lin.value,
        "regresión IC95": tuple(np.round(lin.get_confidence_intervals()[0], 4)),
        "IPW": ipw.value,
        "tasa base P(Y|T=0)": d.loc[d.top5 == 0, "booking_bool"].mean(),
    }
    if d is R0:
        est_lin0, est_ipw0 = lin, ipw
tabla_ate = pd.DataFrame(ate).T
tabla_ate["sesgo de confusión (ingenuo − regresión)"] = (
    tabla_ate["ingenuo (Y ~ T)"] - tabla_ate["regresión backdoor"]
)
tabla_ate["sesgo relativo %"] = (
    100 * tabla_ate["sesgo de confusión (ingenuo − regresión)"] / tabla_ate["regresión backdoor"]
)
tabla_ate

# %% [markdown]
# **Lectura.** El estimando identificado es Backdoor con $Z$ (DoWhy excluye el clic porque es mediador).
# Con el orden del algoritmo, la diferencia ingenua (0,1055) baja a 0,0961 con regresión y a 0,0890
# con IPW (propensión escalada y con convergencia verificada): el ajuste por $Z$ remueve ≈ 9 %
# (regresión) y ≈ 16 % (IPW) del coeficiente ingenuo. La
# **dirección** del sesgo es la esperada: el algoritmo pone arriba a hoteles que se reservarían igual,
# así que la comparación ingenua sobreestima.
#
# Las magnitudes absolutas de R = 0 y R = 1 **no son comparables**: la tasa base es 0,0206 contra
# 0,0037 porque el proveedor sobremuestreó búsquedas con compra solo en R = 0 (Entrada 4, amenaza A).
# Además, el ajuste agrupado no controla los confusores de nivel búsqueda. Por eso la comparación
# principal es la de 4.3.

# %% [markdown]
# ## 4.3 Escala relativa dentro de la búsqueda: observacional vs. cuasi-experimento
#
# **Estimador principal** (decisión del 12/11): **logit condicional** por búsqueda,
# $P(Y_{is}=1 \mid \textstyle\sum_i Y_{is}, X) = \frac{\exp(\tau T_{is} + Z_{is}^\top\gamma)}{\sum_{j \in s}\exp(\tau T_{js} + Z_{js}^\top\gamma)}$,
# un modelo de elección: "dado que en esta lista hubo una reserva, ¿cuánto aumenta el top-5 la
# chance de que sea este hotel?". $e^\tau$ es un **OR intra-búsqueda**, invariante a la selección
# de listas por su resultado (`tests/test_causal.py::test_or_condicional_sobrevive_al_filtro_de_seleccion`).
#
# * **Benchmark (R = 1):** orden (casi) aleatorio. Se informa sin ajuste y ajustado por $Z$ por la
#   aleatorización imperfecta (Entrada 4, amenaza B).
# * **Observacional (R = 0):** sin ajuste (ingenuo) y ajustado por $Z$ (Backdoor).
# * Como escala absoluta, el MPL con efectos fijos por búsqueda.

# %%
filas_rel = []
for estrato, d in [("R = 0 (algoritmo)", R0), ("R = 1 (aleatorio)", R1)]:
    for ajuste, cov in [("sin ajuste", []), ("ajustado por Z", Z_HOTEL)]:
        cl = logit_condicional(d, "top5", "booking_bool", cov)
        fe = lpm_efectos_fijos(d, "top5", "booking_bool", cov)
        filas_rel.append(
            {
                "estrato": estrato,
                "ajuste": ajuste,
                "OR intra": cl.estimacion,
                "OR IC95": tuple(np.round(cl.ic95, 3)),
                "p": cl.p_valor,
                "búsquedas informativas": cl.n,
                "MPL-EF (Δp)": fe.estimacion,
                "Δp IC95": tuple(np.round(fe.ic95, 4)),
            }
        )
tabla_rel = pd.DataFrame(filas_rel).set_index(["estrato", "ajuste"])
or_ = tabla_rel["OR intra"]
or_ing0, or_adj0 = (
    or_[("R = 0 (algoritmo)", "sin ajuste")],
    or_[("R = 0 (algoritmo)", "ajustado por Z")],
)
or_bench = or_[("R = 1 (aleatorio)", "ajustado por Z")]
brecha = np.log(or_ing0 / or_bench)
print(f"Brecha total en log-OR (ingenuo R=0 vs. benchmark R=1 ajustado): {brecha:.3f}")
print(
    f"  explicada por Z observado: {np.log(or_ing0 / or_adj0):.3f} ({100 * np.log(or_ing0 / or_adj0) / brecha:.1f} %)"
)
print(
    f"  residual (compatible con confusión no observada U): {np.log(or_adj0 / or_bench):.3f} ({100 * np.log(or_adj0 / or_bench) / brecha:.1f} %)"
)
print(
    f"  el estimador observacional ajustado sobrestima el OR en un {100 * (or_adj0 / or_bench - 1):.1f} %"
)
sens_t = logit_condicional(R1, "top4_dataset", "booking_bool", Z_HOTEL)
print(
    f"Sensibilidad T = top4_dataset (R = 1, ajustado): OR = {sens_t.estimacion:.3f} IC95 [{sens_t.ic95[0]:.3f}; {sens_t.ic95[1]:.3f}]"
)
tabla_rel

# %%
fig, ax = plt.subplots(figsize=(8, 3.6))
etiquetas = [f"{e} · {a}" for e, a in tabla_rel.index]
y_pos = np.arange(len(tabla_rel))[::-1]
for yy, (idx, fila) in zip(y_pos, tabla_rel.iterrows(), strict=True):
    color = gr.AZUL if idx[0].startswith("R = 0") else gr.NARANJA
    ax.errorbar(
        fila["OR intra"],
        yy,
        xerr=[[fila["OR intra"] - fila["OR IC95"][0]], [fila["OR IC95"][1] - fila["OR intra"]]],
        fmt="o",
        color=color,
        capsize=4,
        ms=7,
    )
ax.axvline(or_bench, color=gr.NARANJA, ls="--", lw=1)
ax.annotate(
    "benchmark (R = 1 ajustado)",
    xy=(or_bench, y_pos[0] + 0.35),
    xytext=(6, 0),
    textcoords="offset points",
    fontsize=8,
    color=gr.TEXTO_2,
)
ax.set(
    yticks=y_pos,
    yticklabels=etiquetas,
    xscale="log",
    xlabel="OR intra-búsqueda del top-5 sobre la reserva (IC 95 %, escala log)",
    title="Observacional vs. cuasi-experimento",
)
fig.tight_layout()
gr.guardar(fig, "f4_or_comparacion")
plt.show()

# %% [markdown]
# **Resultado central del TP.**
#
# | | OR intra-búsqueda (IC 95 %) |
# |---|---|
# | Orden del algoritmo, ingenuo | 5,38 [5,20; 5,57] |
# | Orden del algoritmo, ajustado por $Z$ | 3,38 [3,25; 3,51] |
# | **Orden aleatorio, ajustado (benchmark)** | **2,11 [1,90; 2,35]** |
# | Orden aleatorio, sin ajuste | 3,00 [2,73; 3,30] |
#
# 1. **El top-5 tiene efecto causal:** con orden aleatorio, estar en el top-5 multiplica por 2,11 las
#    chances de que el hotel sea el reservado de su lista. El IC excluye 1 ($p$ = 1,2·10⁻⁴⁴).
# 2. **El estimador observacional sobreestima, aun ajustado.** Ajustar por $Z$ cierra la mitad de la
#    brecha en log-odds (ver la salida de arriba). El OR observacional ajustado (3,38) sigue un 60 %
#    por encima del benchmark, y sus IC no se superponen. Es lo que predice el DAG: $Z$ no bloquea
#    $T \leftarrow U \to Y$. **El supuesto Backdoor con $Z$ observado se rechaza empíricamente.**
# 3. Con orden aleatorio, ajustar por $Z$ también baja el OR (de 3,00 a 2,11). Es coherente con la
#    aleatorización imperfecta (amenaza B): por eso el benchmark es el ajustado. Una diferencia por la
#    no colapsabilidad del OR iría en sentido contrario (alejaría el OR de 1), así que no explica la
#    baja.
# 4. La sensibilidad con $T$ = top-4 del dataset (salida de arriba) no cambia la conclusión.
#
# Esta conclusión no depende de la escala absoluta, que la selección del dataset distorsiona: el OR
# intra-búsqueda es invariante a ese filtro (`test_or_condicional_sobrevive_al_filtro_de_seleccion` y
# sección 4.5).

# %% [markdown]
# ## 4.4 La trampa del mediador (clic)
#
# **Pregunta trampa del enunciado:** ¿qué pasa si el modelo causal incluye el mediador $M$? En el
# DAG todo el efecto del top-5 pasa por el clic ($T \to M \to Y$; no hay reserva sin clic). Ajustar
# por $M$ bloquea ese canal y estima solo el efecto directo, que debería ser ≈ 0. Además, como
# $M$ es colisionador ($T \to M \leftarrow I$), condicionar en él abre $T - I - Y$.

# %%
med = {
    "efecto total (ajuste Z)": lpm_efectos_fijos(R0, "top5", "booking_bool", Z_HOTEL),
    "ajustando además por el clic (M)": lpm_efectos_fijos(
        R0, "top5", "booking_bool", Z_HOTEL + ["click_bool"]
    ),
}
solo_clic = R0[R0.click_bool == 1]
med["solo hoteles clickeados (condicionar en M)"] = lpm_efectos_fijos(
    solo_clic, "top5", "booking_bool", Z_HOTEL
)
tabla_med = pd.DataFrame(
    {
        k: {"Δp": v.estimacion, "IC95": tuple(np.round(v.ic95, 4)), "p": v.p_valor}
        for k, v in med.items()
    }
).T
print(
    f"El clic absorbe el {100 * (1 - med['ajustando además por el clic (M)'].estimacion / med['efecto total (ajuste Z)'].estimacion):.1f} % del efecto total (R = 0)"
)
print(
    f"P(reserva | clic) en R = 0: top-5 = {solo_clic.loc[solo_clic.top5 == 1, 'booking_bool'].mean():.4f}, resto = {solo_clic.loc[solo_clic.top5 == 0, 'booking_bool'].mean():.4f}"
)
tabla_med

# %% [markdown]
# **Trampa del mediador confirmada.** Ajustar por el clic baja el efecto de 0,0778 a 0,0042: el clic
# absorbe el ≈ 95 % del efecto total. Tal como dice el DAG, el top-5 actúa casi enteramente a través del
# clic, y un modelo que "controla" por el clic concluiría erróneamente que la posición casi no importa.
#
# **Condicionar en el clic** (analizar solo los hoteles clickeados) da 0,115 con un IC muy amplio
# [0,045; 0,185]. No es ni el efecto total ni el directo: al seleccionar sobre un colisionador
# ($T \to M \leftarrow I$), la exposición queda asociada con la intención de compra. Es la trampa
# típica de "miremos solo a quienes hicieron clic".

# %% [markdown]
# ## 4.5 La trampa del colisionador
#
# **(a) Colisionador natural: la selección del dataset ($S$).** Todo el dataset está condicionado
# en "≥ 1 clic en la búsqueda" (y, en R = 0, sobremuestreado en compras). Lo simulamos con verdad
# conocida: listas de 25 hoteles con orden aleatorio, efecto del top-5 sobre el clic con OR = 2,23
# ($e^{0{,}8}$) e intención latente por lista. Comparamos los estimadores con todas las listas y solo
# con las seleccionadas.
#
# **(b) Colisionador simulado $C$** sobre los datos reales con R = 1: "el hotel aparece en el
# reporte de destacados" si fue expuesto o vendió, $C \sim \text{Bernoulli}(\operatorname{expit}(-3 + 2T + 2{,}5Y))$.
# Con orden aleatorio, $T$ es (casi) independiente de los atributos del hotel; condicionar en $C$ debería
# asociar $T$ con las causas de $Y$ (Berkson) y sesgar el efecto. Lo contrastamos también con una
# simulación de verdad conocida (`tpi.causal.simular_colisionador`), donde la calidad $Z_h$ causa $Y$ con fuerza.

# %%
sim = simular_listas(n_listas=4000, efecto_logit=0.8)
sel = sim[sim.groupby("srch_id")["Y"].transform("max") == 1]
filas_s = []
for nombre, d in [("todas las listas", sim), ("solo listas con ≥ 1 evento (S = 1)", sel)]:
    cl, fe = (
        logit_condicional(d, "T", "Y", ["calidad"]),
        lpm_efectos_fijos(d, "T", "Y", ["calidad"]),
    )
    filas_s.append(
        {
            "muestra": nombre,
            "listas": d.srch_id.nunique(),
            "OR intra": cl.estimacion,
            "OR IC95": tuple(np.round(cl.ic95, 3)),
            "Δp (MPL-EF)": fe.estimacion,
            "Δp IC95": tuple(np.round(fe.ic95, 4)),
        }
    )
tabla_s = pd.DataFrame(filas_s).set_index("muestra")
print(f"Verdad: OR = {np.exp(0.8):.3f}; Δp marginal = {efecto_marginal_real(sim, 0.8):.4f}")
tabla_s

# %%
rng_c = np.random.default_rng(SEED)
r1c = R1.copy()
r1c["C"] = (
    rng_c.uniform(size=len(r1c)) < expit(-3 + 2 * r1c["top5"] + 2.5 * r1c["booking_bool"])
).astype(int)
filas_c = []
for nombre, d in [("todas las filas (R = 1)", r1c), ("condicionando en C = 1", r1c[r1c.C == 1])]:
    # location_score2: el atributo de hotel que más predice la reserva (OR 4,38 en la Fase 3)
    rho_tz = stats.pearsonr(d["top5"], d["prop_location_score2"])
    ef = smf.ols("booking_bool ~ top5", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["srch_id"]}
    )
    filas_c.append(
        {
            "muestra": nombre,
            "n": len(d),
            "corr(T, location_score2)": rho_tz.statistic,
            "p corr": rho_tz.pvalue,
            "Δp ingenuo": ef.params["top5"],
            "Δp IC95": tuple(ef.conf_int().loc["top5"].round(4)),
        }
    )
tabla_c = pd.DataFrame(filas_c).set_index("muestra")
tabla_c

# %%
# Contraste con verdad conocida (tpi.causal.simular_colisionador): T aleatorio, Zh → Y fuerte
sim_c = simular_colisionador()
filas_cs = []
for nombre, d in [("todas las filas", sim_c), ("condicionando en C = 1", sim_c[sim_c.C == 1])]:
    filas_cs.append(
        {
            "muestra": nombre,
            "n": len(d),
            "corr(T, Zh)": stats.pearsonr(d["T"], d["Zh"]).statistic,
            "Δp ingenuo": d.groupby("T")["Y"].mean().diff().iloc[-1],
        }
    )
tabla_cs = pd.DataFrame(filas_cs).set_index("muestra")
tabla_cs

# %%
res_gasto = R1[R1.booking_bool == 1]
g_t = res_gasto.groupby("top5")["gross_bookings_usd"].median()
print(
    f"(c) Trampa de selección en el gasto (R = 1, solo reservas): mediana del gasto top-5 = {g_t[1]:.2f} USD vs. resto = {g_t[0]:.2f} USD (n = {len(res_gasto)})"
)
print(
    "    Condicionar en 'reservó' compara reservantes distintos: no es el efecto del top-5 sobre el gasto esperado."
)

# %% [markdown]
# **(a) Selección del dataset (simulación con verdad conocida).** Con todas las listas, el logit
# condicional da OR = 2,320 (IC 95 % [2,156; 2,496]; verdad $e^{0{,}8}$ = 2,226) y el MPL da
# Δp = 0,0338 (verdad 0,0321). Con solo las listas seleccionadas (≥ 1 evento: 1.964 de 4.000):
# - **el OR es exactamente el mismo** (las listas sin eventos no aportan a la verosimilitud
#   condicional);
# - **la diferencia absoluta se duplica** (0,0684 [0,0615; 0,0752]).
#
# Es la justificación cuantitativa de usar la escala relativa intra-búsqueda (Entrada 4, amenaza A, c).
#
# **(b) Colisionador C.**
# - *Simulación con verdad conocida* (calidad $Z_h$ que causa fuerte la reserva): condicionar en
#   $C = 1$ crea una asociación negativa entre $T$ y $Z_h$ (correlación de −0,004 a −0,062: Berkson)
#   y **cambia el signo** del efecto ingenuo (de +0,093 a −0,089).
# - *Sobre los datos reales con orden aleatorio*, el mismo mecanismo produce un sesgo más chico: la Δp
#   ingenua pasa de 0,0105 a 0,0136 (+29 %, con IC superpuestos) y la correlación de $T$ con
#   `location_score2` cambia poco (0,105 → 0,126).
# - La diferencia se explica porque con orden aleatorio la reserva es rara (≈ 0,5 %) y depende poco de
#   cada atributo por separado. El sesgo de Berkson es proporcional a cuánto causan $T$ e $Y$ al
#   colisionador y a cuánto causa el atributo a $Y$.
# - *Lección:* la dirección y el tamaño del sesgo por colisionador **no son predecibles sin modelar
#   el mecanismo**. La única regla segura es no condicionar.
#
# (La correlación de base entre $T$ y `location_score2` con R = 1 es la aleatorización imperfecta de
# la amenaza B.)
#
# **(c) Gasto.** Entre las reservas con orden aleatorio, la mediana del gasto es 208,17 USD en el
# top-5 y 233,36 USD fuera de él. Comparar el gasto solo entre reservas compara **reservantes
# distintos**: es otra selección sobre el resultado y no estima el efecto del top-5 sobre el gasto
# esperado.

# %% [markdown]
# ## 4.6 Batería de refutaciones (DoWhy) sobre la regresión backdoor de R = 0
#
# Las tres del enunciado más dos extras:
# 1. **Causa común aleatoria:** se agrega una covariable de ruido independiente; el ATE no debería moverse.
# 2. **Placebo:** se permuta $T$; el ATE debe ir a 0 ($p > 0{,}05$).
# 3. **Subconjunto de datos:** se reestima en submuestras del 80 %; debe quedar dentro del IC original.
# 4. *(Extra)* **Causa común no observada** (sensibilidad): se simula un confusor de fuerza creciente.
# 5. *(Extra)* **Experimento natural:** la refutación más fuerte es la de 4.3, porque compara contra
#    el benchmark con orden aleatorio.

# %%
logging.disable(logging.WARNING)
N_SIM = 20
refut = {}
t0 = time.time()
refut["Causa común aleatoria"] = modelo0.refute_estimate(
    estimando0, est_lin0, method_name="random_common_cause", num_simulations=N_SIM, random_seed=SEED
)
refut["Placebo (T permutado)"] = modelo0.refute_estimate(
    estimando0,
    est_lin0,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
    num_simulations=N_SIM,
    random_seed=SEED,
)
refut["Subconjunto 80 %"] = modelo0.refute_estimate(
    estimando0,
    est_lin0,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    num_simulations=N_SIM,
    random_seed=SEED,
)
ic_lin0 = est_lin0.get_confidence_intervals()[0]
tabla_refut = pd.DataFrame(
    {
        k: {
            "ATE original": r.estimated_effect,
            "ATE refutado": r.new_effect,
            "p": r.refutation_result["p_value"],
        }
        for k, r in refut.items()
    }
).T
tabla_refut["criterio"] = [
    "estable si p > 0,05",
    "pasa si ATE ≈ 0 y p > 0,05 (no se rechaza que el placebo sea nulo)",
    f"pasa si queda en el IC original [{ic_lin0[0]:.4f}; {ic_lin0[1]:.4f}]",
]
tabla_refut["pasa"] = [
    tabla_refut.iloc[0]["p"] > 0.05,
    abs(tabla_refut.iloc[1]["ATE refutado"]) < 0.1 * abs(est_lin0.value)
    and tabla_refut.iloc[1]["p"] > 0.05,
    ic_lin0[0] <= tabla_refut.iloc[2]["ATE refutado"] <= ic_lin0[1],
]
print(f"Refutaciones: {N_SIM} simulaciones cada una, {time.time() - t0:.0f} s")
tabla_refut

# %%
sens_u = []
for k_t, k_y in [(0.01, 0.01), (0.02, 0.02), (0.05, 0.05), (0.1, 0.1)]:
    r = modelo0.refute_estimate(
        estimando0,
        est_lin0,
        method_name="add_unobserved_common_cause",
        confounders_effect_on_treatment="binary_flip",
        confounders_effect_on_outcome="linear",
        effect_strength_on_treatment=k_t,
        effect_strength_on_outcome=k_y,
    )
    sens_u.append(
        {
            "fuerza sobre T (fracción de T invertida)": k_t,
            "fuerza sobre Y": k_y,
            "ATE con U simulado": r.new_effect,
        }
    )
tabla_u = pd.DataFrame(sens_u)
tabla_u["cambio %"] = 100 * (tabla_u["ATE con U simulado"] / est_lin0.value - 1)
tabla_u

# %% [markdown]
# **Las tres refutaciones del enunciado pasan:**
#
# | Refutación | ATE original | ATE refutado | p | Resultado |
# |---|---|---|---|---|
# | Causa común aleatoria | 0,0961 | 0,0961 | 0,485 | Estable |
# | Placebo (T permutado) | 0,0961 | −0,0002 | 0,416 | Converge a 0 |
# | Subconjunto 80 % | 0,0961 | 0,0962 | 0,443 | Dentro del IC [0,0944; 0,0978] |
#
# **Lectura crítica.** Pasar estas pruebas indica que la estimación es **estable**: no depende del
# azar de la muestra ni de una covariable espuria, y un tratamiento sin sentido no produce efecto.
# **No indica que no haya confusión no observada.** La sensibilidad a un confusor simulado lo muestra:
# uno que invierta el 5 % de los tratamientos y tenga una fuerza comparable sobre Y reduce el ATE un
# 48 %, y uno del 10 % le cambia el signo. La prueba que sí detecta la confusión es el benchmark
# aleatorio de 4.3, que muestra que el estimador observacional sobreestima. **El semáforo de la app
# informa estabilidad, no identificación.**

# %% [markdown]
# ## 4.7 Double Machine Learning y efectos heterogéneos (CATE)
#
# **DML (Chernozhukov et al., 2018):** con modelos de ruido no lineales (Random Forest) se predicen
# $\hat m(W, X) = \mathbb{E}[Y \mid W, X]$ y $\hat e(W, X) = P(T=1 \mid W, X)$ con *cross-fitting*
# (los pliegues respetan las búsquedas: `groups = srch_id`). Después se regresa el residuo de
# $Y$ sobre el residuo de $T$: la ortogonalidad de Neyman hace que los errores de los modelos de
# ruido afecten el estimador de $\tau$ solo en segundo orden.
#
# * `LinearDML`: $\tau(x) = \theta_0 + x^\top\theta$ con IC; da el ATE y la heterogeneidad lineal.
# * `CausalForestDML`: $\tau(x)$ no paramétrico, para las curvas de CATE.
#
# $X$ (modificadores del efecto) = estrellas, precio relativo, ubicación y usuario recurrente; $W$
# = el resto de $Z$. Por cómputo, se usan 6.000 búsquedas de cada estrato (búsquedas completas,
# semilla fija). Escala absoluta: el nivel está afectado por la selección; la heterogeneidad
# **relativa** entre perfiles es lo interpretable.

# %%
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import RandomForestRegressor

X_CATE = ["prop_starrating", "log_precio_rel", "prop_location_score2", "usuario_recurrente"]
W_CATE = [c for c in Z_TODAS if c not in X_CATE]
dml = {}
t0 = time.time()
for estrato, d in [("R = 0", R0), ("R = 1", R1)]:
    s = submuestra_busquedas(d, 6000 / d.srch_id.nunique())
    rf_y = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=50, max_depth=12, n_jobs=-1, random_state=SEED
    )
    rf_t = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=50, max_depth=12, n_jobs=-1, random_state=SEED
    )
    m = LinearDML(model_y=rf_y, model_t=rf_t, discrete_treatment=True, cv=5, random_state=SEED)
    m.fit(s["booking_bool"], s["top5"], X=s[X_CATE], W=s[W_CATE], groups=s["srch_id"])
    ate_inf = m.ate_inference(X=s[X_CATE])
    dml[estrato] = {
        "modelo": m,
        "datos": s,
        "ATE": ate_inf.mean_point,
        "IC95": ate_inf.conf_int_mean(),
        "tasa base": s.loc[s.top5 == 0, "booking_bool"].mean(),
    }
    print(
        f"LinearDML {estrato}: ATE = {ate_inf.mean_point:.4f}, IC 95 % [{ate_inf.conf_int_mean()[0]:.4f}; {ate_inf.conf_int_mean()[1]:.4f}] | n = {len(s):,} | ATE / tasa base = {ate_inf.mean_point / dml[estrato]['tasa base']:.2f}"
    )
print(f"({time.time() - t0:.0f} s)")
print(dml["R = 0"]["modelo"].summary(feature_names=X_CATE).tables[0])

# %%
t0 = time.time()
s0 = dml["R = 0"]["datos"]
cf = CausalForestDML(
    model_y=RandomForestRegressor(
        n_estimators=100, min_samples_leaf=50, max_depth=12, n_jobs=-1, random_state=SEED
    ),
    model_t=RandomForestClassifier(
        n_estimators=100, min_samples_leaf=50, max_depth=12, n_jobs=-1, random_state=SEED
    ),
    discrete_treatment=True,
    n_estimators=400,
    min_samples_leaf=100,
    cv=5,
    random_state=SEED,
    n_jobs=-1,
)
cf.fit(s0["booking_bool"], s0["top5"], X=s0[X_CATE], W=s0[W_CATE], groups=s0["srch_id"])
print(f"CausalForestDML (R = 0): {time.time() - t0:.0f} s")


def curva_cate(modelo, base: pd.DataFrame, variable: str, grilla: np.ndarray) -> pd.DataFrame:
    """CATE promedio (y su IC) moviendo `variable` en la grilla y dejando las otras X como en la muestra."""
    filas = []
    for v in grilla:
        Xv = base[X_CATE].copy()
        Xv[variable] = v
        inf_v = modelo.effect_inference(Xv.sample(min(len(Xv), 3000), random_state=SEED))
        filas.append(
            {
                "valor": v,
                "CATE": inf_v.pred.mean(),
                "IC_inf": np.percentile(inf_v.conf_int()[0], 50),
                "IC_sup": np.percentile(inf_v.conf_int()[1], 50),
            }
        )
    return pd.DataFrame(filas)


curvas = {
    "prop_starrating": curva_cate(cf, s0, "prop_starrating", np.array([1, 2, 3, 4, 5])),
    "log_precio_rel": curva_cate(cf, s0, "log_precio_rel", np.linspace(-0.8, 0.8, 9)),
    "prop_location_score2": curva_cate(
        cf,
        s0,
        "prop_location_score2",
        np.quantile(s0["prop_location_score2"], [0.05, 0.25, 0.5, 0.75, 0.95]),
    ),
    "usuario_recurrente": curva_cate(cf, s0, "usuario_recurrente", np.array([0, 1])),
}
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
titulos = {
    "prop_starrating": "Estrellas",
    "log_precio_rel": "log(precio / mediana búsqueda)",
    "prop_location_score2": "location_score2",
    "usuario_recurrente": "Usuario recurrente (0/1)",
}
for a, (v, c) in zip(ax, curvas.items(), strict=True):
    a.plot(c["valor"], c["CATE"], "o-", color=gr.AZUL, ms=5)
    a.fill_between(c["valor"], c["IC_inf"], c["IC_sup"], color=gr.AZUL, alpha=0.2, lw=0)
    a.axhline(dml["R = 0"]["ATE"], color=gr.NARANJA, ls="--", lw=1)
    a.set(xlabel=titulos[v], title=f"CATE según {titulos[v].split(' (')[0]}")
ax[0].set_ylabel("τ(x): Δ P(reserva) por top-5")
ax[0].annotate(
    "ATE (LinearDML)",
    xy=(1, dml["R = 0"]["ATE"]),
    xytext=(2, 6),
    textcoords="offset points",
    fontsize=8,
    color=gr.TEXTO_2,
)
fig.tight_layout()
gr.guardar(fig, "f4_cate")
plt.show()
pd.concat(curvas, names=["variable"]).round(4)

# %% [markdown]
# **DML.**
# - Con el orden del algoritmo, `LinearDML` estima un ATE de 0,0833 (IC 95 % [0,0788; 0,0877]),
#   consistente con la regresión backdoor (0,0961) y el IPW (0,0890).
# - Con orden aleatorio, 0,0073 [0,0059; 0,0088].
# - Relativos a sus tasas base (0,0202 y 0,0037), los cocientes son 4,1 y 2,0: el mismo patrón que
#   el logit condicional. El observacional casi duplica la amplificación relativa del benchmark.
#
# **CATE (Causal Forest DML, orden del algoritmo).** El efecto del top-5 es heterogéneo:
# - **Mayor para hoteles baratos relativos a su búsqueda:** CATE ≈ 0,10-0,12 con precio relativo
#   bajo, contra ≈ 0,05-0,06 con precio relativo alto.
# - **Mayor para hoteles mejor ubicados:** de 0,076 a 0,111 entre los percentiles 5 y 95 de
#   `location_score2`.
# - Casi plano en estrellas (0,082-0,088) y en usuario recurrente (0,083 vs. 0,085).
#
# **Negocio:** el slot rinde más cuando muestra una oferta atractiva (buen precio relativo, buena
# ubicación), no cuando muestra el hotel de mayor categoría. Las magnitudes absolutas heredan la
# selección del dataset y el sesgo por U de R = 0; lo interpretable es el **orden relativo** de los
# perfiles.

# %% [markdown]
# ## 4.8 Ensayo crítico: el choque entre capacidad predictiva y palanca causal
#
# **¿Si la posición es la variable más importante del Random Forest, puede la dirección usarla como
# palanca?** En la Fase 3, `position` es la variable con mayor importancia por permutación en Test (la
# PR-AUC cae 0,103 al permutarla), y la regla raíz del árbol podado es `position ≤ 4,5`. La tentación
# es: "el top-5 multiplica la conversión por 6 (5,38 en OR intra-búsqueda); subamos a los hoteles que
# queremos vender".
#
# La Fase 4 muestra por qué es un error.
# 1. **Gran parte de esa capacidad predictiva es selección, no causa.** El algoritmo pone arriba a los
#    hoteles que se iban a reservar igual (por calidad, precio, competidores y, sobre todo, por un
#    desempeño histórico que no vemos). Para *predecir*, la posición es útil justamente porque resume
#    todo eso. Para *intervenir*, solo cuenta el efecto que queda cuando esa selección se rompe: el
#    orden aleatorio da un OR de 2,11, menos de la mitad que la asociación ingenua en log-odds.
# 2. **La palanca existe, pero es más chica y heterogénea.** Mover un hotel al top-5 sí aumenta sus
#    reservas (efecto causal positivo, robusto a las refutaciones), sobre todo si es barato para su
#    búsqueda y está bien ubicado. Subir un hotel caro y mal ubicado rinde la mitad.
# 3. **Cuándo una variable muy predictiva no tiene poder causal, o tiene el signo opuesto:**
#    - cuando es un **proxy de confusores** (la posición resume U);
#    - cuando es **consecuencia del resultado o un colisionador** (el "destacado" C, o el clic si se
#      condiciona en él);
#    - cuando hay **selección de la muestra** sobre el resultado (S).
#
#    En nuestro DAG, condicionar en C hace que la exposición parezca dañina para ventas de hoteles de
#    calidad. Y un modelo que "controla" por el clic concluye que la posición no importa: el
#    **mismo dato da tres respuestas distintas según qué se condiciona**.
# 4. **Otra cara del choque:** `random_bool` es la segunda variable más importante del Random Forest
#    (ΔPR-AUC 0,079). No es una palanca de negocio: refleja que el proveedor seleccionó distinto las
#    búsquedas de cada estrato. Es importancia predictiva generada por el diseño muestral.
#
# **Conclusión del ensayo:** la importancia predictiva responde "¿qué variables ayudan a adivinar Y?";
# la palanca causal responde "¿qué pasa con Y si muevo X?". Solo el DAG, la identificación y, cuando
# existe, un experimento, permiten pasar de una a la otra.

# %% [markdown]
# ## 4.9 Traducción a decisiones de negocio
#
# 1. **Valuación del slot premium.**
#    - Con orden aleatorio, la tasa de reserva por hotel mostrado es 1,42 % en el top-5 contra 0,37 %
#      fuera de él (salida de abajo).
#    - En la población de búsquedas con al menos un clic, cada exposición en el top-5 agrega ≈ 1,05
#      puntos porcentuales de probabilidad de reserva.
#    - Con el gasto mediano por reserva (salida de abajo), cada 1.000 exposiciones en el top-5 agregan
#      ≈ 10,5 reservas.
#    - Es una cifra ilustrativa: la escala absoluta depende de la selección del dataset.
# 2. **No pagar el sesgo.** Valuar el slot con datos del algoritmo, aun ajustados, sobreestima su
#    efecto relativo en un 60 % (OR 3,38 vs. 2,11). Un contrato de posicionamiento pago calculado así
#    sobreprecia el inventario.
# 3. **Asignación.** El slot rinde más con hoteles baratos para su búsqueda y bien ubicados (CATE ≈ el
#    doble que con hoteles caros). Si el objetivo es maximizar reservas incrementales, se prioriza la
#    exposición de esos perfiles.
# 4. **Experimentar en forma continua.** El orden aleatorio es la única fuente de verdad que permitió
#    detectar el sesgo. Mantener una fracción chica de tráfico con orden aleatorio funciona como
#    auditoría permanente del valor causal del ranking.
# 5. **Límites de la recomendación:**
#    - Vale para búsquedas con al menos un clic.
#    - Con slots de suma cero, subir a un hotel desplaza a otro (SUTVA): el efecto neto sobre la
#      plataforma es menor que la suma de efectos individuales.

# %%
gasto_medio = dc.loc[
    (dc.booking_bool == 1) & (dc.gross_bookings_usd > 0) & (dc.precio_outlier == 0),
    "gross_bookings_usd",
]
print(
    f"Gasto por reserva (todas las reservas sin outliers): media = {gasto_medio.mean():.2f} USD, mediana = {gasto_medio.median():.2f} USD, n = {len(gasto_medio)}"
)
d1 = R1.groupby("top5")["booking_bool"].mean()
print(f"R = 1: P(reserva | top-5) = {d1[1]:.4f}, P(reserva | resto) = {d1[0]:.4f}")

# %% [markdown]
# ## 4.10 Síntesis de la Fase 4
#
# | Pregunta | Respuesta | Evidencia |
# |---|---|---|
# | ¿El top-5 causa reservas? | Sí: OR intra-búsqueda 2,11 [1,90; 2,35] con orden aleatorio | 4.3, benchmark |
# | ¿Cuánto sesgo tiene el observacional? | Ingenuo 5,38; ajustado 3,38: sobreestima 60 % aun ajustado | 4.3; Backdoor con $Z$ rechazado |
# | ¿El efecto pasa por el clic? | Sí, ≈ 95 % | 4.4 |
# | ¿Qué pasa si se condiciona en colisionadores? | La diferencia absoluta se infla (S); aparece Berkson (C) | 4.5 y simulaciones con verdad conocida |
# | ¿Es robusta la estimación? | Estable en las 3 refutaciones; sensible a un U moderado | 4.6 |
# | ¿Para quién rinde más? | Hoteles baratos para su búsqueda y bien ubicados | 4.7, CATE |
# | ¿La variable más predictiva es una palanca? | Solo en parte: buena parte de su poder predictivo es selección | 4.8 |

# %% [markdown]
# ---
# # Fase 5 — Artefactos para la aplicación desplegada
#
# La app (`app/app.py`) **no ve los datos**: lee solo los artefactos que genera esta sección en
# `models/`. Reglas de confidencialidad (`tpi.artefactos`; decisión de Verónica del 12/11 de
# publicar solo agregados):
#
# * Conteos con supresión de celdas con menos de 10 filas; sin identificadores ni fechas.
# * El modelo de la app es un **Random Forest compacto** (100 árboles, profundidad ≤ 12, hojas de
#   ≥ 100 filas): cada predicción promedia al menos 100 observaciones de entrenamiento. Su
#   desempeño en Test se compara con el del Random Forest completo.
# * La heterogeneidad causal se exporta como una grilla de CATE precalculada (la app no corre
#   EconML).

# %%
import json

import joblib

from tpi.artefactos import K_MIN, MIN_HOJA, claves_prohibidas, grilla_2d, guardar_json, histograma
from tpi.config import MODELS_DIR

t0 = time.time()
rf_app = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=MIN_HOJA,
    max_features=0.3,
    random_state=SEED,
    n_jobs=-1,
).fit(X_tr, y_tr)
p_app = rf_app.predict_proba(X_te)[:, 1]
ruta_modelo = MODELS_DIR / "rf_app.joblib"
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(rf_app, ruta_modelo, compress=3)
print(
    f"RF de la app: PR-AUC Test = {average_precision_score(y_te, p_app):.4f} (RF completo: {tabla_test.loc['Random Forest', 'PR_AUC']:.4f}), "
    f"ROC-AUC = {roc_auc_score(y_te, p_app):.4f}, Brier = {brier_score_loss(y_te, p_app):.5f} | "
    f"{ruta_modelo.stat().st_size / 1e6:.1f} MB | {time.time() - t0:.0f} s"
)

# %%
VARS_APP = {
    "prop_starrating": ("Estrellas", np.arange(0.5, 6, 1)),
    "prop_review_score": ("Puntaje de reseñas", np.arange(0.75, 5.5, 0.5)),
    "log_precio_rel": ("log(precio / mediana de la búsqueda)", np.linspace(-1.2, 1.2, 25)),
    "log_precio": ("log(precio por noche, USD)", np.linspace(2, 7.6, 29)),
    "prop_location_score2": ("Puntaje de ubicación 2", np.linspace(0, 1, 21)),
    "srch_booking_window": ("Anticipación de la búsqueda (días)", np.arange(0, 370, 15)),
    "srch_length_of_stay": ("Noches", np.arange(0.5, 15, 1)),
}
art = {
    "meta": {
        "fuente": "Kaggle, Personalize Expedia Hotel Searches - ICDM 2013 (muestra de 30.000 búsquedas)",
        "filas": len(df),
        "busquedas": int(df.srch_id.nunique()),
        "prevalencia_reserva": float(df.booking_bool.mean()),
        "k_min_supresion": K_MIN,
        "min_filas_por_hoja": MIN_HOJA,
        "semilla": SEED,
    },
    "variables": {},
    "ctr_posicion": {},
    "tasa_reserva_por_estrellas": {},
}
for v, (etq, bordes) in VARS_APP.items():
    art["variables"][v] = {
        "etiqueta": etq,
        **{
            f"R{r}": {
                f"reserva_{b}": histograma(
                    df.loc[(df.random_bool == r) & (df.booking_bool == b), v], bordes
                )
                for b in (0, 1)
            }
            for r in (0, 1)
        },
    }
for r in (0, 1):
    c = (
        df[(df.random_bool == r) & (df.position <= 38)]
        .groupby("position")
        .agg(
            n=("click_bool", "size"), clics=("click_bool", "sum"), reservas=("booking_bool", "sum")
        )
    )
    c = c[c.n >= K_MIN]
    art["ctr_posicion"][f"R{r}"] = {
        "posicion": c.index.tolist(),
        "n": c.n.tolist(),
        "ctr": (c.clics / c.n).tolist(),
        "tasa_reserva": (c.reservas / c.n).tolist(),
    }
    e = (
        df[df.random_bool == r]
        .groupby("prop_starrating")
        .agg(n=("booking_bool", "size"), reservas=("booking_bool", "sum"))
    )
    e = e[e.n >= K_MIN]
    art["tasa_reserva_por_estrellas"][f"R{r}"] = {
        "estrellas": e.index.tolist(),
        "n": e.n.tolist(),
        "tasa": (e.reservas / e.n).tolist(),
    }

esc, pca_ = pca_pipe.named_steps["escala"], pca_pipe.named_steps["pca"]
Zt = pca_pipe.transform(te[X_PCA].astype(float))
bx, by = np.linspace(-6, 8, 57), np.linspace(-5, 7, 49)
art["pca"] = {
    "variables": X_PCA,
    "media": esc.mean_,
    "desvio": esc.scale_,
    "componentes": pca_.components_[:4],
    "autovalores": pca_.explained_variance_,
    "pct_var": 100 * pca_.explained_variance_ratio_,
    "grilla_test": {
        f"reserva_{b}": grilla_2d(
            Zt[te.booking_bool.to_numpy() == b, 0], Zt[te.booking_bool.to_numpy() == b, 1], bx, by
        )
        for b in (0, 1)
    },
}
art["modelo"] = {
    "archivo": "rf_app.joblib",
    "features": FEATURES,
    "medianas_train": X_tr.median().to_dict(),
    "p01": X_tr.quantile(0.01).to_dict(),
    "p99": X_tr.quantile(0.99).to_dict(),
    "metricas_test_app": metricas(
        y_te,
        p_app,
        umbral_f1(
            y_sub,
            cross_val_predict(
                clone(rf_app), X_sub, y_sub, groups=g_sub, cv=CV, method="predict_proba", n_jobs=-1
            )[:, 1],
        ),
    ),
    "metricas_test_torneo": tabla_test.drop(columns=["seg. entrenamiento"]).to_dict(orient="index"),
    "importancia_permutacion": imp["Permutación (ΔPR-AUC)"].sort_values(ascending=False).to_dict(),
    "calibracion_app": dict(
        zip(
            ["frec_obs", "prob_media"],
            calibration_curve(y_te, p_app, n_bins=10, strategy="quantile"),
            strict=True,
        )
    ),
}

# %%
grilla_cate = []
base_x = s0[X_CATE].median()
for est_ in [1, 2, 3, 4, 5]:
    for pr in [-0.6, -0.3, 0.0, 0.3, 0.6]:
        for rec_ in [0, 1]:
            x = pd.DataFrame(
                [
                    {
                        **base_x.to_dict(),
                        "prop_starrating": est_,
                        "log_precio_rel": pr,
                        "usuario_recurrente": rec_,
                    }
                ]
            )[X_CATE]
            inf_x = cf.effect_inference(x)
            lo, hi = inf_x.conf_int()
            grilla_cate.append(
                {
                    "estrellas": est_,
                    "log_precio_rel": pr,
                    "usuario_recurrente": rec_,
                    "cate": float(inf_x.pred[0]),
                    "ic_inf": float(lo[0]),
                    "ic_sup": float(hi[0]),
                }
            )
art["causal"] = {
    "dag_dot": a_dot(g_alg),
    "dag_dot_aleatorio": a_dot(g_rand),
    "ate_dowhy": tabla_ate.reset_index(names="estrato").to_dict(orient="records"),
    "or_intra": [
        {"estrato": e, "ajuste": a, **fila}
        for (e, a), fila in tabla_rel.to_dict(orient="index").items()
    ],
    "or_benchmark": or_bench,
    "or_observacional_ajustado": or_adj0,
    "or_ingenuo": or_ing0,
    "dml": {
        k: {"ATE": v["ATE"], "IC95": v["IC95"], "tasa_base": v["tasa base"]} for k, v in dml.items()
    },
    "cate_grilla": grilla_cate,
    "cate_curvas": {k: c.to_dict(orient="list") for k, c in curvas.items()},
    "refutaciones": tabla_refut.reset_index(names="prueba").to_dict(orient="records"),
    "sensibilidad_u": tabla_u.to_dict(orient="records"),
    "mediador": tabla_med.reset_index(names="especificacion").to_dict(orient="records"),
}
ruta_art = guardar_json(art, MODELS_DIR / "artefactos_app.json")
prohibidas = claves_prohibidas(json.loads(ruta_art.read_text("utf-8")))
assert not prohibidas, prohibidas
print(
    f"{ruta_art.name}: {ruta_art.stat().st_size / 1e3:.0f} kB | identificadores encontrados: {len(prohibidas)}"
)

# %% [markdown]
# **Resultados para el informe y las diapositivas.** Todos los números que citan el informe Quarto y
# las diapositivas se leen de este archivo (`models/resultados.json`), que solo contiene estadísticos
# agregados. Así ningún número se copia a mano.

# %%
from dataclasses import asdict

from tpi.inferencia import Contraste


def _c(x: Contraste) -> dict:
    return {k: v for k, v in asdict(x).items() if k != "extra"} | {
        "extra": x.extra,
        "texto": x.texto(),
    }


resultados = {
    "fase0": {
        "filas_train_csv": 9_917_530,
        "busquedas_train_csv": 399_344,
        "muestra_filas": len(crudo),
        "muestra_busquedas": int(crudo.srch_id.nunique()),
    },
    "fase1": {
        "contraste1": _c(c1)
        | {
            "ctr_top5": float(por_busq[True].mean()),
            "ctr_resto": float(por_busq[False].mean()),
            "mde_dz": efecto_minimo_detectable(len(D)),
        },
        "contraste2": _c(c2)
        | {
            "cociente_geom": float(np.exp(c2.diferencia)),
            "ic_cociente": [float(np.exp(c2.ic95[0])), float(np.exp(c2.ic95[1]))],
            "mde_g": efecto_minimo_detectable(*c2.n),
            "n_rec": c2.n[0],
            "n_nue": c2.n[1],
        },
        "contraste2_mw": _c(c2b),
        "anova": {k: _c(v) for k, v in an.items()} | {"omega2": an["clasico"].extra["omega2"]},
        "posthoc_cociente_5_vs_12": float(
            1 / ph.loc[(ph.A == "1-2") & (ph.B == "5"), "cociente_precios"].iloc[0]
        ),
        "centinelas": tabla_centinelas.to_dict(orient="index"),
        "escala_precio_pct": tabla_escala["pct"].to_dict(),
        "outliers_precio": {
            "filas": int(df.precio_outlier.sum()),
            "pct": float(100 * df.precio_outlier.mean()),
            "cerca_inf": lo_abs,
            "cerca_sup": hi_abs,
        },
        "faltantes_pct": faltantes["pct_faltante"].to_dict(),
        "mcar_max_efecto": resumen_mcar["max_|efecto|"].to_dict(),
        "zc": tabla_zc.to_dict(orient="index"),
        "slots_especiales": SLOTS_ESPECIALES,
    },
    "fase2": {
        "ingenuo": {
            "r2": ingenuo.rsquared,
            "curtosis": dg_ing.curtosis,
            "cook_max": float(inf_ing.cook.max()),
        },
        "mco": {
            "n": int(mco.nobs),
            "r2": mco.rsquared,
            "r2_aj": mco.rsquared_adj,
            "F": mco.fvalue,
            "gl": [int(mco.df_model), int(mco.df_resid)],
            "coef": mco.params.to_dict(),
            "p_hc3": mco_hc3.pvalues.to_dict(),
        },
        "gauss_markov": asdict(dg),
        "influencia": {
            "outlier_y": int(inf.outlier_y.sum()),
            "alto_leverage": int(inf.alto_leverage.sum()),
            "influyente_4n": int(inf.influyente_4n.sum()),
            "influyente_05": int(inf.influyente_05.sum()),
            "cook_max": float(inf.cook.max()),
        },
        "vif_max_modelo": float(vif["VIF modelo 2.1"].max()),
        "vif_max_ampliado": float(vif["VIF conjunto ampliado (Fase 3)"].max()),
        "pca": {
            "autovalores": var_pca["autovalor"].tolist(),
            "pct_acum": var_pca["pct_acum"].tolist(),
            "n_kaiser": n_kaiser,
        },
    },
    "fase3": {
        "test": tabla_test.drop(columns=["seg. entrenamiento"]).to_dict(orient="index"),
        "mejor": mejor,
        "rf_params": busqueda_rf.best_params_,
        "logit_params": grid_log.best_params_,
        "arbol_libre": {
            "hojas": int(arbol_libre.get_n_leaves()),
            "pr_auc_train": float(average_precision_score(y_tr, p_libre_tr)),
            "pr_auc_test": float(average_precision_score(y_te, p_libre_te)),
        },
        "arbol_podado": {
            "alfa": float(alfa_opt),
            "hojas": int(arbol.get_n_leaves()),
            "profundidad": int(arbol.get_depth()),
        },
        "oob_pr_auc": float(average_precision_score(y_tr[ok], p_oob[ok])),
        "cronologico": {
            "pr_auc": float(average_precision_score(futuro[OBJETIVO], p_fut)),
            "roc_auc": float(roc_auc_score(futuro[OBJETIVO], p_fut)),
        },
        "importancia": imp[["MDI", "Permutación (ΔPR-AUC)"]].to_dict(orient="index"),
        "rho_cardinalidad_mdi": float(rho_card.statistic),
        "ranking_top5": pd.DataFrame(filas_rank).set_index("estrato").to_dict(orient="index"),
        "pca_features_pr_auc": float(average_precision_score(y_te, p_pca)),
    },
    "fase4": {
        "ate_dowhy": tabla_ate.to_dict(orient="index"),
        "or_intra": {
            f"{e} | {a}": fila for (e, a), fila in tabla_rel.to_dict(orient="index").items()
        },
        "or_ingenuo_R0": or_ing0,
        "or_ajustado_R0": or_adj0,
        "or_benchmark": or_bench,
        "brecha_logor": float(brecha),
        "pct_explicado_Z": float(100 * np.log(or_ing0 / or_adj0) / brecha),
        "sens_top4": {"or": sens_t.estimacion, "ic": list(sens_t.ic95)},
        "mediador": tabla_med.to_dict(orient="index"),
        "seleccion_sim": tabla_s.to_dict(orient="index"),
        "colisionador_C": tabla_c.to_dict(orient="index"),
        "refutaciones": tabla_refut.to_dict(orient="index"),
        "sens_u": tabla_u.to_dict(orient="records"),
        "dml": {
            k: {"ATE": v["ATE"], "IC95": v["IC95"], "tasa_base": v["tasa base"]}
            for k, v in dml.items()
        },
        "gasto_medio": float(gasto_medio.mean()),
        "gasto_mediana": float(gasto_medio.median()),
        "p_top5_R1": float(d1[1]),
        "p_resto_R1": float(d1[0]),
    },
}
ruta_res = guardar_json(resultados, MODELS_DIR / "resultados.json")
assert not claves_prohibidas(json.loads(ruta_res.read_text("utf-8")))
print(f"{ruta_res.name}: {ruta_res.stat().st_size / 1e3:.0f} kB")
