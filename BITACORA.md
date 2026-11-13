# BITÁCORA de Ingeniería — TPI EAyMR 2026

Cuaderno de laboratorio y registro de decisiones. No es un resumen del notebook ni del historial de
commits: registra **qué decidimos, por qué, y qué salió mal**.

## Roles Scrum

| Integrante | Rol | Firma en esta bitácora |
|---|---|---|
| Verónica López Marsico | Project Lead & Scrum Master (Product Owner) | `— V. López Marsico (Lead)` |
| Lucía Báez | Backend & ML / Causal Engineer | `— L. Báez (Backend/ML)` |
| Antonella Robiollio | Frontend & UI/UX Engineer + QA & Validation Engineer | `— A. Robiollio (Frontend/QA)` |

Asignación (Fase 0, 11/11/2026): somos tres para cuatro roles. QA se combina con Frontend porque
la app recién se construye en la Fase 5. La asignación la propuso Claude Code y la aprobó Verónica.

**Reorganización (12/11/2026, decisión de Verónica).** Desde esta fecha Verónica ejecuta y valida
todo el trabajo con asistencia de Claude Code, incluidos los veredictos de QA en los PRs. Los roles
de la tabla se mantienen porque la defensa oral es por rol; Lucía y Antonella no revisan ni validan
PRs. Cada PR registra quién lo revisó realmente, y `docs/ia_log.md` registra quién propuso y quién
aprobó cada decisión.

---

## Entrada 1 — Marco estratégico e hipótesis
*Responsable: Project Lead. Fase 0, 12/11/2026.*

### Problema de negocio

Un buscador de hoteles muestra ~25 resultados por búsqueda, pero la atención del usuario se
concentra en los primeros. Los slots superiores son un recurso escaso de suma cero: subir un hotel
baja a otro. El recomendador decide quién los ocupa y, a la vez, elige hoteles que probablemente
se venderían igual (mejor calidad, mejor ubicación, en promoción). La pregunta operativa es si el
slot **causa** ventas o solo las **predice**: la respuesta define cuánto vale un slot premium
(publicidad, comisiones) y si conviene reasignarlo.

### Preguntas fundacionales

1. ¿Cuál es el efecto causal de que el recomendador exponga un hotel en el top-5 sobre la
   probabilidad de reserva? (estimando: ATE y CATE de $T = 1\{\text{position} \le 5\}$ sobre
   `booking_bool`).
2. ¿Cuánto sesgo introduce estimarlo con el orden del algoritmo (datos observacionales) en lugar
   del orden aleatorio? Es decir, ¿cuánto de la "eficacia" del top-5 es autoselección de buenos
   hoteles?
3. ¿El efecto pasa entero por el clic (mediador) o hay un canal directo?
4. ¿Para qué hoteles y usuarios el slot rinde más (heterogeneidad), es decir, dónde conviene
   asignarlo?

**Preguntas adicionales a la consigna:**

5. ¿Qué tan "aleatorio" es el orden aleatorio? (ya respondida en parte, ver abajo).
6. ¿Cómo condiciona el mecanismo de selección del dataset (búsquedas con clic, sobremuestreo de
   compras) lo que se puede estimar? (Entrada 4).

### Hipótesis de trabajo (a contrastar; ninguna está confirmada)

- **H1.** La exposición en el top-5 aumenta la probabilidad de reserva (efecto positivo con R = 1).
- **H2.** El estimador observacional (R = 0) sobreestima el efecto: el algoritmo pone arriba a
  hoteles que se venderían igual (confusión positiva).
- **H3.** Ajustar por los confusores observados (Z) reduce esa brecha pero no la cierra, por el
  desempeño histórico del hotel que usa el ranker y no observamos (U).
- **H4.** El efecto es heterogéneo: mayor para hoteles con menos señales de calidad visibles, en
  los que la posición sustituye información.

### Mapa conceptual

```
contexto de búsqueda (Zb) ─┐          ┌──────────── intención de compra (I, latente)
historial del usuario (Zu) ├─► RANKER ─► posición/top-5 (T) ─► clic (M) ─► reserva (Y) ─► gasto
calidad y precio (Zh)     ─┤    ▲                                ▲            ▲
popularidad previa (Zp)   ─┘    └─ desempeño histórico (U, latente) ─┴────────────┘
random_bool (R) = 1: el ranker se reemplaza por un orden (casi) aleatorio
```

### Selección fundamentada del dataset

**Elegido:** Personalize Expedia Hotel Searches — ICDM 2013 (Kaggle). Motivos:

- Datos reales de producción con la fricción que pide el enunciado: faltantes no aleatorios
  (historial del usuario, scores), centinelas, colas pesadas en precio y confusión no observada.
- Escala: `train.csv` tiene 9.917.530 filas búsqueda × hotel y 399.344 búsquedas (cifras
  verificadas contra Liu et al., 2013), con más de 40 variables explicativas continuas,
  discretas y categóricas. Trabajamos con una muestra de 30.000 búsquedas completas (739.323 filas).
- **Experimento natural**: según la documentación de la competencia, `random_bool` vale "+1 when
  the displayed sort was random, 0 when the normal sort order was displayed" (fuente primaria,
  pestaña Data). Permite contrastar empíricamente el supuesto Backdoor: algo que casi ningún
  dataset observacional ofrece.

**Verificación de la premisa experimental.** La definición se cumple en el mecanismo, pero no en
forma pura: con R = 1 la asociación intra-búsqueda posición–atributo cae entre 71 % y 87 %
respecto del algoritmo, aunque sigue siendo distinta de cero para `prop_starrating`
(ρ̄ = −0,0743) y `prop_location_score2` (ρ̄ = −0,0651). **Decisión** (propuesta por Claude Code, aprobada por Verónica el 12/11):
R = 1 se usa como cuasi-experimento con ajuste por Z y efectos fijos por búsqueda (detalle en
`docs/ia_log.md`, error 10).

**Riesgo de licencia y decisión.** Las reglas de la competencia declaran confidenciales "the data
set and resulting model parameters" y prohíben compartirlos con no participantes, mientras que el
TP exige repo y app públicos y una defensa ante el docente. **Decisión de Verónica (12/11/2026):
no consultar a la cátedra y seguir con Expedia**, con estas mitigaciones ya implementadas:

- Ningún archivo de datos en el repo, ni crudo ni muestra; solo el script de descarga y un
  manifiesto con hashes. Lo garantiza `tests/test_confidencialidad.py`.
- Crudo fuera del repo y de OneDrive (`~/data/tpi-eaymr/raw`, configurable con `TPI_RAW_DIR`).
- El notebook y la app muestran solo agregados, distribuciones y resultados, nunca registros
  individuales.
- El plan B (Online Shoppers, UCI #468, CC BY 4.0, tratamiento semisintético) queda evaluado y
  activable con una variable de entorno (`docs/plan_b.md`), pero **no se activa**.

**Riesgo residual que asumimos:** los resultados publicados (ATE, CATE, coeficientes) pueden
considerarse "resulting model parameters" en sentido estricto.

## Entrada 2 — Matriz de decisiones estadísticas y supuestos
*Roles responsables: Backend/ML + QA. Fase 1 (12/11/2026): análisis ejecutado por Claude Code con
el diseño aprobado por Verónica; revisión a cargo de Verónica. La Fase 2 se agrega al cerrarla.*

### Decisiones de diseño

El precio relativo a la búsqueda, las dos reglas de outliers y la estrategia de imputación (MAR por
grupo con indicador; MNAR estructural sin imputar) fueron **propuestos por Claude Code y aprobados
por Verónica el 12/11**.

| Decisión | Alternativa descartada | Justificación |
|---|---|---|
| Contrastes sobre **unidades independientes** (búsquedas o reservas) | Filas búsqueda × hotel | Las filas de una lista comparten usuario y contexto; tratarlas como independientes subestima los errores estándar |
| α = 0,05 fijado a priori + **efecto mínimo detectable** con 1 − β = 0,80 | Solo p-valor | Con n en miles todo es "significativo"; el MDE dice qué no puede ver la muestra |
| **Precio relativo** a la mediana de la búsqueda (winsorizado 1-99 %) | `price_usd` crudo | Desplazamiento de escala: 3,2 % de las reservas de ≥ 2 noches tienen precio por estadía, disperso entre sitios (1,9-7,5 %), así que no se puede corregir por sitio |
| Outliers de precio con **dos reglas**: cercas extremas de Tukey sobre log (8,41-1951,14 USD) **o** > 10× / < 1/10 de la mediana de la búsqueda | Solo regla relativa | La relativa sola no detectó una búsqueda entera con precios de 4,1 millones (su mediana también era absurda) |
| Centinelas (0) → NaN + indicador | Dejarlos como valores | El 0 de `prop_log_historical_price` (14,8 % de filas) sesga la media −14,8 % e infla el desvío de 0,525 a 1,860 |

### Supuestos contrastados, violaciones y remediación

| Análisis | Supuesto | Prueba y resultado | Remediación |
|---|---|---|---|
| Faltantes | MCAR | Indicador de faltante vs. variables observadas: MCAR se rechaza en las 5 variables (p ajustado por Bonferroni < 0,05); dependencia apreciable para `location_score2` (SMD −0,506), `orig_destination_distance` (0,304), `review_score` (−0,238), `query_affinity` (0,180) | MAR → mediana por grupo + indicador; MNAR estructural (historial del usuario, afinidad) → **no se imputa**, se usa un indicador |
| Contraste 1 (pareado) | Normalidad de $D_s$ | Asimetría −0,540, curtosis en exceso 3,608; n = 14.983 | TCL + Wilcoxon de resguardo (coincide: p = 7,7·10⁻¹³¹) |
| Contraste 2 (dos grupos) | Varianzas iguales | Levene F(1; 16062) = 3,19, p = 0,074 (no se rechaza H0) | Igual se usa **Welch**: grupos muy desbalanceados (1.025 vs. 15.039), donde un error de tipo II de Levene sería costoso; Mann-Whitney de resguardo |
| Contraste 2 | Normalidad | Variable en log: asimetrías 0,076 y 0,507 | Log del gasto (cola pesada a derecha en USD) |
| ANOVA | Normalidad por grupo | Shapiro rechaza en los 4 grupos (es esperable con n grande), pero asimetrías en [−0,213; 0,314] y curtosis en exceso en [0,49; 1,61]; Q-Q con colas algo pesadas | Normalidad aproximada aceptable por TCL (n ≥ 1.252 por grupo); Kruskal-Wallis de resguardo |
| ANOVA | Homocedasticidad | **Levene rechaza**: F(3; 16060) = 56,57, p = 2,27·10⁻³⁶; razón de varianzas 1,59 | **ANOVA de Welch** como principal; post-hoc **Games-Howell**; Tukey y Bonferroni como comparación |

**Por qué Welch y no Kruskal-Wallis en el ANOVA.** La pregunta es sobre el valor esperado (la
media) del log precio, y Welch la responde sin suponer varianzas iguales. Kruskal-Wallis contrasta
dominancia estocástica, que es otra hipótesis. Ambos coinciden (Welch F(3; 4432,0) = 1807;
H(3) = 4150; p < 10⁻³⁰⁰ en los dos).

**Por qué no Shapiro como criterio.** Con miles de observaciones, Shapiro-Wilk rechaza ante desvíos
irrelevantes para el TCL. Evaluamos la forma con asimetría (criterio |asimetría| ≤ 1), curtosis y
Q-Q.

### Clasificación de las variables de competidores ($Z_c$)

Criterio de Verónica: confusor si el ranking la usa ($Z_c \to T$) **y** afecta la reserva
($Z_c \to Y$); si solo afecta Y, es predictor. A pedido de Verónica se contrastaron por separado
las dos flechas. Aplicación de Claude Code, **pendiente de revisión por Verónica**:

| Variable | $Z_c \to T$: interacción $Z_c \times R$ en el logit de T, condicional a $Z_h$ | $Z_c \to Y$ con R = 1 | Clase |
|---|---|---|---|
| `comp_saldo` (+ = Expedia más barata) | Razón de OR 0,842, p = 2,7·10⁻¹⁸; top-5 vs. resto D = +0,049 con R = 0 y −0,006 (p = 0,073) con R = 1 | Reserva OR 1,137 (p = 0,041) | Confusor |
| `comp_n_info` | Razón de OR 1,019, p = 2,8·10⁻⁵ (efecto chico) | Reserva OR 1,118 (p = 2,9·10⁻¹⁷) | Confusor (débil en T) |
| `comp_diff_max` | p = 0,054 (no se rechaza H0) | p = 0,53 | Se excluye |

El DAG se actualizó (arcos $Z_c \to T$ y $Z_c \to M$) y un test de d-separación demuestra que
omitir $Z_c$ rompe el criterio Backdoor.

**Corrección de la fuente.** La primera versión de esta clasificación se hizo sin conocer el signo
de `compN_rate`, que el asistente traía de memoria. Verónica transcribió la definición de Kaggle:
+1 = Expedia más barata. La clasificación no cambia (no dependía del signo); la interpretación
quedó fijada con la fuente.

## Entrada 3 — Autopsia de experimentos fallidos (Data Autopsy)
*Responsable: Backend/ML.* Pendiente (Fases 2 y 3, con al menos dos experimentos fallidos reales).

## Entrada 4 — Formalización del debate causal
*Responsables: Backend/ML + Project Lead.* La justificación de cada arco se completa en la Fase 4
(borrador en `docs/dag_borrador.md`). Adelantamos dos amenazas a la validez detectadas en la Fase 0.

### Amenaza A — El dataset está seleccionado sobre el mediador y, en R = 0, sobre el resultado

**Hechos (sobre las 399.344 búsquedas de `train.csv`, `docs/ia_log.md` error 14):** ninguna
búsqueda tiene cero clics; ninguna tiene más de una reserva; tienen reserva el 93,86 % de las
búsquedas con R = 0 y el 13,04 % con R = 1. La documentación de Kaggle ("most of the data are for
searches that resulted in a purchase") describe solo el estrato del algoritmo y omite el filtro
por clic.

Sea $S_s = 1\{\text{la búsqueda } s \text{ entra al dataset}\}$. Empíricamente,
$S_s = 1 \Rightarrow \sum_i M_{is} \ge 1$ en ambos estratos, y en R = 0 además
$\Pr(\sum_i Y_{is} = 1 \mid S_s = 1) \approx 0{,}94$.

**(a) Es un caso real de la trampa del colisionador.** En el DAG, el clic es colisionador:
$T \to M \leftarrow I$ (la intención latente también causa el clic). $S$ es descendiente de $M$
(y, en R = 0, de $Y$). Condicionar en un descendiente de un colisionador abre parcialmente el
camino $T \to M \leftarrow I \to Y$: entre las búsquedas seleccionadas, un hotel fuera del top-5
que recibió el clic "necesitó" más intención, así que la exposición queda asociada negativamente
con la intención (sesgo de Berkson). Es el **colisionador natural** que buscábamos; el simulado
($T \to C \leftarrow Y$) queda como validación con verdad conocida.

**(b) Tasa base y validez externa.** La selección no es pareja entre estratos, así que la tasa de
reserva por fila difiere 7 veces (0,0375 en R = 0 vs. 0,0054 en R = 1, muestra de 30.000
búsquedas). Consecuencias:

- El ATE en escala absoluta de R = 0 está inflado por un factor desconocido (depende de cuánto se
  sobremuestrearon las compras) y **no es comparable** con el de R = 1.
- La población objetivo es "búsquedas con al menos un clic". El efecto sobre el tráfico total,
  que incluye a los usuarios que no hacen clic, **no está identificado** con estos datos, y así
  hay que decirlo en las recomendaciones de negocio.

**(c) ¿Mitiga la comparación dentro de la búsqueda?** En parte, y por eso la elegimos (propuesta
de Claude Code, aprobada por Verónica el 12/11; el planteo de los puntos a-c es de Verónica):

- *Lo que resuelve.* Como $M_{is} = 1 \Rightarrow S_s = 1$, para cada hotel
  $\Pr(M_{is}=1 \mid T_{is}=t, S_s=1, L_s) = \Pr(M_{is}=1 \mid T_{is}=t, L_s) / \Pr(S_s=1 \mid T_{is}=t, L_s)$,
  donde $L_s$ es la lista. Si el denominador casi no depende de qué hotel ocupa el slot (listas
  largas: ~25 hoteles), el cociente dentro de la búsqueda (risk ratio) es el mismo que en la
  población y la diferencia absoluta se infla por $1/\Pr(S_s=1 \mid L_s)$. Con selección por el
  resultado, el odds ratio es invariante (lógica de caso-control; Prentice y Pyke, 1979). Además,
  los efectos fijos por búsqueda eliminan todo confusor a nivel búsqueda (intención de la sesión,
  destino, fecha, dispositivo).
- *Lo que no resuelve.* (i) Es una aproximación: con slots fijos, mover un hotel al top-5 desplaza
  a otro, y el denominador cambia algo (interferencia, violación de SUTVA). (ii) No recupera el
  efecto sobre búsquedas sin clic. (iii) Con a lo sumo una reserva por búsqueda, el resultado
  dentro de la lista es una **elección** ("qué hotel se lleva la reserva"), no una conversión
  independiente.
- **Verificación prevista (Fase 4):** simular listas con efecto conocido, aplicar el mismo filtro
  $S$ y comprobar que el risk ratio intra-búsqueda se recupera y el ATE absoluto no.

### Amenaza B — Aleatorización imperfecta en R = 1 (limitación declarada)

Con orden aleatorio, la posición sigue asociada a algunos atributos: ρ̄ intra-búsqueda −0,074 con
`prop_starrating` y −0,065 con `prop_location_score2`, frente a −0,256 y −0,299 bajo el algoritmo.
Descartamos dos explicaciones: la selección por compra (la asociación persiste en búsquedas sin
reserva ni clic) y la remoción de filas (los huecos son slots fijos, ver C). **No tenemos una
explicación verificable y no forzamos una hipótesis** (decisión de Verónica, 12/11). Remediación
(propuesta por Claude Code, aprobada por Verónica el 12/11): el benchmark se estima con ajuste por
Z y efectos fijos por búsqueda, y se informa sin ajuste vs. ajustado como diagnóstico.

### Característica del DGP C — Slots fijos fuera del dataset

Las posiciones 5, 11, 17 y 23 (una cada seis) aparecen en 6,2-7,6 % de las búsquedas con R = 0 y
1,4-1,7 % con R = 1, contra ≈ 99 % de sus vecinas. Son slots que Expedia llena con listados que
Kaggle no publica, y explican casi todo el ~13 % de posiciones faltantes por lista. Consecuencias:

- T se define sobre la posición **en pantalla** (top-5, decisión confirmada por Verónica el
  12/11), porque es la exposición real del usuario. En general el top-5 contiene 4 hoteles del
  dataset.
- **Sensibilidad:** `top4_dataset` (los 4 primeros hoteles del dataset en el ranking de la
  búsqueda), pedida por Verónica. Coincide con `top5` en el 99,89 % de las filas; la Fase 4
  reporta ambas.
- Las pocas filas que sí ocupan esos slots tienen tasa de clic atípicamente baja y se marcan con
  `slot_especial`.

Fijado por `test_slots_estructurales_cada_seis_posiciones`.

### Riesgo de licencia (validez externa de la publicación, no del estimador)

Ver Entrada 1. Decisión de Verónica: seguir con Expedia sin consultar a la cátedra, con
mitigaciones verificadas por tests; el plan B queda disponible y no se activa.

## Entrada 5 — Trazabilidad y uso ético de asistentes de IA
*Responsable: QA.* Se construye a partir de [`docs/ia_log.md`](docs/ia_log.md), que se actualiza
en cada fase.
