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
la app recién se construye en la Fase 5; hasta entonces la carga de Antonella es auditar los PRs del
backend. Para evitar que se autoapruebe, en los PRs de la app el veredicto lo emite Lucía como QA
suplente.

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
(ρ̄ = −0,0743) y `prop_location_score2` (ρ̄ = −0,0651). **Decisión:** R = 1 se usa como
cuasi-experimento con ajuste por Z y efectos fijos por búsqueda (detalle en `docs/ia_log.md`,
error 10).

**Restricción legal y plan B.** Las reglas de la competencia declaran confidenciales los datos y
los "resulting model parameters", y el TP exige repo y app públicos. Consultamos al profesor.
Mientras tanto: ningún dato en el repo (lo garantiza un test), crudo fuera de OneDrive, la app
solo muestra agregados, y el plan B (Online Shoppers, UCI #468, CC BY 4.0, con tratamiento
semisintético) queda evaluado y activable con una variable de entorno (`docs/plan_b.md`).

_Firma: pendiente — V. López Marsico (Lead)._

## Entrada 2 — Matriz de decisiones estadísticas y supuestos
*Responsables: Backend/ML + QA.* Pendiente (Fases 1 y 2).

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

**(c) ¿Mitiga la comparación dentro de la búsqueda?** En parte, y por eso la elegimos (decisión
del 12/11):

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

### Amenaza B — Aleatorización imperfecta en R = 1

Ver Entrada 1 y `docs/ia_log.md` (error 10). El benchmark se estima con ajuste por Z y efectos
fijos por búsqueda, y se informa sin ajuste vs. ajustado como diagnóstico.

## Entrada 5 — Trazabilidad y uso ético de asistentes de IA
*Responsable: QA.* Se construye a partir de [`docs/ia_log.md`](docs/ia_log.md), que se actualiza
en cada fase.
