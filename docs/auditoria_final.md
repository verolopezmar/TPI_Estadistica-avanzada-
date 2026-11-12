# Auto-auditoría final contra el enunciado y la rúbrica (12/11/2026)

> Redactada por Claude Code; revisión de Verónica. Estados: ✅ cumplido con evidencia · ⚠️ cumplido
> con salvedad · ❌ falta. "NB x.y" = sección del notebook `notebooks/TP_Integrador_Final.ipynb`.

## A. Entregables obligatorios (enunciado, sección 5)

| Requisito | Estado | Evidencia |
|---|---|---|
| Repo modular (`/src`, `/data` o script de descarga, `/notebooks`, `/app`, `/tests`) | ✅ | Raíz del repo; `data/download_data.py` |
| README editorial con badges, diagrama, reproducción paso a paso y link al deploy | ⚠️ | `README.md`; **falta la URL del deploy** (la carga Verónica después del deploy) |
| Dependencias fijadas | ✅ | `requirements.txt` (con motivos de cada pin), `app/requirements.txt` |
| Notebook por fases, narrativa previa a cada bloque, Restart & Run All | ✅ | `notebooks/TP_Integrador_Final.ipynb`, ejecutado con `jupyter nbconvert --execute` (≈ 18 min) |
| BITÁCORA con 5 entradas **firmadas** | ⚠️ | `BITACORA.md`: las 5 entradas están completas; **faltan las firmas, que pone Verónica** |
| App desplegada con URL funcional | ⚠️ | `app/` corre local y pasa `AppTest`; **el deploy lo hace Verónica** (README, "Deploy") |
| Slides PDF de 10 a 12 diapositivas | ✅ | `report/slides.pdf` (beamer, generado de `report/slides.qmd`) |
| Roles Scrum en el encabezado del repo | ✅ | `README.md` (tabla y organización real del trabajo) |

## B. Fases (enunciado, sección 4)

| Fase / ítem | Estado | Evidencia |
|---|---|---|
| F1: EDA (KDE, boxplots, Spearman y Pearson) | ✅ | NB 1.2 y 1.5; `report/figures/f1_*.png` |
| F1: MCAR/MAR/MNAR con justificación de la imputación | ✅ | NB 1.3; `tpi.auditoria.contraste_mcar` |
| F1: ≥ 2 contrastes con estadístico, α a priori, p exacto, IC 95 % y tamaño de efecto | ✅ | NB 1.6 y 1.7; `tpi.inferencia` |
| F1: ANOVA, Shapiro vs. asimetría y curtosis, Levene, Welch/Kruskal, post-hoc Tukey vs. Bonferroni | ✅ | NB 1.8 (además Games-Howell) |
| F2: MCO con R², R² ajustado, F y t | ✅ | NB 2.1 |
| F2: BP/White, JB + Q-Q, Durbin-Watson | ✅ | NB 2.2 |
| F2: residuo estudentizado, leverage, Cook, Huber/RANSAC | ✅ | NB 2.3 |
| F2: VIF con estrategia | ✅ | NB 2.4 (no hay VIF > 5: se justifica no actuar) |
| F2: PCA solo en Train, scree, Kaiser, cargas | ✅ | NB 2.5; `test_pca_se_ajusta_solo_con_train` |
| F3: 80/20 + K-fold 5 estratificado | ✅ | `tpi.particion`, `tpi.modelos.cv_agrupado` (`StratifiedGroupKFold`) |
| F3: logística regularizada con OR e IC 95 % | ✅ | NB 3.1 |
| F3: Naive Bayes y violación de la independencia condicional, calibración vs. logística | ✅ | NB 3.2 y 3.5 |
| F3: CART sin restricciones vs. podado (`cost_complexity_pruning_path`, CV), regla raíz | ✅ | NB 3.3 |
| F3: RF con búsqueda sobre los 5 hiperparámetros, OOB vs. CV | ✅ | NB 3.4 (con curva de convergencia) |
| F3: métricas de test, ROC, PR, Brier, reliability diagrams | ✅ | NB 3.5 |
| F3: MDI vs. permutación con el dilema explicado | ✅ | NB 3.6 |
| F4: DAG (networkx/DOT) con T, Y, Z, M y C | ✅ | NB 4.1; `tpi.dag`; `docs/figuras/dag_borrador.png` |
| F4: demostración Backdoor y trampas del mediador y del colisionador | ✅ | NB 4.1, 4.4 y 4.5; `tests/test_dag.py` |
| F4: DoWhy en 4 pasos con regresión e IPW, comparación con Y ~ T | ✅ | NB 4.2 |
| F4: DML (`LinearDML`/`CausalForestDML`) con RF y curvas de CATE | ✅ | NB 4.7 |
| F4: 3 refutaciones | ✅ | NB 4.6 (más sensibilidad a un U simulado) |
| F4: ensayo crítico | ✅ | NB 4.8 |
| F5: 4 roles, Quality Gate, BITÁCORA | ✅ | README, `.github/`, `BITACORA.md` |
| F5: app con 3 módulos (Diagnóstico + PCA, What-Is con probabilidad e importancia, What-If con DAG, ATE, CATE y semáforo) | ✅ | `app/app.py`; `tests/test_app.py` |
| F5: bonus copiloto conversacional | ❌ | No implementado (es opcional); priorizamos Causal y ML por puntaje |

## C. Rúbrica (nivel "Excelente")

| Criterio | Evaluación | Evidencia / qué falta |
|---|---|---|
| 1.1 Diagnóstico del problema | ✅ | Entrada 1 (problema de suma cero de los slots, preguntas propias 5 y 6) |
| 1.2 Estructura relacional | ✅ | DAG, clasificación de Zc con contraste de Z → T, Spearman vs. Pearson |
| 1.3 Auditoría forense (DGP) | ✅ | Centinelas, escala del precio, slots fijos, selección del dataset, `compN_inv = −1`, distorsión antes y después |
| 1.4 Diseño muestral | ✅ | Unidad = búsqueda, estratificación, α a priori, MDE/potencia, partición agrupada justificada |
| 2.1 Abstracción antes del código | ✅ | DAG y ecuaciones antes de cada bloque |
| 2.2 Parámetros y supuestos | ✅ | Estimandos (τ, OR condicional), supuestos de Gauss-Markov, SUTVA y positividad (Entrada 4) |
| 2.3 Modelos de resguardo | ✅ | Welch, Kruskal, Games-Howell, HC3, Huber, RANSAC con umbral corregido, umbral F1 |
| 3.1 Escenarios y sensibilidad propios | ✅ | Holdout cronológico, T = top-4, PCA como features, ranking vs. Expedia, U simulado, simulaciones S y C |
| 3.2 Convergencia y estabilidad | ✅ | Curva OOB/CV vs. número de árboles, CV de 5 pliegues, refutación por subconjuntos |
| 3.3 Contrastes diagnósticos | ✅ | Batería completa con estadístico, gl y p; refutaciones |
| 3.4 Visualización avanzada | ✅ | Paneles con IC sombreados, cercas anotadas, Q-Q por grupo, reliability diagrams, forest plot de OR, CATE con bandas |
| 4.1 Vehículo editorial | ✅ | Informe Quarto HTML + PDF con resumen, índice, tablas y figuras numeradas; notebook como anexo |
| 4.2 Redacción | ⚠️ | Sobria y técnica; **pendiente la revisión de estilo de Verónica** |
| 4.3 Traducción a negocio | ✅ | NB 4.9 e informe (valuación del slot, asignación por CATE, experimentación continua, límites) |
| 4.4 Reproducibilidad, fuentes y gobernanza de IA | ✅ | Tests, CI, semilla, bibliografía verificada (Crossref/UCI), Entrada 5 + `docs/ia_log.md` |
| Scrum: Quality Gate de QA | ⚠️ | Hay PRs con veredicto, pero **la revisión la hace la misma persona que integra** (declarado). Riesgo en la dimensión Scrum (10 pts) |
| Defensa oral por rol | ⚠️ | `docs/defensa.md` completo; **Lucía y Antonella tienen que estudiarlo**, porque el log declara que no ejecutaron el trabajo |

## D. Riesgos y pendientes antes del 18/11

1. **Deploy y URL** (Verónica): pasos en el README. Verificar que Streamlit Cloud use
   `app/requirements.txt` y no el de la raíz, que es pesado.
2. **Firmas de la BITÁCORA** (Verónica).
3. **Merge de los PRs en orden** (`scripts/github/abrir_prs_pendientes.sh`).
4. **Confidencialidad:** riesgo residual asumido; los parámetros del modelo y los efectos publicados
   podrían considerarse "resulting model parameters".
5. **Dimensión Scrum:** el registro honesto de que una sola persona hizo y revisó todo puede
   restar puntos en "dinámica de equipo".
6. **Puntos individuales en el notebook (decisión de Verónica).** La app solo publica agregados, pero
   el notebook versionado incluye dos gráficos de dispersión con puntos individuales, sin
   identificadores:
   - `f2_pca_plano`: 4.000 hoteles de Test en el plano PCA más los reservados.
   - `f2_influencia`: una observación por reserva.

   Además, algunas tablas de resumen muestran mínimos y máximos (valores de un solo registro).
   Si se quiere el criterio estricto de "solo agregados" también en el notebook, hay que
   reemplazarlos por hexbin o deciles y re-ejecutar (≈ 15 min).
7. **Avisos de joblib en Windows** al cerrar los procesos de trabajo de las búsquedas paralelas
   (`KeyError` en `resource_tracker`): son ruido de limpieza de temporales, no errores de celda (el
   notebook termina con código 0).
