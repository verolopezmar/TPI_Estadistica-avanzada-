# Estado de cierre: matriz de trazabilidad contra el enunciado y la rúbrica

> Auditoría del 13/11/2026, previa a cualquier cambio de código. Partimos de la rama
> `feat/fase5-entrega` (4e64ae5), que contiene las Fases 0 a 5 en historia lineal. `main` tiene un
> solo commit (c55ba99); el único PR abierto es el #38 (Fase 0), con CI en verde. Hay 37 issues
> abiertos, ya asignados a 6 milestones.
>
> Estado inicial verificado: `ruff check` y `ruff format --check` limpios; `pytest`: 127 pasan,
> 1 skip y 2 xfail estrictos (la premisa refutada de `random_bool`). El notebook ejecutado tiene
> 141 celdas, ninguna vacía, sin errores y con 3 salidas a stderr (`SingularMatrixWarning`).
>
> Estados: ✅ cumplido · ⚠️ parcial · ❌ falta. Prioridad: **A** = puntos en juego o error
> metodológico; **B** = sube un subcriterio de "Bueno" a "Excelente"; **C** = pulido.
> "NB x.y" es la sección del notebook `notebooks/TP_Integrador_Final.ipynb`.

## 0. Hallazgo metodológico de la auditoría

**Fuga de información en la preparación (Prioridad A).** `tpi.preparacion.preparar()` y el
notebook (NB 1.2-1.3) calculan con **toda** la muestra, antes de la partición 80/20:

- las medianas de imputación por grupo (`imputar_con_indicador`);
- los cuantiles de winsorización del precio relativo y las cercas de Tukey (`precio_relativo`).

Test aporta así a las estadísticas con las que se transforma Train. La magnitud es chica (medianas
sobre 739.323 filas), pero contradice la regla de ajustar todo solo con Train y el criterio 1.4 de la
rúbrica. **Corrección:** la partición depende solo de `srch_id`, `random_bool` y la semilla, así
que puede definirse antes de la preparación. Las estadísticas se ajustan con las filas de Train y se
aplican a todas. Se agrega un test que falla si Test modifica lo que ve Train, y se registra en la
BITÁCORA (Entrada 3).

## 1. Fase 1 — Inferencia clásica y ANOVA (15 pts)

| Requisito | Dónde está | Estado | Qué falta para Excelente | Prior. |
|---|---|---|---|---|
| EDA univariado y bivariado: KDE, boxplots, Pearson y Spearman | NB 1.5; `report/figures/f1_correlaciones.png` | ✅ | — | — |
| Mecanismo de faltantes MCAR/MAR/MNAR justificado | NB 1.3; `tpi.auditoria.contraste_mcar` (χ²/Mann-Whitney por variable, Bonferroni, SMD/V) | ⚠️ | Falta un contraste **multivariado** de MCAR. Agregamos el test de Little (1988) con EM, testeado con verdad conocida (MCAR no rechaza, MAR rechaza) | A |
| Imputación justificada y sin fuga | NB 1.3; `tpi.preparacion` | ⚠️ | Ajustar las medianas solo con Train (sección 0) | A |
| ≥ 2 contrastes con H0/H1, α a priori, p exacto, IC 95 % y tamaño de efecto | NB 1.6 y 1.7; `tpi.inferencia` | ✅ | En el informe, explicitar H0/H1 en notación de parámetros (μ, δ) | B |
| Potencia y efecto mínimo detectable | NB 1.1, 1.6 y 1.7 (`efecto_minimo_detectable`, `potencia`) | ✅ | — | — |
| ANOVA de 1 factor con ≥ 3 grupos | NB 1.8 (4 categorías de estrellas) | ✅ | — | — |
| Normalidad por asimetría, curtosis y Q-Q; por qué no Shapiro con N grande | NB 1.8; Entrada 2 | ✅ | Agregar al informe el argumento formal (potencia de Shapiro → 1 con n) | B |
| Levene; Welch o Kruskal-Wallis | NB 1.8 (Levene rechaza → Welch; Kruskal de resguardo) | ✅ | — | — |
| Post-hoc Tukey vs. Bonferroni | NB 1.8 (además Games-Howell) | ✅ | — | — |

## 2. Fase 2 — Gauss-Markov y PCA (15 pts)

| Requisito | Dónde está | Estado | Qué falta para Excelente | Prior. |
|---|---|---|---|---|
| MCO con R², R² ajustado, F y t | NB 2.1 | ✅ | — | — |
| Breusch-Pagan, White, Jarque-Bera, Durbin-Watson | NB 2.2; `tpi.regresion.diagnostico_gauss_markov` | ✅ | Eliminar el `SingularMatrixWarning` de White (cuadrados de dummies = dummies), documentando que statsmodels usa el rango para los gl | C |
| Estudentizados > 3, leverage > 2(p+1)/n, Cook > 4/n y > 0,5, decisión | NB 2.3 (Huber, RANSAC con umbral corregido, HC3) | ✅ | — | — |
| VIF con estrategia | NB 2.4 | ✅ | — | — |
| PCA solo en Train: scree, acumulada, Kaiser/codo, cargas PC1/PC2 | NB 2.5; `test_pca_se_ajusta_solo_con_train` | ✅ | Las variables de entrada al PCA se imputaban con medianas de toda la muestra (sección 0) | A |

## 3. Fase 3 — Torneo ML (20 pts)

| Requisito | Dónde está | Estado | Qué falta para Excelente | Prior. |
|---|---|---|---|---|
| 80/20 estratificado + StratifiedKFold(5) | `tpi.particion` (agrupado por búsqueda, estratificado por `random_bool`); `StratifiedGroupKFold(5)` | ✅ | — | — |
| Logit regularizada con OR e IC 95 % | NB 3.1 (L1/L2 por grilla; OR de logit sin penalizar con EE por clúster) | ✅ | Llevar la tabla de OR al informe desde `resultados.json` | B |
| Naive Bayes, independencia condicional y calibración | NB 3.2 y 3.5 | ✅ | — | — |
| CART sin restricciones vs. podado (`cost_complexity_pruning_path` + CV), árbol y regla raíz | NB 3.3 | ✅ | — | — |
| RF con búsqueda sobre los 5 hiperparámetros; OOB vs. CV | NB 3.4 (con curva de convergencia) | ✅ | — | — |
| Precision, Recall, F1 macro y ponderado, ROC-AUC, PR-AUC, Brier | NB 3.5 | ✅ | — | — |
| Reliability diagrams | NB 3.5; `f3_evaluacion.png` | ✅ | — | — |
| MDI vs. permutación en Test, con el sesgo de MDI explicado | NB 3.6 | ✅ | — | — |

## 4. Fase 4 — Inferencia causal (25 pts)

| Requisito | Dónde está | Estado | Qué falta para Excelente | Prior. |
|---|---|---|---|---|
| DAG (networkx/DOT) con T, Y, Z, M y C, arco por arco | NB 4.1; `tpi.dag`; Entrada 4; `docs/dag_borrador.md` | ✅ | — | — |
| T bien definido; `position` no usada como confusor; clic no condicionado | T = 1{position ≤ 5}; clic = mediador M; R = `random_bool` = benchmark experimental | ✅ | Verificado contra `docs/dag_borrador.md` y `tests/test_dag.py` | — |
| Demostración Backdoor | NB 4.1; 12 pruebas de d-separación | ✅ | — | — |
| Trampa del mediador y del colisionador **demostrada empíricamente** | NB 4.4 (con y sin clic), NB 4.5 (C simulado y selección S, con y sin condicionar) | ✅ | — | — |
| DoWhy en 4 pasos: regresión backdoor e IPW | NB 4.2 | ⚠️ | **IPW sin chequeo de overlap/positividad, sin pesos recortados y sin IC.** Agregar distribución de propensión por grupo, fracción fuera de [0,01; 0,99], tamaño efectivo, IPW recortado e IC bootstrap por búsqueda | A |
| Comparación con el ingenuo Y ~ T y sesgo de confusión cuantificado | NB 4.2 y 4.3; benchmark aleatorio | ✅ | — | — |
| EconML DML con nuisances RF, CATE con curvas e IC | NB 4.7 (LinearDML en R = 0 y R = 1; CausalForestDML en R = 0) | ⚠️ | Los CATE por segmento están solo en el texto del notebook (no en `resultados.json`) y no se cuantifica el impacto de negocio por segmento. Agregar tabla de CATE por segmento con IC, la heterogeneidad del LinearDML con orden aleatorio y la traducción a reservas por 1.000 exposiciones | A |
| 3 refutaciones con interpretación | NB 4.6 | ✅ | — | — |
| Ensayo crítico "Predictivo vs. Causal" | NB 4.8 | ⚠️ | Está en el notebook, pero no en el informe | B |
| Estabilidad: IC bootstrap del efecto | NB 4.3 bis (1000 réplicas por búsqueda) | ✅ | — | — |

## 5. Fase 5 — Scrum, bitácora y deploy (10 + 15 pts)

| Requisito | Dónde está | Estado | Qué falta para Excelente | Prior. |
|---|---|---|---|---|
| Roles Scrum en README y defensa | README; BITÁCORA; `docs/defensa.md` | ✅ | — | — |
| Quality Gate en CI (pytest + ruff + leakage) | `.github/workflows/quality-gate.yml` | ⚠️ | Los tests de fuga corren dentro de `pytest`, sin paso propio. Agregar marcador `leakage` y un paso explícito en el CI | B |
| Veredicto QA en cada PR antes del merge | Solo el PR #38 existe; sin comentario de QA | ❌ | Abrir PRs por fase, comentar `[APROBADO]` con evidencia (CI, pytest, ruff, leakage) y mergear en orden a `main` | A |
| Issues y milestones | 37 issues abiertos en 6 milestones | ⚠️ | Cerrar cada issue con referencia al PR o commit | B |
| BITÁCORA: 5 entradas | `BITACORA.md` | ⚠️ | Falta la firma por rol en cada entrada. Entrada 3: hay 5 experimentos fallidos reales. Agregar la corrección de la fuga. Entrada 5: agregar esta sesión de cierre | A |
| App: Diagnóstico + PCA 2D | `app/app.py` (tab 1) | ✅ | — | — |
| App: What-Is con probabilidad calibrada y explicación del caso | `app/app.py` (tab 2) | ✅ | Verificar en vivo | B |
| App: What-If con DAG, ATE, CATE por perfil y semáforo | `app/app.py` (tab 3) | ✅ | Verificar en vivo | B |
| Bonus: copiloto con tool use y guardarraíl | `app/copiloto.py`; `tests/test_copiloto.py` | ✅ | — | — |
| Deploy en la nube | — | ❌ | Requiere el login de Verónica en Streamlit Community Cloud. Dejar `.streamlit/config.toml`, versión de Python y pasos exactos | A |
| Plan B: Dockerfile testeado | — | ❌ | Escribir `Dockerfile` + `.dockerignore`, construir y probar el contenedor | A |

## 6. Entregables (enunciado, sección 5)

| Entregable | Dónde está | Estado | Qué falta | Prior. |
|---|---|---|---|---|
| Repo modular | raíz | ✅ | — | — |
| README editorial con badges, diagramas, reproducción y link al deploy | `README.md` | ⚠️ | URL del deploy pendiente; agregar diagrama del DAG y Docker | A |
| Dependencias fijadas | `requirements.txt`, `app/requirements.txt`, `pyproject.toml` | ✅ | — | — |
| Notebook Restart & Run All, sin warnings ruidosos | `notebooks/TP_Integrador_Final.ipynb` | ⚠️ | Volver a ejecutar después de las correcciones; eliminar las 3 salidas a stderr; exportar a HTML | A |
| Informe Quarto (PDF + HTML) | `report/informe.qmd` | ⚠️ | Ver 4.1-4.4 abajo | A |
| Slides PDF (10-12) | `report/slides.qmd` | ⚠️ | Agregar slide de Scrum/QA y el link a la app | B |
| Guion y banco de preguntas de la defensa | `docs/defensa.md` | ⚠️ | Hay guía por rol y preguntas de práctica **sin respuesta**. Agregar guion cronometrado y respuestas cortas por rol | B |

## 7. Rúbrica complementaria (15 subcriterios)

| Subcriterio | Estado actual | Qué haría falta para Excelente | Prior. |
|---|---|---|---|
| 1.1 Diagnóstico del problema | Excelente: suma cero de los slots, valuación, preguntas propias 5 y 6 | — | — |
| 1.2 Estructura relacional | Excelente: DAG con Y, T, Z, M, C, S, U; Spearman vs. Pearson; Zc contrastada | — | — |
| 1.3 Auditoría forense (DGP) | Excelente: centinelas, escala, slots fijos, selección, antes/después | Tabla antes/después en el informe | B |
| 1.4 Diseño muestral | **Bueno**, por la fuga de la sección 0 | Corregir la fuga y documentar en el informe α, potencia y partición | A |
| 2.1 Abstracción antes del código | Excelente en el notebook; el informe entra directo a resultados | Sección "Marco formal" en el informe: resultados potenciales, estimandos y DAG antes de los resultados | B |
| 2.2 Parámetros y supuestos | Bueno a excelente | Notación parámetro/estimador explícita (β vs. β̂, τ vs. τ̂) y supuestos de identificación (SUTVA, positividad, ignorabilidad condicional) en el informe | B |
| 2.3 Modelos de resguardo | Excelente: Welch, Kruskal, Games-Howell, HC3, Huber, RANSAC corregido, umbral F1 | Agregar IPW con pesos recortados | A |
| 3.1 Sensibilidad propia | Excelente: holdout cronológico, top-4, U simulado, re-ponderación, escenarios | — | — |
| 3.2 Convergencia y estabilidad | Excelente: OOB/CV vs. árboles, bootstrap B = 1000 | IC bootstrap también para el IPW | A |
| 3.3 Rigor en contrastes | Excelente: estadístico, gl y p en cada test | Little (MCAR multivariado) | A |
| 3.4 Visualización avanzada | Excelente: paneles, IC sombreados, umbrales anotados | Figura de overlap (propensión por grupo con zonas recortadas) | A |
| 4.1 Vehículo editorial | Bueno a excelente: Quarto HTML+PDF con índice y figuras numeradas | Portada institucional (UCA, cátedra, profesor, roles, fecha), conclusiones, anexos y figuras referenciadas en el texto | A |
| 4.2 Redacción | Buena; sobria | Revisión final de estilo y notación | B |
| 4.3 Traducción a negocio | Bueno a excelente: escenarios en USD | Recomendaciones **por segmento** con números (CATE → reservas y USD por 1.000 exposiciones) | A |
| 4.4 Reproducibilidad, fuentes y gobernanza IA | Bueno: la bibliografía tiene 6 entradas | Bibliografía formal (EconML, Kaggle, scikit-learn, statsmodels, Little, Welch, Strobl, Wager-Athey, Niculescu-Mizil, Hernán-Robins) y anexo de IA completo en el informe | A |

## 8. Plan de trabajo (ordenado por puntos en juego)

1. **Causal (25):** IPW con overlap, recorte e IC bootstrap; CATE por segmento con IC en
   `resultados.json`; heterogeneidad del benchmark; recomendaciones cuantificadas por segmento.
2. **ML (20) e Inferencia/GM-PCA (15 + 15):** corregir la fuga de la preparación (afecta Fases 2-4);
   test de Little; tests de no-fuga marcados; limpiar warnings.
3. **Deploy (15):** prueba local de los 3 módulos; `.streamlit/config.toml`; `Dockerfile` probado;
   pasos de deploy para Verónica.
4. **Scrum (10):** PRs por fase con veredicto QA y merge a `main`; cierre de issues; firmas y
   entradas nuevas en la BITÁCORA; paso de leakage en el CI.
5. **Informe, slides, defensa y README:** portada, marco formal, ensayo, negocio por segmento,
   conclusiones, anexo de IA, bibliografía; slide de Scrum/QA; guion y respuestas; README con DAG y
   Docker.
6. **Verificación final:** notebook ejecutado de punta a punta, `pytest`, `ruff`, `quarto render`
   del informe y de las slides, revisión visual de los PDF y HTML, carpeta `entrega/`.
