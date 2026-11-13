# Misión: cerrar el TPI de EAyMR 2026 para sacar un 10

Sos el ingeniero responsable de terminar y entregar este proyecto. El trabajo es el
Trabajo Práctico Integrador "Decision Intelligence Capstone" de Estadística Aplicada y
Modelos de Regresión (UCA, 2S2026). Nuestro problema es el **retorno causal de un
algoritmo de recomendación**, con el dataset **Expedia Hotel Search (Kaggle)**.
Herramientas centrales: statsmodels (OLS con ajuste Backdoor y CATE), DoWhy (DAG,
identificación, refutaciones) y EconML (DML).

La entrega y la defensa oral son el **miércoles 18/11/2026 a las 23:59**. Hoy es 13/11.
Al final del día tengo que mandarle los entregables a mis compañeras. El objetivo no es
llegar a "funciona": cada criterio de la rúbrica tiene que quedar en el nivel
**Excelente**. Trabajá de forma autónoma y completa. Solo pará para preguntarme lo que
de verdad no puedas hacer vos (logins, deploy en la nube, decisiones irreversibles).

---

## 0. Fuentes de verdad (leelas COMPLETAS antes de tocar nada)

1. `docs/Enunciado_TP_Integrador_EAyMR_2026_1.pdf`: el enunciado oficial (5 fases, entregables, tabla de 100 pts).
2. `docs/rubrica.txt`: la rúbrica complementaria (Analizar / Modelar / Explorar / Informar, de 1 a 4 pts por subcriterio).
3. Todo lo que ya existe: `README.md`, `BITACORA.md`, `docs/*.md` (auditoria_final, dag_borrador, defensa, ia_log, plan_b), `report/informe.qmd`, `report/slides.qmd`, `referencias.bib`, `notebooks/`, `src/`, `app/`, `tests/`, `models/`, `scripts/`, `.github/`, `pyproject.toml`, `requirements.txt`.
4. Estado de git y GitHub: `git status`, `git log --all --oneline --graph`, las 7 ramas, el PR abierto y los issues (`gh pr list`, `gh issue list --limit 100`, `gh pr view`). En `main` hay un solo commit, así que probablemente haya trabajo hecho en otras ramas que todavía no se integró. Revisalo bien antes de rehacer cualquier cosa.

**Regla:** no reescribas lo que ya está bien. Primero auditá, después completá.

---

## 1. Auditoría inicial (sin modificar código)

Generá `docs/estado_cierre.md` con una matriz de trazabilidad. Cada fila es un requisito
del enunciado o un subcriterio de la rúbrica, con estas columnas:
`Requisito | Dónde está (archivo:sección) | Estado (✅ / ⚠️ parcial / ❌ falta) | Qué falta para Excelente | Prioridad`.

Cubrí como mínimo:
- **Fase 1:** EDA (KDE, boxplots, correlaciones Pearson y Spearman), mecanismo de faltantes MCAR/MAR/MNAR justificado (Expedia tiene muchísimos NaN en competidores, historial del usuario y review score; testealo, por ejemplo con un test de Little o con un logit de missingness sobre las otras variables), ≥2 contrastes de hipótesis con α fijado a priori, p exacto, IC 95% de la diferencia y tamaño de efecto, ANOVA de 1 factor con ≥3 grupos, normalidad evaluada por skewness, curtosis y Q-Q (con una explicación de por qué Shapiro no sirve con N grande), Levene, Welch o Kruskal-Wallis, post-hoc Tukey vs Bonferroni.
- **Fase 2:** OLS con R², R² ajustado, F y t; Breusch-Pagan o White, Jarque-Bera, Durbin-Watson si corresponde; residuos estudentizados (>3), leverage (>2(p+1)/n), Cook (>4/n y >0.5) y decisión justificada (Huber/RANSAC o HC3); VIF con su estrategia; PCA ajustado SOLO en train (scree, varianza acumulada, codo o Kaiser, interpretación de loadings PC1/PC2).
- **Fase 3:** split 80/20 (estratificado si es clasificación) + StratifiedKFold(5); Logit regularizada con Odds Ratios e IC 95% (o Ridge/Lasso/ElasticNet si es regresión); Naive Bayes con discusión de la independencia condicional y su efecto en la calibración; CART sin restricciones (overfitting train vs test) y después podado con `cost_complexity_pruning_path` + CV, con el árbol visualizado y la regla de la raíz interpretada; Random Forest con Grid o RandomizedSearchCV sobre los 5 hiperparámetros pedidos y OOB vs CV; métricas completas (Precision, Recall, F1 macro y ponderado, ROC-AUC, PR-AUC, Brier); reliability diagrams; MDI vs Permutation Importance en test con la explicación del sesgo de MDI.
- **Fase 4:** DAG en DOT/networkx con T, Y, Z, M (mediador) y C (colisionador) justificados arco por arco; demostración del Backdoor; pregunta trampa del mediador y del colisionador **demostrada empíricamente** (estimar con y sin M, con y sin C, y mostrar el sesgo); DoWhy en 4 pasos con ATE por `backdoor.linear_regression` y `backdoor.propensity_score_weighting` (con chequeo de overlap/positividad y pesos recortados si hace falta); comparación contra el coeficiente ingenuo Y~T, cuantificando el sesgo de confusión; EconML LinearDML y/o CausalForestDML con nuisances RF, CATE con curvas por subgrupo e IC; las 3 refutaciones (random common cause, placebo, data subset 80%) con interpretación; ensayo crítico "Predictivo vs Causal".
  - Si nuestro T es `random_bool` (orden aleatorio vs ranking del algoritmo de Expedia), aprovechalo: es una asignación aleatorizada que sirve como benchmark experimental para comparar contra la estimación observacional. Revisá además que `position` esté tratada como mediador y no como confusor, y que `click_bool` no se use como condición si funciona como colisionador o mediador de `booking_bool`. Verificá todo esto contra lo que ya está en `docs/dag_borrador.md`.
- **Fase 5:** roles Scrum documentados en el README y en la defensa; Quality Gate (pytest + ruff + chequeo de leakage en CI de `.github/`); veredicto QA `[APROBADO]` en cada PR; `BITACORA.md` con las 5 entradas obligatorias firmadas por rol (la Entrada 3 con ≥2 experimentos fallidos reales y la Entrada 5 con uso de IA, alucinaciones detectadas y tests que las refutan); Web App con los 3 módulos (Diagnóstico + PCA 2D, Predictivo "What-Is" con probabilidad calibrada y explicación del caso, Causal "What-If" con DAG, ATE, curva CATE según perfil y semáforo de refutaciones). El copiloto con tool use es bonus: hacelo solo si sobra tiempo y con guardrails contra cifras inventadas.
- **Rúbrica complementaria:** recorré cada uno de los 15 subcriterios (1.1 a 4.4) y escribí qué haría falta para el nivel Excelente. Por ejemplo: análisis de potencia (1−β) y justificación de α (1.4); formulación matemática y DAG antes del código (2.1); parámetros poblacionales vs estimadores bien notados (2.2); modelos de resguardo como Welch, Kruskal, HC3 o umbral por costos asimétricos (2.3); sensibilidad propia no pedida, como con/sin outliers, distintos umbrales τ o subsegmentos (3.1); estabilidad con k-fold e IC bootstrap del ATE (3.2); gráficos compuestos de nivel publicación (3.4); informe Quarto independiente con resumen, índice y figuras/tablas numeradas (4.1); recomendaciones de negocio **cuantificadas** (4.3); reproducibilidad total + bibliografía formal + declaración de gobernanza de IA (4.4).

Después de la auditoría armá la lista de tareas con TaskCreate, ordenada por puntos en
juego (Causal 25 > ML 20 > Inferencia 15 = Gauss-Markov/PCA 15 = Deploy 15 > Scrum 10),
y ejecutala.

---

## 2. Reglas de calidad innegociables

1. **Cero números inventados.** Toda cifra del informe, las slides, el README y la app tiene que salir de código ejecutado. En Quarto usá valores inline calculados o leídos de un `results/*.json` que genere el pipeline. Si una cifra del texto no coincide con la salida, el código manda.
2. **Reproducibilidad:** semilla global fija y documentada, dependencias fijadas, rutas relativas, script o instrucciones para bajar el dataset de Kaggle (los datos crudos no van en git si son pesados; si se usa una submuestra, que sea determinística y justificada).
3. **Notebook maestro** `notebooks/TP_Integrador_Final.ipynb` (nombre exigido por el enunciado), ordenado en las 4 fases técnicas, con teoría antes de cada bloque de código, gráficos con título y ejes, y salidas guardadas. Tiene que correr de punta a punta con Restart & Run All sin errores, sin celdas vacías y sin warnings ruidosos ni prints de debug. La lógica pesada va en `src/tpi/` y el notebook la importa.
4. **Sin data leakage:** el escalado, el PCA, la imputación y la selección se ajustan solo en train (con Pipelines). Tiene que haber tests en pytest que lo verifiquen.
5. **Lenguaje estadístico preciso:** "no se rechaza H0" (nunca "se acepta"); distinguir parámetro (β, μ, τ) de estimador (β̂); reportar estadístico, gl y p en cada test.
6. **Redacción:** español rioplatense técnico, sobrio y denso, en primera persona del plural ("estimamos", "decidimos"), con tono natural de estudiantes. Nada de relleno ni frases que suenen a IA. Ecuaciones en LaTeX.
7. **Verificación final obligatoria:** `pytest` en verde, `ruff check` limpio, notebook ejecutado completo, `quarto render` del informe y de las slides sin errores. Abrí los PDF y HTML generados y revisalos: figuras que se vean, numeración, índice, referencias resueltas, sin `??` ni placeholders tipo TODO/XXX/lorem.

---

## 3. Git y GitHub

- Integrá el trabajo de las ramas existentes y del PR abierto (revisá los conflictos con criterio, no sobrescribas a ciegas).
- Trabajá en ramas por fase y abrí PRs con el checklist del Quality Gate. Antes del merge a `main`, dejá en cada PR un comentario de QA con el veredicto `[APROBADO]` o `[BLOQUEADO]` y su evidencia (tests, ruff, leakage). Esto lo exige el enunciado.
- Cerrá o actualizá los 37 issues según corresponda, con referencia al commit o PR, y agrupalos en milestones por fase.
- Commits convencionales en español (`feat(fase-4): ...`). **Nunca** hagas force push a `main` ni borres ramas sin preguntarme.

---

## 4. Deploy de la Web App

- La app (`app/`, Streamlit) carga modelos y resultados serializados (joblib/json); no reentrena al iniciar. Tiene que levantar rápido y con datos livianos (una muestra incluida en el repo, si hace falta).
- Probala localmente (`streamlit run`) y verificá los 3 módulos.
- Dejá todo listo para Streamlit Community Cloud (requirements de la app, versión de Python, `.streamlit/config.toml`) y además un `Dockerfile` testeado como plan B.
- El deploy en la nube requiere mi login. Cuando llegues ahí, dame los pasos exactos, cortos y numerados para hacerlo yo. Cuando te pase la URL, agregala al README, al informe y a las slides.

---

## 5. Informe, slides, README y anexo de IA

- **`report/informe.qmd` → `informe.pdf` + `informe.html`:** portada (UCA, materia, profesor Ivan Gonzalo Vilaboa, integrantes con su rol, fecha), resumen ejecutivo, índice, introducción y preguntas de negocio, datos y auditoría de calidad (antes/después de la limpieza), las 4 fases, el ensayo crítico, **recomendaciones de negocio cuantificadas** (por ejemplo, lift causal en conversión o reservas y a qué segmentos conviene aplicar el algoritmo según el CATE), limitaciones y amenazas a la validez, conclusiones, bibliografía formal (`referencias.bib`: Pearl, Chernozhukov et al. DML, docs de DoWhy y EconML, Kaggle, etc.) y anexos.
- **Anexo "Uso de IA" (obligatorio):** herramientas usadas (Claude / Claude Code), resumen de las conversaciones y de los prompts de diseño principales, cómo se trabajó con el agente (harness, tareas delegadas), alucinaciones y errores detectados, tests determinísticos que los refutan, y una declaración de alcance y auditoría humana. Basalo en `docs/ia_log.md` y alineado con la Entrada 5 de la bitácora. Incluí este mismo prompt como ejemplo.
- **`report/slides.qmd` → `slides.pdf`:** de 10 a 12 slides siguiendo la estructura recomendada del enunciado (problema → inferencia/ANOVA/Gauss-Markov/PCA → torneo ML y calibración → DAG/ATE/CATE/refutaciones → demo de la app → conclusiones y bitácora), para 12 a 15 minutos.
- **`docs/defensa.md`:** guion de la exposición y banco de preguntas probables del tribunal, con respuestas cortas por rol (sobre todo: por qué Shapiro no sirve con N grande, la trampa del mediador y del colisionador, por qué MDI está sesgado, qué hace DML, qué significa cada refutación).
- **`README.md`:** de calidad editorial, con badges (CI, Python, deploy), diagrama del pipeline y del DAG, estructura del repo, roles, reproducción paso a paso, link a la app, al informe y a las slides.

---

## 6. Al terminar, entregame esto

1. **Lista de entregables finales** con su ruta exacta, para mandarlos (informe PDF/HTML, notebook ejecutado y su export HTML, slides PDF, BITACORA.md, link del repo, link de la app). Si conviene, armá un `entrega/` o un .zip con todo junto.
2. **Tabla final de la rúbrica:** cada subcriterio, el nivel que estimás que alcanza y la evidencia.
3. **Lo que tengo que hacer yo a mano** (deploy, revisar algo puntual, subir a EVA), en pasos cortos.
4. **Riesgos residuales**, si los hay, explicados con honestidad.

No me digas que algo está terminado si no lo ejecutaste y verificaste. Si encontrás un
error metodológico en lo que ya hicimos, corregilo y anotalo en la bitácora; no lo
escondas.
