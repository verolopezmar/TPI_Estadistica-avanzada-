# Guía de estudio para la defensa oral, por rol

> Para: Verónica, Lucía y Antonella. En la ronda de preguntas, cada una defiende el trabajo de su
> rol. Todo número de esta guía sale de un output ejecutado; la fuente está entre paréntesis.

**Cómo usar esta guía.**
1. Leé tu rol en cada fase.
2. Abrí la sección del notebook o el archivo que se cita y mirá el output con tus propios ojos.
3. Respondé en voz alta las **preguntas de práctica** del final, sin mirar, y después verificá la
   respuesta.
4. Si te preguntan quién hizo el trabajo, la respuesta honesta está en el README y en la
   BITÁCORA: Verónica lo ejecutó y validó con asistencia de Claude Code, y los roles se mantienen
   para la defensa. Lo que sí tenés que dominar es el razonamiento de tu rol.

---

## Fase 0 — Setup, datos y contrato (cerrada el 12/11/2026)

### Verónica — Project Lead & Scrum Master

1. **Pregunta de negocio:** ¿el slot top-5 *causa* reservas o solo las *predice*? El ranker pone
   arriba hoteles que se venderían igual. Distinguir una cosa de la otra define cuánto vale un
   slot premium.
2. **Por qué Expedia:** son datos reales de producción. `train.csv` tiene 9.917.530 filas y
   399.344 búsquedas, las mismas cifras que Liu et al. (2013), y trae un experimento natural
   (`random_bool`) que casi ningún dataset observacional ofrece (`data/manifiesto.json`).
3. **Muestra:** 30.000 búsquedas *completas*, 15.000 por estrato de `random_bool`. Se muestrean
   búsquedas y no filas porque los hoteles de una lista comparten usuario y contexto.
4. **Confidencialidad:** las reglas de Kaggle prohíben compartir los datos y los "resulting model
   parameters". Mitigaciones:
   - Ningún dato en el repo; lo garantiza `tests/test_confidencialidad.py`.
   - El crudo queda fuera del repo y de OneDrive.
   - La app muestra solo agregados.
   - Decisión mía (12/11): no consultar a la cátedra y asumir el riesgo residual, documentado en
     la BITÁCORA (Entradas 1 y 4).
5. **Plan B:** Online Shoppers (UCI #468, CC BY 4.0) con tratamiento semisintético. Se activa con
   una variable de entorno porque el pipeline depende de un contrato genérico, no de Expedia. Con
   los datos reales, el ATE verdadero es 0,0646 y el estimador ingenuo da 0,2906
   (`docs/plan_b.md`).
6. **Gobernanza de git:** los primeros 5 commits se pushearon sin PR, pero nunca llegaron a
   `main`. No reescribimos el historial: los auditamos en el PR de la Fase 0.
7. **Qué no está identificado:** el efecto sobre usuarios que no hacen clic. El dataset solo trae
   búsquedas con al menos un clic.

### Lucía — Backend & ML / Causal Engineer

1. **Estimando:** $T = 1\{\text{position} \le 5\}$, $Y$ = `booking_bool`, $M$ = `click_bool`.
   ATE $= E[Y(1) - Y(0)]$ y CATE $\tau(x)$.
2. **Experimento natural:** según Kaggle, `random_bool` = "+1 when the displayed sort was random".
   Con R = 1 la intervención corta los arcos que entran a T (cirugía del grafo).
3. **Pero no es un RCT puro.** La correlación de Spearman intra-búsqueda entre la posición y las
   estrellas es −0,074 con orden aleatorio y −0,256 con el algoritmo; con `location_score2`,
   −0,065 y −0,299. Por eso R = 1 se usa como **cuasi-experimento ajustado**: efectos fijos por
   búsqueda más ajuste por Z.
4. **Cómo descartamos la selección como causa:** la asociación residual persiste en búsquedas sin
   reserva y sin clics (−0,070 y −0,063).
5. **Selección del dataset (colisionador natural):** ninguna búsqueda tiene cero clics.
   - El clic es colisionador: $T \to M \leftarrow I$, donde I es la intención de compra.
   - El filtro de inclusión desciende de M, así que condicionar en él abre el camino entre T e I
     (sesgo de Berkson).
6. **Por qué la escala relativa dentro de la búsqueda:** como un clic implica estar en el
   dataset,
   $P(M=1 \mid T, S=1) = P(M=1 \mid T) / P(S=1)$.
   - El cociente de riesgos sobrevive a la selección; la diferencia absoluta se infla.
   - Con selección por el resultado, el odds ratio es invariante (Prentice y Pyke, 1979).
7. **Tasas base no comparables:** hay reserva en el 93,86 % de las búsquedas R = 0 y en el
   13,04 % de las R = 1. El ATE absoluto de R = 0 no se compara con el de R = 1.
8. **Unidades independientes:** los contrastes se hacen sobre búsquedas o sobre reservas (a lo
   sumo una por búsqueda), nunca sobre filas sueltas.

### Antonella — Frontend & UI/UX + QA & Validation Engineer

1. **Quality gate:** `ruff check`, `ruff format --check` y `pytest` en CI, más la plantilla de PR
   con veredicto `[APROBADO]`/`[BLOQUEADO]`.
2. **Qué garantiza cada test clave:**
   - `test_ningun_archivo_de_datos_trackeado`: ningún dato en git.
   - `test_crudo_fuera_del_repo`: el crudo no queda dentro del repo.
   - `test_tamano_train_coincide_con_lo_reportado`: N real igual a la cifra citada.
   - `test_mecanismo_de_seleccion_del_dataset`: el filtro de al menos un clic.
3. **Refutaciones registradas, no borradas:** el test "orden aleatorio = independiente" quedó como
   `xfail(strict=True)` en dos variables. Si un día pasa, pytest avisa. Borrar un test que
   falla es esconder evidencia.
4. **Umbral post hoc declarado:** el nuevo test exige que R = 1 reduzca al menos un 50 % la
   asociación del algoritmo. Se fijó *después* de ver los datos, y así se declara.
5. **Errores de la IA atrapados por tests:**
   - `df.T` es la transpuesta y no la columna "T" (`test_roles_no_chocan_con_atributos_de_dataframe`).
   - Nombres de columnas de pingouin 0.6.1 (`tests/test_inferencia.py`).
   - pyarrow incompatible con numpy (`test_environment.py`).
6. **Integridad de la descarga:** el zip usa Deflate64; la extracción verifica tamaño y CRC-32, y
   el manifiesto guarda el SHA-256.
7. **App (Fase 5):** por las reglas de confidencialidad, solo mostrará agregados y distribuciones,
   nunca registros individuales.

---

## Fase 1 — Inferencia clásica, auditoría y ANOVA (12/11/2026; pendiente de revisión de Verónica)

### Verónica — Project Lead & Scrum Master

1. **Diseño muestral:** la unidad de muestreo es la búsqueda (15.000 por estrato); los contrastes
   se hacen sobre unidades independientes (búsquedas o reservas), nunca sobre filas.
2. **α = 0,05 a priori + efecto mínimo detectable:** con n grande todo da "significativo". El MDE
   dice qué no puede ver la muestra. Ejemplo: en el contraste 2 no detectamos diferencias menores
   que g ≈ 0,09.
3. **Valor de negocio del contraste 1:** con orden aleatorio, estar en el top-5 **duplica** la
   tasa de clic (cociente 2,16; +5,6 p.p., IC 95 % [5,31; 5,90]). Es el valor bruto del slot sobre
   la atención.
4. **Slots estructurales:** las posiciones 5, 11, 17 y 23 casi nunca están en el dataset (las ocupa
   otro tipo de listado). El top-5 en pantalla suele tener 4 hoteles nuestros.
5. **Recurrentes vs. nuevos:** no hay evidencia de distinto gasto por reserva (cociente de medias
   geométricas 1,039, IC [0,981; 1,101]). "No se rechaza H0" no es "son iguales".

### Lucía — Backend & ML / Causal Engineer

1. **Contraste 1:** t pareado sobre $D_s$ = CTR top-5 − CTR resto, por búsqueda con R = 1.
   t(14982) = 37,47, p = 7,74·10⁻²⁹⁴, $d_z$ = 0,306. Es pareado porque comparte la búsqueda y eso
   elimina los confusores de búsqueda. Wilcoxon de resguardo.
2. **Contraste 2:** Welch sobre log gasto: t(1158,6) = 1,303, p = 0,193. Usamos Welch aunque
   Levene no rechazó (p = 0,074), porque los grupos están muy desbalanceados (1.025 vs. 15.039).
3. **ANOVA:** Levene rechaza (p = 2,27·10⁻³⁶), así que el principal es **Welch** (F(3; 4432,0) =
   1807). Kruskal-Wallis y Games-Howell como resguardo; η² = 0,254. Un hotel de 5 estrellas cuesta
   2,65 veces uno de 1-2.
4. **Por qué no Shapiro:** con n grande rechaza ante desvíos irrelevantes. Mirar asimetría
   (en [−1, 1]), curtosis y Q-Q.
5. **Faltantes:**
   - MCAR se contrasta (indicador de faltante vs. variables observadas) y se rechaza en las 5
     variables contrastadas.
   - MAR vs. MNAR no se puede contrastar: se argumenta por el mecanismo.
   - Lo MAR se imputa por grupo con indicador; lo MNAR estructural (historial del usuario) no se
     imputa.
6. **$Z_c$ (competidores):** confusor exige **dos** flechas, y se contrastaron por separado.
   - **Z → T:** interacción $Z_c \times R$ en el logit de T, condicional a la calidad. Si el ranker
     usa $Z_c$, su efecto sobre T cambia cuando el orden es aleatorio.
     `comp_saldo`: razón de OR 0,842, p = 2,7·10⁻¹⁸. `comp_n_info`: 1,019, p = 2,8·10⁻⁵ (chico).
   - **Z → Y:** con R = 1, la reserva tiene OR 1,137 (`comp_saldo`) y 1,118 (`comp_n_info`).
   - Las dos son confusoras; `comp_diff_max` se excluye (ninguna flecha significativa).
   - **Signo (fuente: Kaggle):** `compN_rate` = +1 si Expedia es **más barata**. `comp_saldo` > 0
     significa precio competitivo: el algoritmo lo sube al top-5 y además aumenta la reserva por su
     cuenta. Por eso omitirla sesgaría el efecto del top-5 hacia arriba.
   - El DAG suma Zc → T y Zc → M; un test demuestra que sin $Z_c$ no se cumple Backdoor.
7. **Slots fijos y T de sensibilidad:** las posiciones 5, 11, 17 y 23 son listados que no están
   en el dataset. T principal = top-5 en pantalla; sensibilidad = top-4 del dataset (coinciden en
   el 99,89 % de las filas).
8. **Limitación declarada:** con orden aleatorio queda una asociación residual posición–estrellas
   (ρ̄ −0,074) sin explicación verificable. No forzamos una hipótesis: ajustamos por Z y mostramos
   el efecto con y sin ajuste.
9. **Escala del precio:** gasto / (precio·noches·habitaciones) ≈ 1,13. El 87,5 % de las reservas
   de ≥ 2 noches es "por noche" y el 3,2 % "por estadía". Por eso usamos el precio relativo a la
   búsqueda.

### Antonella — Frontend & UI/UX + QA & Validation Engineer

1. **Tests de la auditoría:** cada regla se prueba con datos sintéticos de verdad conocida.
   - El precio relativo es invariante a la escala.
   - El outlier absoluto detecta búsquedas enteras imposibles.
   - El contraste de MCAR no da falsos positivos con faltantes aleatorios y detecta los MAR.
2. **Error de la IA atrapado:** la primera regla de outliers no marcaba precios de millones de USD por noche
   (error 17). La corrección quedó fijada por un test.
3. **Reglas de reporte automatizadas:**
   - `Contraste.texto()` nunca dice "se acepta H0".
   - Informa los gl sin notación científica.
   - Ante un subdesborde a 0, escribe "p < 1e-300" en lugar de "p = 0".
4. **Reproducibilidad:** el notebook se ejecuta de punta a punta con `jupyter nbconvert --execute`
   (equivalente a Restart & Run All), con rutas relativas y semilla 42.
5. **Visualización:**
   - Paleta categórica validada en orden fijo y una sola escala Y por panel.
   - IC 95 % sombreados, cercas de Tukey anotadas, Q-Q por grupo.
   - Figuras en `report/figures/f1_*.png`.
6. **Confidencialidad en el notebook:** solo tablas agregadas y figuras; ningún registro individual.

---

## Fase 2 — Gauss-Markov, influyentes, VIF y PCA (12/11/2026; pendiente de revisión de Verónica)

### Verónica — Project Lead & Scrum Master

1. **Partición única y sin fuga:** 80/20 agrupada por búsqueda (24.000 / 6.000 búsquedas),
   estratificada por `random_bool`, semilla 42. Test no se toca hasta la Fase 3.
2. **Por qué aleatoria y no cronológica:** son 8 meses de datos y el Durbin-Watson (1,95) no
   muestra autocorrelación temporal. El objetivo es generalizar a búsquedas nuevas del período,
   no pronosticar.
3. **Mensaje de negocio del PCA:** lo que se reserva no es lo más caro ni lo de mayor categoría
   (PC1), sino lo bien ubicado, en promoción y barato para su búsqueda (PC2: 0,270 en reservados
   vs. −0,004 en no reservados).
4. **Precio pagado:** la promoción baja el precio por noche un 14,9 %; cada estrella lo sube un
   6,2 %. El orden aleatorio no cambia el precio del hotel reservado (p = 0,661).
5. **Autopsia (Entrada 3):** tres experimentos fallidos reales (precio crudo, RANSAC por defecto,
   multicolinealidad anticipada que no apareció).

### Lucía — Backend & ML / Causal Engineer

1. **Modelo:** MCO de log(precio por noche) sobre 12.858 reservas de Train. R² = 0,604; F(17;
   12840) = 1150,64, p < 10⁻³⁰⁰. Los coeficientes se leen como % de cambio: $100(e^\beta - 1)$.
2. **Gauss-Markov:**
   - BP (p = 4,96·10⁻⁹⁹) y White rechazan la homocedasticidad, así que uso **HC3**: los
     coeficientes siguen insesgados y solo se corrige su varianza (EE hasta +31 %).
   - JB rechaza normalidad (curtosis 5,8), pero con n grande t y F valen por TCL.
   - DW = 1,95.
3. **Influencia:**
   - Umbrales: |r| > 3 (119 casos), h > 2(p+1)/n (843), D > 4/n (746) y D > 0,5 (ninguno).
   - Los influyentes son el doble de frecuentes cuando la escala del precio es incompatible (9,5 %
     vs. 4,9 %): **error de medición**.
4. **Robustez:**
   - Huber coincide con MCO sin influyentes (diferencia máx. 0,012).
   - RANSAC por defecto falla (umbral = MAD de *y*, 55 % de inliers, signos invertidos); con
     umbral 3σ̂ de los residuos queda bien.
5. **VIF:** máx 1,74 (modelo) y 2,98 (conjunto de la Fase 3). Una correlación de a pares alta
   (0,62) no implica un VIF alto: lo que cuenta es el $R^2_j$ sobre todos los regresores.
6. **PCA:**
   - Escalado y autovectores solo con Train; Test se proyecta con `transform`.
   - Kaiser: 4 componentes (69,7 %); el codo está en 2.
   - PC1 = categoría-precio; PC2 = ubicación y promoción, independiente de cadena.

### Antonella — Frontend & UI/UX + QA & Validation Engineer

1. **Tests de no fuga:**
   - `test_particion_agrupada_sin_busquedas_compartidas`: ninguna búsqueda queda en Train y Test
     a la vez.
   - `test_pca_se_ajusta_solo_con_train`: la media del escalador es la de Train, los autovectores
     no cambian al transformar Test, y un corrimiento de Test se ve en la proyección (no se
     "absorbe").
2. **Tests de diagnóstico con verdad conocida:**
   - BP detecta heterocedasticidad plantada y no da falsos positivos.
   - Cook marca un punto plantado.
   - VIF > 10 ante una casi copia de una variable.
3. **Reglas de reporte:** todos los p pasan por el formateador (nunca "p = 0"). Lo detecté y
   corregí en la Fase 2 (error 25 del log).
4. **Figuras de la Fase 2:** residuos vs. ajustados (hexbin en escala log), Q-Q, dispersión por
   ventil, influencia (tamaño ∝ Cook, umbrales anotados), scree con línea de Kaiser, varianza
   acumulada, cargas, y Test proyectado en PC1-PC2.
5. **App (Fase 5):** el módulo de diagnóstico va a mostrar el plano PC1-PC2 calculado con la
   rotación de Train, solo con agregados o muestras de puntos sin identificadores.

---

## Fase 3 — Torneo predictivo (12/11/2026; pendiente de revisión de Verónica)

### Verónica — Project Lead & Scrum Master

1. **Pregunta predictiva vs. causal:** la Fase 3 responde $P(Y \mid X)$ (qué hotel se va a
   reservar); la Fase 4 responde $P(Y \mid do(T))$ (qué pasa si lo subimos).
2. **Ganador:** Random Forest, con PR-AUC 0,154 en Test frente a una prevalencia de 0,022 (unas 7
   veces el azar) y ROC-AUC 0,849. El holdout cronológico (último 20 % del período) da 0,146: el
   modelo se sostiene hacia adelante.
3. **Por qué PR-AUC y no exactitud:** con 2 % de reservas, decir siempre "no reserva" da 98 % de
   exactitud y no sirve para nada.
4. **Decisión de cómputo:** hiperparámetros en el 25 % de las búsquedas de Train y reentrenamiento
   en todo Train, justificado y reportado.

### Lucía — Backend & ML / Causal Engineer

1. **Sin fuga:**
   - Se excluyen el clic (pasa en la misma sesión que la reserva) y el gasto.
   - La CV es `StratifiedGroupKFold(5)` por búsqueda.
   - `position` sí entra porque se conoce al mostrar la lista.
2. **Logística L1/L2:** la grilla elige L1 con C = 0,1. Los OR con IC 95 % salen de un logit sin
   penalizar con errores por clúster, porque la penalización sesga los coeficientes.
3. **Naive Bayes:**
   - La independencia condicional se viola: hay correlaciones dentro de cada clase de hasta 0,72
     (precio vs. precio histórico).
   - Resultado: probabilidades extremas y mal calibradas (Brier 0,070 contra 0,020 de la
     logística).
4. **CART:** sin poda, 18.896 hojas y PR-AUC de 1,000 en Train contra 0,031 en Test (memoriza).
   Podado por $R_\alpha(T) = R(T) + \alpha|T|$ con $\alpha$ elegido por CV, queda en 14 hojas. Regla
   raíz: `position ≤ 4,5`.
5. **Random Forest:** `RandomizedSearchCV` (20 combinaciones) sobre `n_estimators`, `max_depth`,
   `min_samples_split`, `min_samples_leaf` y `max_features`. El OOB converge con la CV al crecer el
   número de árboles.
6. **MDI vs. permutación:**
   - El MDI suma reducciones de impureza en Train y favorece variables con muchos puntos de corte
     (correlación con la cardinalidad ρ = 0,749).
   - La permutación mide cuánto cae la PR-AUC en **Test** al romper la variable, así que no premia
     cortes que solo sirven para memorizar.

### Antonella — Frontend & UI/UX + QA & Validation Engineer

1. **Tests de no fuga:**
   - `test_features_no_incluyen_variables_con_fuga`, `test_matriz_rechaza_fuga_explicita`.
   - `test_cv_agrupado_no_parte_busquedas`.
2. **El umbral se elige en Train, nunca en Test.** Con el umbral por defecto de 0,5, el F1 es 0 en
   logística, CART y RF (Entrada 3).
3. **Reliability diagrams:** Naive Bayes está fuera de la diagonal; logística y RF están cerca.
4. **Reproducibilidad:** semilla 42 en todas las búsquedas, `n_jobs=-1`, y verificación de que el
   pipeline reutilizable da exactamente la misma matriz que los pasos del notebook.

---

## Fase 4 — Inferencia causal (12/11/2026; pendiente de revisión de Verónica)

### Verónica — Project Lead & Scrum Master

1. **Resultado principal:** el top-5 **sí** causa reservas. Con orden aleatorio, multiplica por
   **2,11** las chances de que el hotel sea el reservado de su lista (IC 95 % [1,90; 2,35]).
2. **El dato del algoritmo engaña:** el ingenuo da 5,38 y, ajustado por todo lo observable, 3,38.
   Sigue un 60 % arriba del benchmark: es el sesgo por lo que el ranker sabe y nosotros no (U).
3. **Negocio:** no valuar el slot con datos del algoritmo. Priorizar hoteles baratos para su
   búsqueda y bien ubicados (allí el efecto es el doble). Mantener una fracción de tráfico con orden
   aleatorio como auditoría.
4. **Ensayo:** la posición es la variable más predictiva del RF, pero solo en parte es una palanca;
   el resto es selección.

### Lucía — Backend & ML / Causal Engineer

1. **Backdoor:** con el algoritmo, $Z$ bloquea $T \leftarrow Z \to Y$ pero no $T \leftarrow U \to Y$.
   Con orden aleatorio alcanza el conjunto vacío. Lo demuestran los tests de d-separación.
2. **DoWhy en 4 pasos:**
   - Model: DAG con el clic como mediador.
   - Identify: Backdoor.
   - Estimate: regresión 0,0961 e IPW 0,0896, contra un ingenuo de 0,1055.
   - Refute: tres pruebas, todas pasan.
3. **Por qué el logit condicional:** es un modelo de elección dentro de la búsqueda. El OR es
   invariante a la selección de listas por su resultado (lo prueba una simulación con verdad
   conocida), mientras que la diferencia absoluta se infla.
4. **El clic no es un mediador sustantivo:** no hay reserva sin clic, así que "ajustar por el clic"
   anula el efecto casi por construcción.
   - Lo informativo es la identidad del embudo RR = RR_clic × RR_conversión|clic.
   - Con orden aleatorio: 2,96 × 1,30 = 3,86; el 80 % del efecto (escala log) son más clics.
   - El residuo de 0,004 al ajustar por el clic no es un artefacto: es la diferencia de conversión
     entre quienes hicieron clic (0,078) por la tasa de clic.
4 bis. **Robustez de la conclusión (4.3 bis):**
   - Balance: la mayor |SMD| pasa de 0,248 a 0,005 al re-ponderar, y el cociente de OR casi no
     cambia (1,60).
   - En escala de diferencia de riesgos, que es colapsable: cociente de efectos marginales 1,65
     [1,41; 1,98].
   - Bootstrap por búsqueda con 1000 réplicas: cociente de OR [1,42; 1,80]. Ninguna réplica ≤ 1.
   - Wald z = 7,92.
   - Redacción correcta: "los datos son incompatibles con que el ajuste por lo observable elimine
     el sesgo", no "se rechaza Backdoor".
5. **Colisionadores:**
   - La selección del dataset (≥ 1 clic) es un colisionador natural.
   - C simulado sobre orden aleatorio: condicionar en C induce Berkson entre exposición y calidad.
6. **Refutaciones:** las tres pasan, y eso indica **estabilidad**, no ausencia de confusión. Un
   confusor simulado moderado (5 %) reduce el ATE un 48 %. La prueba que detecta la confusión es el
   benchmark.
7. **DML:**
   - Cross-fitting agrupado por búsqueda, con RF como modelos de ruido y ortogonalidad de Neyman.
   - ATE 0,083 con el algoritmo y 0,007 con orden aleatorio.
   - CATE (Causal Forest): mayor con precio relativo bajo y buena ubicación; plano en estrellas.

### Antonella — Frontend & UI/UX + QA & Validation Engineer

1. **`tests/test_causal.py`:**
   - El logit condicional recupera el OR verdadero, también después del filtro S.
   - La diferencia absoluta se infla con S.
   - Condicionar en C induce Berkson.
   - DoWhy identifica Backdoor sin el mediador y recupera el efecto.
2. **Errores de la IA atrapados en esta fase:**
   - dowhy incompatible con networkx 3.6 (error 26).
   - Colinealidad intra-búsqueda (error 27).
   - Logit condicional no convergido: ahora lanza un error (error 28).
3. **Semáforo de la app:** verde = las tres refutaciones pasan. Leyenda honesta: mide estabilidad, no
   identificación.

---

## Fase 5 — App, informe y entrega (12/11/2026; pendiente de revisión de Verónica)

### Verónica — Project Lead & Scrum Master

1. **Entregables:**
   - App desplegada (link en el README).
   - Notebook ejecutado de punta a punta.
   - Informe Quarto (HTML y PDF) y slides en PDF.
   - BITÁCORA con 5 entradas y log de IA.
2. **Confidencialidad:**
   - La app no ve datos, solo agregados con supresión de celdas menores a 10 filas.
   - El modelo de la app es un RF con hojas de al menos 100 filas.
   - Riesgo residual (los parámetros del modelo) asumido y documentado.
3. **Flujo Scrum:** ramas por fase, PRs con veredicto QA, merge con "Create a merge commit", CI con
   ruff y pytest.

### Lucía — Backend & ML / Causal Engineer

1. **Serialización:** `models/rf_app.joblib` (≈ 5,5 MB, scikit-learn 1.5.2 fijado en
   `app/requirements.txt`) y `models/artefactos_app.json`.
2. **Sin reentrenar en vivo:** la grilla de CATE está precalculada, porque EconML no corre en la
   app.
3. **`models/resultados.json`:** fuente única de los números del informe y las slides; el notebook
   lo regenera.

### Antonella — Frontend & UI/UX + QA & Validation Engineer

1. **Tres módulos:**
   - **Diagnóstico:** filtros, histogramas agregados, CTR por posición, plano PCA de Train con
     proyección de un caso.
   - **What-Is:** probabilidad del RF, dispersión entre árboles, explicación local ceteris paribus
     y calibración.
   - **What-If:** DAG, ATE, OR observacional vs. benchmark, CATE por perfil y semáforo de
     refutaciones.
2. **`tests/test_app.py`:**
   - Sin identificadores.
   - Ninguna celda publicada describe menos de 10 filas.
   - El modelo tiene hojas de al menos 100 filas y pesa menos de 50 MB.
   - Las entradas fuera de rango se recortan.
   - La app corre sin excepciones (`AppTest`).
3. **UX:**
   - Paleta validada; texto que explica la diferencia entre predecir e intervenir.
   - El aviso del What-Is: "esto es una predicción, no un efecto causal".

---

## Preguntas de práctica (respondé sin mirar; la respuesta está en la fuente indicada)

### Verónica — Project Lead & Scrum Master
1. ¿Por qué el top-5 "vende" y a la vez el dato del algoritmo lo sobreestima? (notebook 4.3 y 4.3 bis)
2. ¿Por qué la muestra se arma por búsquedas y no por filas? (notebook 1.1)
3. ¿Qué riesgo de licencia asumieron y cómo lo mitigaron? (BITÁCORA, Entrada 1)
4. ¿Cuánto vale un slot del top-5? ¿Por qué das un rango y no un número? (notebook 4.9, escenarios)
5. ¿Cómo funcionó Scrum en el equipo, honestamente? (BITÁCORA, Proceso Scrum)
6. ¿Qué es un experimento fallido de la Entrada 3 y qué aprendieron? (BITÁCORA, Entrada 3)

### Lucía — Backend & ML / Causal Engineer
1. Demostrá con el DAG por qué $Z$ no alcanza para Backdoor con el orden del algoritmo. (notebook 4.1)
2. ¿Qué es un logit condicional y por qué su OR sobrevive a la selección del dataset? (4.3; `test_or_condicional_sobrevive_al_filtro_de_seleccion`)
3. ¿Qué es la no colapsabilidad del OR y cómo la descartaron como explicación de la brecha? (4.3 bis b)
4. Si las búsquedas aleatorias fueran distintas de las del algoritmo, ¿qué harían? ¿Qué hicieron? (4.3 bis a)
5. ¿Por qué el clic no es un "mediador sustantivo"? ¿Qué descomposición usaron? (4.4)
6. ¿Qué miden las refutaciones de DoWhy y qué **no** miden? (4.6)
7. ¿Qué es la ortogonalidad de Neyman en DML? ¿Para quién rinde más el slot? (4.7)
8. ¿Por qué el MDI favorece variables continuas y la permutación no? (3.6)

### Antonella — Frontend & UI/UX + QA & Validation Engineer
1. ¿Cómo garantizan que ningún dato quede en el repo y que la app no exponga registros? (`test_confidencialidad.py`, `test_app.py`)
2. ¿Qué es un `xfail(strict=True)` y por qué es mejor que borrar el test? (`test_data_contract.py`)
3. Dame dos errores de la IA que atrapó un test y explicá el test. (BITÁCORA, Entrada 5)
4. ¿Qué pasa si el logit condicional no converge? (`tpi.causal`, guarda de convergencia)
5. ¿Qué muestra el semáforo de la app y qué no garantiza? (app, módulo What-If)
6. ¿Cómo evita el copiloto inventar cifras? ¿Qué **no** garantiza el guardarraíl? (`app/copiloto.py`, `tests/test_copiloto.py`)
7. ¿Cómo se reproduce todo desde cero? (README, "Reproducción paso a paso")
