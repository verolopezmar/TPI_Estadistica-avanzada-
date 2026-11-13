# Guía para la defensa oral, por rol

> Para: Verónica, Lucía y Antonella. Se completa al cierre de cada fase.
> En la ronda de preguntas, cada una defiende el trabajo de su rol. Todo número de esta guía sale
> de un output ejecutado; la fuente está entre paréntesis.

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
2. **Error de la IA atrapado:** la primera regla de outliers no marcaba un precio de 4,1 millones
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
