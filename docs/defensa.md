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
   - Consultamos a la cátedra.
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
