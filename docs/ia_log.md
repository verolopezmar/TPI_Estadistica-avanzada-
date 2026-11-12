# Registro de uso de IA (insumo de la Entrada 5 de la BITÁCORA)

Asistente: **Claude Code** (modelo Claude Opus 5.5, Anthropic), usado desde VS Code.
Alcance: el asistente propone código, textos y diseños y ejecuta lo que se le pide; **las
decisiones las toma Verónica López Marsico** y quedan registradas acá con su autoría. Ningún número
del informe se toma del texto del asistente: todo valor sale de un output ejecutado y versionado.

**Convención de autoría.** Cada decisión indica quién la propuso y quién la aprobó, p. ej.
"propuesto por Claude Code, aprobado por Verónica el 12/11". Todos los prompts los escribió
Verónica (en algunos casos con ayuda de otro asistente de IA para redactarlos; ver error 12).

**Organización real del trabajo (desde el 12/11/2026).** Verónica ejecuta y valida todo el
trabajo con asistencia de Claude Code, incluidas las revisiones de QA de los PRs. Los roles Scrum
del README se mantienen porque la defensa oral es por rol. Lucía Báez y Antonella Robiollio no
revisan ni validan PRs. Ver también la BITÁCORA (Roles Scrum).

Formato por fase: prompt (resumido) · qué produjo la IA · qué se decidió y quién ·
errores/alucinaciones detectados · test determinístico que los refuta o previene.

---

## Fase 0 — Setup y selección del dataset (11/11/2026)

**Prompt (Verónica, resumen).** Brief completo del TPI: leer enunciado y rúbrica, armar la estructura del
repo, verificar herramientas (Python, Quarto, TinyTeX), asignar roles Scrum y proponer 3-4
datasets reales para estimar el efecto causal de un recomendador, sin descargar datos hasta la
aprobación del equipo.

**Qué produjo.**
- Estructura de carpetas, `pyproject.toml` (ruff + pytest), `.gitignore`, `src/tpi/config.py`
  (SEED = 42, α = 0,05, rutas relativas), workflow de CI `quality-gate.yml`, plantilla de PR con
  veredicto QA, esqueletos de README y BITÁCORA, este log.
- Verificación del entorno: Python 3.11.9 y 3.13, Quarto 1.11.9 con TinyTeX 2026 OK; `gh` no
  instalado; `git` sin `user.name`/`user.email` global.
- Propuesta de 4 datasets candidatos con recomendación de Expedia.
- Propuesta de asignación de roles.

**Qué se decidió.** Dataset (Expedia) y roles: propuestos por Claude Code, aprobados por Verónica
el 11/11. Detalle en "Decisiones (12/11/2026)" más abajo.

**Errores / alucinaciones detectados y verificaciones pendientes.**
1. *Afirmación no verificada en fuente primaria.* La definición de `random_bool` del dataset de
   Expedia ("1 si el orden mostrado fue aleatorio, 0 si fue el orden del algoritmo") la aportó el
   asistente de memoria; la página de datos de Kaggle no pudo leerse sin sesión y las fuentes
   secundarias consultadas listan la columna sin definirla. La licencia tampoco se pudo confirmar
   (Kaggle muestra las reglas solo con sesión iniciada).
   **Test previsto** (`tests/test_data_contract.py`, al descargar): (a) `random_bool ∈ {0,1}` y
   constante dentro de cada `srch_id`; (b) en búsquedas con `random_bool = 1` la correlación de
   Spearman entre `position` y la calidad del hotel (`prop_review_score`, `prop_starrating`) no es
   significativamente distinta de 0, mientras que con `random_bool = 0` sí lo es. Si (b) falla, la
   premisa del benchmark experimental es falsa y se descarta el diseño.
2. *Atribución de cifras.* El primer resumen de búsqueda web atribuía el tamaño del dataset a "un
   paper" sin decir cuál. Antes de citarlo, el asistente verificó el texto completo: Liu et al. (2013,
   arXiv:1311.7679) dice textualmente "The training data contains 399,344 unique search lists and
   9,917,530 points". El mismo paper **no** define `random_bool` (solo lo lista en una tabla de
   importancia), lo que confirma el punto 1.
   **Test previsto:** `test_data_contract.py` registra N y número de `srch_id` del archivo real y
   los compara con esas cifras.

---

## Fase 0 (continuación) — Render, contrato de datos y DAG borrador (11/11/2026)

**Prompt (Verónica, resumen).** Dataset aprobado: Expedia (plan B: Online Shoppers semi-sintético). Roles
confirmados. Sin datos todavía: render de prueba de Quarto (¿rompe la ruta de OneDrive con
tilde?), script de descarga, contrato de datos, test de `random_bool` y DAG borrador. Sin
commits hasta tener `gh`.

**Qué produjo.** `report/informe.qmd` + `referencias.bib` (render HTML y PDF OK);
`src/tpi/data_contract.py`, `sampling.py`, `aleatorizacion.py`, `dag.py`;
`data/download_data.py`; `tests/test_environment.py`, `test_data_contract.py`, `test_dag.py`;
`docs/dag_borrador.md` + figura.

**Qué se decidió.** Dataset A (Expedia) y roles según la propuesta de Claude Code: aprobados por
Verónica. Informe Quarto independiente con el notebook como anexo: decisión de Verónica. El DAG
borrador (propuesta de Claude Code) quedó pendiente de revisión; la revisó y aprobó Verónica el
12/11 (ver "Decisiones").

**Errores / alucinaciones detectados.**
3. *Instrucción desactualizada del asistente (autenticación de Kaggle).* En la Fase 0 el
   asistente indicó guardar el token en `%USERPROFILE%\.kaggle\kaggle.json`. Al inspeccionar el
   paquete instalado (`kaggle==2.2.5`), su propio mensaje indica el esquema nuevo:
   `kaggle auth login` (OAuth), un token en `~/.kaggle/access_token` o la variable
   `KAGGLE_API_TOKEN`. **Corrección:** el docstring de `data/download_data.py` documenta el
   esquema vigente. **Prevención:** antes de escribir llamadas a la API, el asistente inspecciona
   las firmas reales (`inspect.signature`) en lugar de confiar en la memoria del asistente.
4. *Incompatibilidad silenciosa de dependencias.* pip instaló `pyarrow==26.0.1` sobre
   `numpy==1.26.4` sin advertir; `import pyarrow` falla con "pyarrow requires NumPy 2.0 or
   newer". El asistente había fijado el `requirements.txt` a partir de `pip freeze` sin probar
   los imports. **Corrección:** `pyarrow==19.0.1`. **Test:** `tests/test_environment.py`
   (importa cada dependencia compilada y hace un parquet ida y vuelta).
5. *Bloqueo del sistema operativo.* Smart App Control (Windows 11) bloqueó la DLL de
   `pyzmq==27.3.0`, así que no arrancaba ningún kernel de Jupyter (notebook ni Quarto).
   Diagnóstico con el log de CodeIntegrity (eventos 3033/3077). **Corrección:** `pyzmq==26.4.0`,
   que carga sin desactivar la protección. **Test:** `test_modulo_importa[zmq]`.
6. *Test de equivalencia sin potencia (error de diseño del asistente).* El test sintético de
   "orden aleatorio" falló con 200 búsquedas: ρ̄ = 0,0044, pero el IC 95 % [−0,027; 0,036] no
   cabe en el margen ±0,03. No era un error de la lógica: un test de equivalencia necesita N
   suficiente para que el IC sea más angosto que el margen. **Corrección:** 2.000 búsquedas en la
   fixture. Con las 15.000 búsquedas reales por estrato, el IC esperado es ≈ 9 veces más angosto.
7. *Figura engañosa.* En la primera versión del DAG la flecha Zu → T pasaba por detrás del nodo
   R y parecía un arco Zu → R inexistente (y Y → C parecía salir de I). Se detectó al revisar
   la imagen y se corrigieron las posiciones. **Prevención:** la fuente de verdad del DAG es la
   lista `ARCOS` y los tests de d-separación, no la figura.

---

## Interrupción de sesión y retoma (12/11/2026)

**Prompt (Verónica, resumen).** La sesión anterior del asistente se cerró. Se pide releer enunciado,
rúbrica, BITÁCORA, README y este log; revisar git; informar fase, trabajo terminado, trabajo a
medias y decisiones pendientes, **sin avanzar** hasta la confirmación del equipo.

**Qué hizo la IA.** Solo lectura y verificación: `git status` limpio en
`feat/fase0-contrato-datos` (6 commits, sincronizada con `origin`, sin PR a `main`); suite
ejecutada en el venv `~/.venvs/tpi-eaymr` (Python 3.11.9): 42 passed, 7 skipped (los tests de
datos se saltean porque `data/raw/` está vacío); `ruff check .` sin errores.

**Riesgo de la interrupción y cómo se mitigó.** El asistente no conserva memoria entre sesiones:
el estado se reconstruyó **solo** a partir de archivos versionados y de la ejecución de los
tests, no de recuerdos del asistente. Lo que no estaba escrito (por ejemplo, la ubicación del
venv fuera de OneDrive) se verificó en disco antes de afirmarlo.

**Inconsistencias detectadas al retomar.**
8. *Documentación desalineada con el entorno real.* El README indica crear `.venv` dentro del
   repo, pero el entorno que funciona está en `~/.venvs/tpi-eaymr` (fuera de OneDrive). El
   Python global no tiene `pytest` ni `ruff`. **Pendiente:** decidir cuál es el procedimiento
   oficial y corregir el README.
9. *Plan vs. ejecución.* En la entrada anterior figura "sin commits hasta tener `gh`", pero hay
   5 commits del 12/11 00:11 ya pusheados y `gh` sigue sin instalarse. No es un error de código,
   pero hay que registrar quién lo decidió.

### Resolución de las inconsistencias (12/11/2026)

8. **Resuelta (decisión de Verónica, 12/11).** Procedimiento oficial: venv fuera de carpetas
   sincronizadas (`~/.venvs/tpi-eaymr`), instalado desde `requirements.txt`. El README lo
   corrigió Claude Code a pedido de Verónica. Motivo: OneDrive bloquea y corrompe archivos del venv.
9. **Registrada tal cual, sin reescribir el historial.** Los 5 commits de
   `feat/fase0-contrato-datos` (12/11/2026 00:11) se hicieron y pushearon antes de instalar `gh`,
   contra lo planificado. No pasaron por un PR, pero tampoco llegaron a `main`, así que la regla
   de oro (ningún merge a `main` sin firma del QA) no se violó. Decisión de Verónica: no
   reescribir el historial (hacerlo sobre una rama pública sería peor que el desvío) y abrir el
   PR de la Fase 0, para auditar los 5 commits juntos antes del merge. La auditoría la hace
   Verónica (ver "Organización real del trabajo").

### Decisiones (12/11/2026)

- **Dataset:** Expedia "Personalize Expedia Hotel Searches — ICDM 2013" (Kaggle). Plan B:
  Online Shoppers con tratamiento semisintético. Propuestos por Claude Code, aprobados por
  Verónica.
- **Roles (nominales, para la defensa oral):** Verónica López Marsico = Project Lead & Scrum
  Master; Lucía Báez = Backend & ML/Causal Engineer; Antonella Robiollio = Frontend & UI/UX + QA.
  Propuesta de Claude Code, aprobada por Verónica. Desde el 12/11 todo el trabajo y las
  validaciones los hace Verónica (decisión de Verónica).
- **Informe:** Quarto independiente, con `TP_Integrador_Final.ipynb` como anexo (decisión de
  Verónica).
- **DAG** (borrador de Claude Code; definiciones aprobadas por Verónica el 12/11):
  - T binaria = hotel en el top-5; posición continua como sensibilidad. Propuesto por Claude
    Code, aprobado por Verónica.
  - Y principal = `booking_bool` (propuesto por Claude Code, aprobado por Verónica). Usar el
    gasto, que solo existe si hubo reserva, como ejemplo de la trampa de selección/colisionador:
    decisión de Verónica sobre una observación del borrador de Claude Code.
  - Zc: criterio fijado por Verónica (confusor si el ranking la usa **y** afecta la reserva;
    predictor si solo afecta Y). La clasificación empírica la hace Claude Code en la Fase 1 y la
    revisa Verónica.
  - Colisionador natural si hay uno defendible, más el simulado como validación: decisión de
    Verónica.
  - `click_bool` como mediador natural (posición → clic → reserva): estaba en el borrador de
    Claude Code; Verónica pidió evaluarlo como trampa del mediador.
  - `random_bool` como experimento natural: propuesto por Claude Code en la Fase 0, aprobado por
    Verónica con la condición de verificarlo antes en los datos y en la documentación.
- **Entorno oficial:** venv en `~/.venvs/tpi-eaymr` (decisión de Verónica; ver punto 8).
- **Historial de git:** no se reescribe (decisión de Verónica; ver punto 9).

---

## Fase 0 (cierre) — Descarga, verificación de `random_bool` y confidencialidad (12/11/2026)

**Prompt (Verónica, resumen).** Autenticar Kaggle y descargar; verificar `random_bool` y la licencia;
correr los 7 tests de datos. Luego, con la definición de `random_bool` y las reglas copiadas
por Verónica de la pestaña Data/Rules, con su sesión de Kaggle: citar la fuente primaria, analizar el sesgo de selección
del dataset ("most of the data are for searches that resulted in a purchase"), mitigar el riesgo
de confidencialidad (las reglas prohíben compartir datos y "resulting model parameters") y dejar
el plan B listo para activar sin rehacer la estructura.

**Qué produjo la IA.**
- `src/tpi/config.py`: crudo fuera del repo y de OneDrive (`~/data/tpi-eaymr/raw`, configurable
  con `TPI_RAW_DIR`); manifiesto versionado en `data/manifiesto.json`.
- `data/download_data.py`: descarga solo `data.zip`, extrae solo `train.csv` (Deflate64 vía
  `inflate64`, con verificación de tamaño y CRC-32) y registra el SHA-256 del zip y del csv.
- `tests/test_confidencialidad.py` (falla si git trackea archivos de datos) y
  `test_crudo_fuera_del_repo`.
- `src/tpi/contrato.py` (contrato causal genérico; `TPI_DATASET` elige el dataset),
  `src/tpi/plan_b.py` (Online Shoppers semisintético con efecto verdadero conocido),
  `data/download_plan_b.py`, `docs/plan_b.md`, `tests/test_contrato.py`.
- Diagnóstico de la premisa de `random_bool` y del mecanismo de selección del dataset (cifras
  abajo, todas de outputs ejecutados).

**Resultados verificados.**
- `train.csv`: 9.917.530 filas y 399.344 búsquedas, coinciden exactamente con Liu et al. (2013)
  (`test_tamano_train_coincide_con_lo_reportado` pasa). 277.798 búsquedas con R = 0 y 121.546
  con R = 1; ninguna con `random_bool` inconsistente. Contrato de datos sin violaciones.
- Muestra: 30.000 búsquedas completas (15.000 por estrato), 739.323 filas.

**Qué se decidió y quién.**
- Citar la definición de `random_bool` de la fuente primaria: pedido de Verónica.
- R = 1 como **cuasi-experimento ajustado** (efectos fijos por búsqueda + ajuste por Z,
  reportando sin ajuste vs. ajustado): **propuesto por Claude Code, aprobado por Verónica el
  12/11** (opción recomendada entre tres presentadas, tras la refutación parcial del error 10).
- **Escala relativa dentro de la búsqueda como principal** (risk ratio / log-odds) y, como
  robustez, ambos estratos restringidos a búsquedas con reserva (ATE absoluto); `click_bool` como
  resultado secundario: **propuesto por Claude Code, aprobado por Verónica el 12/11**.
- Análisis del sesgo de selección del dataset: planteado por Verónica (puntos a, b, c); el
  hallazgo del filtro "≥ 1 clic" y la derivación del risk ratio son de Claude Code, revisados por
  Verónica.
- Mitigaciones de confidencialidad (nada de datos en el repo con test que lo garantice, crudo
  fuera de OneDrive configurable por variable de entorno, app solo con agregados, plan B listo):
  definidas por Verónica, implementadas por Claude Code. Consulta a la cátedra: la envía Verónica
  el 12/11. Si el 14/11 no hay respuesta o es negativa, se activa el plan B (decisión de
  Verónica).
- Descargar solo `data.zip` y usar solo `train.csv`: decisión de Verónica. Elegir `inflate64` para
  Deflate64: Claude Code.

**Errores / alucinaciones detectados.**
10. *Premisa experimental refutada en parte.* El asistente propuso en la Fase 0 usar R = 1 como
    "RCT puro". La fuente primaria define `random_bool` como "the displayed sort was random", pero
    en los datos el orden aleatorio **no** es independiente de los atributos: ρ̄ intra-búsqueda
    entre posición y `prop_starrating` = −0,0743 (IC 95 % [−0,0795; −0,0690]) y con
    `prop_location_score2` = −0,0651 ([−0,0713; −0,0589]), fuera del margen ±0,03. Bajo el
    algoritmo son −0,2564 y −0,2992. Descartamos la selección por compra como causa: persiste en
    búsquedas sin reserva ni clic (−0,0698 y −0,0629). Una explicación compatible, que **no
    podemos verificar**, es que ~13 % de las posiciones mostradas faltan en ambos estratos
    (cobertura media `count/max` = 0,870 en R = 1 y 0,876 en R = 0): si el proveedor removió
    filas de forma no aleatoria, la permutación observada deja de ser uniforme.
    **Test:** `test_random_bool_1_orden_independiente_de_atributos` queda como
    `xfail(strict=True)` en esas dos variables (si alguna vez pasa, pytest avisa), y el nuevo
    `test_random_bool_1_reduce_la_asociacion_del_algoritmo` exige que R = 1 reduzca al menos a la
    mitad la asociación del algoritmo (umbral **post hoc**, declarado como tal).
11. *Supuesto falso sobre la estructura de la descarga.* El script de la Fase 0 pedía
    `train.csv` suelto; la competencia publica `data.zip` + 3 zips de benchmarks. Además el zip
    usa Deflate64, que `zipfile` no soporta (`NotImplementedError`). `zipfile-deflate64` no tiene
    wheel para Windows/Python 3.11 y exige compilar con MSVC. **Corrección:** `inflate64==1.0.4`
    (wheel, sin bloqueo de Smart App Control). **Test:** `test_modulo_importa[inflate64]` y
    verificación de CRC-32 en cada extracción.
12. *Error de transcripción de otro asistente de IA.* El prompt traía el tamaño de `data.zip`
    como 434.537.097 bytes. La cifra la transcribió mal el asistente de IA que ayudó a Verónica a
    armar los prompts (no Claude Code). La API de Kaggle y el archivo descargado dicen
    **434.573.097** (dos dígitos traspuestos); Claude Code lo detectó al listar los archivos de la
    competencia. **Prevención:** el manifiesto registra tamaños y hashes del archivo real y nunca
    se copian cifras a mano de un texto generado por IA.
13. *Error del asistente en su propio diagnóstico.* Al resumir el plan B escribió `d.T` para la
    columna de tratamiento `"T"`: en pandas `d.T` es la **transpuesta**, y el resultado salió
    `nan` sin error. Lo detectó el propio output. **Corrección:** la columna se llama
    `expuesto`. **Test:** `test_roles_no_chocan_con_atributos_de_dataframe`.
14. *Descripción de la fuente incompleta (no es un error del asistente, pero cambia el
    análisis).* Kaggle dice que "most of the data are for searches that resulted in a purchase".
    En el `train.csv` completo eso vale **solo para R = 0** (93,86 % de las búsquedas con reserva
    vs. 13,04 % en R = 1). Además encontramos un criterio no documentado: **ninguna** de las
    399.344 búsquedas tiene 0 clics, y ninguna tiene más de una reserva. El dataset está
    seleccionado sobre el mediador (≥ 1 clic) en ambos estratos y, en R = 0, sobre el resultado.
    Análisis en la Entrada 4 de la BITÁCORA. **Test:** `test_mecanismo_de_seleccion_del_dataset`.

---

## Fase 1 — Inferencia clásica (inicio, 12/11/2026)

**Prompt (Verónica, resumen).** Arrancar la Fase 1 en una rama nueva mientras se revisa la Fase 0.

**Qué produjo la IA.** `src/tpi/inferencia.py` (Welch, Mann-Whitney con r biserial y
Hodges-Lehmann, t pareado con Wilcoxon de resguardo, potencia y efecto mínimo detectable, ANOVA
clásico + Levene/Brown-Forsythe + Welch + Kruskal-Wallis, post-hoc Tukey vs. Games-Howell vs.
Bonferroni) y `tests/test_inferencia.py`. `Contraste.texto()` aplica las reglas de reporte del TPI
(estadístico, gl, p exacto, IC 95 %, tamaño de efecto, "no se rechaza H0").

**Diseño propuesto por Claude Code, aprobado por Verónica el 12/11.** Contrastes sobre unidades independientes:
(1) pareado dentro de la búsqueda con R = 1, CTR del top-5 vs. del resto (sesgo de posición, arco
T → M del DAG); (2) gasto por reserva, usuarios recurrentes vs. nuevos (Welch + Mann-Whitney);
ANOVA: log del precio por noche del hotel reservado según categoría de estrellas.

**Errores / alucinaciones detectados.**
15. *API recordada de una versión vieja.* El asistente usó los nombres de columnas de pingouin
    `p-unc`, `p-val` y `CI95%`; en `pingouin==0.6.1` son `p_unc`, `p_val` y `CI95`. Fallaron 3
    tests con `KeyError`. **Corrección:** se inspeccionaron las columnas reales antes de
    corregir. **Test:** `tests/test_inferencia.py` compara cada resultado contra pingouin, así que
    un cambio de API rompe el test en lugar de producir un número equivocado.

### Fase 1 (continuación) — Auditoría, faltantes, competidores, contrastes y ANOVA (12/11/2026)

**Prompt (Verónica, resumen).** Reorganización: Verónica hace y valida todo. Aprueba los tres
contrastes propuestos y pide seguir con la auditoría de datos, los faltantes, las variables de
competidores, el notebook y la Entrada 2.

**Qué produjo la IA.** `src/tpi/auditoria.py` + `tests/test_auditoria.py` (centinelas, diagnóstico
de escala del precio, outliers, contraste de MCAR, imputación con indicador, agregados de
competidores); `src/tpi/graficos.py` (paleta validada); notebook
`notebooks/TP_Integrador_Final.ipynb`, Fase 1, ejecutado de punta a punta (fuente jupytext:
`TP_Integrador_Final.py`); figuras en `report/figures/f1_*.png`; Entrada 2 de la BITÁCORA.

**Qué se decidió y quién.**
- Los tres contrastes y el ANOVA: propuestos por Claude Code, **aprobados por Verónica el 12/11**.
- Clasificación de $Z_c$: criterio de Verónica; aplicación de Claude Code (`comp_n_info` y
  `comp_saldo` confusores, `comp_diff_max` descartada). **Pendiente de revisión por Verónica.**
- Precio relativo, reglas de outliers, tratamiento de centinelas y estrategia de imputación por
  mecanismo: propuestos por Claude Code. **Pendientes de revisión por Verónica.**
- T se mantiene como `position ≤ 5` **en pantalla** (definición aprobada), documentando que en
  general contiene 4 hoteles del dataset (error 16). Interpretación de Claude Code, **a confirmar
  por Verónica**.

**Errores / alucinaciones / hallazgos.**
16. *Hallazgo que corrige una hipótesis del asistente.* En el error 10 el asistente sugirió que el
    ~13 % de posiciones faltantes podía deberse a una remoción no aleatoria de filas. El análisis
    por posición muestra otra cosa: las posiciones 5, 11, 17 y 23 aparecen en solo 6,2-7,6 % de
    las búsquedas con R = 0 y 1,4-1,7 % con R = 1, contra ≈ 99 % de sus vecinas. Son slots fijos
    ocupados por listados que no están en el dataset. Un hueco estructural **no** explica la
    asociación residual con R = 1, que queda sin explicación. **Test:**
    `test_slots_estructurales_cada_seis_posiciones`.
17. *Regla de outliers incompleta (error del asistente).* La primera regla (precio > 10× la mediana
    de la búsqueda) no marcó precios de millones de USD por noche, porque toda esa búsqueda tenía precios
    absurdos y su mediana también. Lo detectó el propio asistente al comparar el máximo antes y
    después. **Corrección:** se agregaron cercas extremas de Tukey sobre log(precio). **Test:**
    `test_outlier_absoluto_detecta_busquedas_enteras_imposibles`.
18. *Formato que violaba la regla del "p-valor exacto".* `Contraste.texto()` imprimía
    `t(1.498e+04)` y `p = 0`. Un p = 0 es un subdesborde de doble precisión, no un p-valor nulo.
    **Corrección:** gl sin notación científica y "p < 1e-300 (subdesborde de doble precisión)".
    **Test:** `test_texto_formatea_gl_y_p_sin_notacion_engañosa`.
19. *Omisión en un texto interpretativo (error del asistente).* La primera redacción del EDA decía
    que, fuera del clic, las asociaciones con la reserva eran débiles, y omitía la posición
    (ρ = −0,14 en la matriz). Se detectó al revisar la figura contra el texto. **Prevención:**
    cada número del texto se coteja con un output del notebook antes de commitear.
20. *Semántica no verificada.* El signo de `compN_rate` (si "+1" significa que Expedia es más
    barata que el competidor) proviene de la memoria del asistente, no de la fuente primaria.
    **Mitigación:** la clasificación de $Z_c$ usa agregados que no dependen del signo.
    **Pendiente:** Verónica transcribe la definición textual de la pestaña Data de Kaggle.

### Revisión de Verónica del cierre de la Fase 1 (12/11/2026)

**Prompt (Verónica, resumen).** Confirma T; aprueba la auditoría; pide contrastar $Z_c \to T$ antes
de llamar confusor a $Z_c$ y actualizar el DAG; transcribe las definiciones de `compN_*` de Kaggle;
decide no consultar a la cátedra; pide PRs por fase y seguir con la Fase 2.

**Qué se decidió y quién.**
- T = `position ≤ 5` en pantalla: **confirmado por Verónica**. Sensibilidad `top4_dataset`
  (top-4 del ranking entre los hoteles del dataset): **pedida por Verónica**, implementada por
  Claude Code en `tpi.contrato` (coincide con `top5` en el 99,89 % de las filas).
- Slots fijos 5/11/17/23 como característica del DGP y la asociación residual con R = 1 como
  **limitación declarada sin hipótesis forzada**: decisión de Verónica (BITÁCORA, Entrada 4, B y C).
  Se retira la hipótesis de "remoción no aleatoria de filas" que el asistente había sugerido en el
  error 10 (refutada por el error 16).
- Precio relativo a la búsqueda, dos reglas de outliers con su test e imputación (MAR por grupo con
  indicador, MNAR estructural sin imputar): **propuestos por Claude Code, aprobados por Verónica
  el 12/11**.
- $Z_c$: Verónica señaló que la primera clasificación mostraba solo $Z_c \to Y$. Claude Code agregó
  el contraste de $Z_c \to T$ (diferencia pareada top-5 vs. resto por estrato e interacción
  $Z_c \times R$ en el logit de T, condicional a $Z_h$). Resultado: `comp_saldo` y `comp_n_info`
  siguen siendo confusoras (interacción p = 2,7·10⁻¹⁸ y 2,8·10⁻⁵); `comp_diff_max` se excluye. DAG
  actualizado (Zc → T, Zc → M, nodo de selección S) con tests de d-separación nuevos.
  **Pendiente de revisión por Verónica.**
- Confidencialidad: **Verónica decide no consultar a la cátedra** y seguir con Expedia con las
  mitigaciones implementadas; el plan B queda disponible y no se activa. Se retira el hito del
  14/11 del calendario de riesgos.

**Errores / alucinaciones detectados.**
20. *(Resuelto) Semántica de memoria.* La versión anterior del texto y de los docstrings no fijaba
    el signo de `compN_rate`, y la única noción de signo que tenía el asistente venía de su memoria,
    no de la fuente. Verónica transcribió la definición de Kaggle: +1 = Expedia **más barata** que
    el competidor N; `rate_percent_diff` es **absoluta**. La agregación `comp_saldo` ya usaba
    (+1) − (−1), así que el cálculo no cambia; se corrigieron la documentación y la lectura
    (`comp_saldo` > 0 = Expedia más barata). **Test:** `test_agregado_de_competidores_ignora_diferencias_imposibles`
    fija la aritmética del saldo.
21. *Dominio de `compN_inv` tomado de memoria.* El contrato admitía {−1, 0, 1} para `compN_inv`; la
    definición de Kaggle solo documenta {+1, 0, NULL}. En los datos hay −1 en el 1,27 % de los
    valores no nulos, y el contrato lo aceptaba sin avisar. **Corrección:** se documenta como valor
    no documentado y no se interpreta (`comp_inv` no entra en $Z_c$). **Test:**
    `test_comp_inv_tiene_un_valor_no_documentado`. Además, `rate_percent_diff` nunca baja de 2 %:
    probable umbral de registro, no documentado.
22. *Afirmación de aprobación no ocurrida (detectada por el asistente antes de commitear).* Al
    redactar la nueva clasificación de $Z_c$, el asistente escribió "revisada y aprobada por
    Verónica", cuando Verónica todavía no la había revisado. Se corrigió a "pendiente de revisión".
    **Prevención:** una aprobación solo se registra si consta en un mensaje de Verónica.

### GitHub: PRs no abiertos desde el entorno del asistente (12/11/2026)

Verónica pidió abrir el PR de la Fase 0 (y luego uno por fase). Claude Code no pudo: `gh` no tiene
sesión en su entorno, aunque sí en la terminal de Verónica. El clasificador de permisos de Claude
Code bloqueó dos alternativas: correr `gh` fuera del sandbox y pasarle a `gh` la credencial que git
usa para los pushes (lo marcó como "exploración de credenciales"). El asistente no intentó otras
vías. **Quién abre los PRs:** Verónica, con los scripts preparados por Claude Code
(`github_setup.sh`, que crea etiquetas, milestones, issues y el PR de la Fase 0, y `pr_fase1.sh`).
Los PRs registran que la revisión QA la hace Verónica.

---

## Fase 2 — Gauss-Markov, influyentes, VIF y PCA sin fuga (12/11/2026)

**Prompt (Verónica, resumen).** Seguir con la Fase 2 en una rama nueva, con el mismo formato de
cierre, y actualizar `docs/defensa.md`.

**Qué produjo la IA.** `src/tpi/particion.py` (80/20 agrupada por búsqueda, estratificada por
`random_bool`), `src/tpi/regresion.py` (diagnósticos de Gauss-Markov, influencia, VIF, PCA solo en
Train), `tests/test_fase2.py`, la sección Fase 2 del notebook (figuras `report/figures/f2_*.png`),
la Entrada 2 (Fase 2) y la Entrada 3 de la BITÁCORA, y `docs/defensa.md` (Fase 2).

**Qué se decidió y quién.** Todo el diseño de la Fase 2 lo propuso Claude Code y está **pendiente
de revisión por Verónica**:
- Partición única 80/20 agrupada y estratificada.
- MCO de log precio por noche sobre las reservas.
- HC3 como remediación y Huber como resguardo.
- RANSAC con umbral 3σ̂.
- No eliminar influyentes.
- 2 PCs para la visualización y 4 como features alternativas.

**Errores / alucinaciones detectados.**
23. *Parámetro por defecto no auditado (error del asistente).* Claude Code usó `RANSACRegressor`
    con su umbral por defecto (MAD de *y*): 55,02 % de inliers y coeficientes con signo invertido.
    **Corrección:** umbral 3σ̂ robusto de los residuos (89,03 % de inliers). Registrado como
    experimento fallido (Entrada 3).
24. *Anticipación no confirmada (del asistente).* En la Fase 1 el asistente escribió que las
    correlaciones calidad-precio "anticipan multicolinealidad". Los VIF (máx 2,98) la refutan. Se
    corrigió en el notebook (2.4) y se registró en la Entrada 3.
25. *Incumplimiento de la regla propia de reporte (detectado por el asistente).* Los prints de
    Jarque-Bera y del F global mostraban "p = 0"; el formateador `_p` del módulo de inferencia no
    se estaba usando ahí. **Corrección:** todos los p de la Fase 2 pasan por `formato_p`
    ("p < 1e-300 (subdesborde de doble precisión)").

### Aprobaciones de Verónica (12/11/2026, revisión de las Fases 1 y 2)

**Propuestos por Claude Code, aprobados por Verónica el 12/11:**
- Fase 1: clasificación de competidores (`comp_saldo` y `comp_n_info` confusores, `comp_diff_max`
  excluida); DAG con Zc → T, Zc → M y nodo S; corrección del signo de `compN_rate`;
  `compN_inv = −1` tratado como faltante con indicador (`comp_inv_no_documentado`) y declarado en la
  auditoría.
- Fase 2: partición 80/20 por búsqueda estratificada por `random_bool` (unidad = búsqueda, sin fuga
  intra-búsqueda; no es un problema de pronóstico temporal); no eliminar influyentes; HC3 como
  principal y Huber de respaldo; 2 PCs para la app y 4 como features alternativas.

**Cambio de ritmo (decisión de Verónica, 12/11):** terminar las Fases 3, 4 y 5 en el día. Claude Code
decide con criterio y registra cada decisión como "propuesto por Claude Code"; Verónica revisa todo
junto al final. Solo se frena ante acciones físicas (login, deploy) o decisiones que cambien la
conclusión causal. Los scripts de GitHub pasan a `scripts/github/` (Git Bash); Verónica abre y
mergea los PRs en orden ("Create a merge commit"). Ya existe el PR #38 (Fase 0).

---

## Fases 3, 4 y 5 (12/11/2026)

**Prompt (Verónica, resumen).** Terminar el TP completo hoy: torneo predictivo, inferencia causal
completa, app con solo agregados, deploy listo, notebook ejecutable, informe, README, BITÁCORA,
slides, guía de defensa y auto-auditoría.

**Qué produjo la IA.** `src/tpi/preparacion.py` (pipeline de la Fase 1 en una función, verificada
contra los pasos del notebook), `modelos.py`, `causal.py`, `artefactos.py`, `app/` (app Streamlit +
lógica testeable), los tests correspondientes (`test_modelos.py`, `test_causal.py`, `test_app.py`),
las secciones 3 a 5 del notebook, los artefactos de `models/`, el informe, las slides, el README y
`scripts/github/abrir_prs_pendientes.sh`.

**Decisiones propuestas por Claude Code (pendientes de revisión por Verónica):**
- *Fase 3:*
  - Y = `booking_bool` por fila; PR-AUC como métrica de selección (prevalencia ≈ 2 %).
  - `StratifiedGroupKFold(5)` por búsqueda.
  - Excluir `click_bool` y el gasto por fuga; `position` sí entra.
  - Búsqueda de hiperparámetros en una submuestra del 25 % de las búsquedas de Train (Grid para
    logística y CART, `RandomizedSearchCV` con 20 combinaciones para RF, `n_jobs=-1`), con
    reentrenamiento en todo Train.
  - Umbral de clasificación elegido en Train (máximo F1 fuera de pliegue).
  - Holdout cronológico (último 20 % de las búsquedas por fecha) como sensibilidad, a pedido de
    Verónica.
  - Métrica extra de ranking (¿el hotel reservado está en el top-5 del modelo?).
- *Fase 4:*
  - Logit condicional por búsqueda como estimador principal (escala relativa) y MPL con efectos fijos
    como escala absoluta.
  - DoWhy con DAG de columnas reales (mediador explícito).
  - Ajuste intra-búsqueda solo con variables de hotel (las de búsqueda las absorbe el efecto fijo).
  - Refutaciones con 20 simulaciones, más la sensibilidad a un confusor no observado.
  - DML con 6.000 búsquedas por estrato (RF como modelos de ruido, cross-fitting agrupado por
    búsqueda).
  - Colisionador C simulado sobre datos reales con R = 1.
  - Simulación del filtro S con verdad conocida.
- *Fase 5:*
  - App con artefactos agregados (supresión de celdas < 10) y RF compacto (hojas ≥ 100 filas), sin
    EconML en vivo (grilla de CATE precalculada).
  - `app/requirements.txt` mínimo.
  - `models/resultados.json` como única fuente de los números del informe y las slides (expresiones
    inline de Quarto).

**Errores / alucinaciones detectados.**
26. *Incompatibilidad silenciosa de dependencias (del `requirements.txt` armado por el
    asistente).* `dowhy==0.12` llama a `networkx.algorithms.d_separated`, que no existe en
    `networkx==3.6.1`: la identificación fallaba con `AttributeError`. El test de imports no lo
    detectaba. **Corrección:** `networkx==3.4.2` (tiene esa función y también `is_d_separator`).
    Además, el puente `nx_pydot.from_pydot` es incompatible con `pydot 4`, así que el grafo de
    DoWhy se construye directo en networkx. **Test:**
    `test_dowhy_identifica_backdoor_sin_mediador_y_recupera_el_efecto`.
27. *Colinealidad introducida por el asistente.* El prototipo del ajuste intra-búsqueda incluía
    `log_precio` y `log_precio_rel` juntos. Dentro de una búsqueda difieren solo en la mediana de la
    lista, que absorbe el efecto fijo: la matriz quedaba de rango incompleto (`SingularMatrixWarning`).
    **Corrección:** `log_precio` solo en el ajuste agrupado (DoWhy).
28. *Estimaciones no convergidas casi usadas como resultado (error del asistente).* El logit
    condicional del prototipo emitió `ConvergenceWarning` y el asistente llegó a reportar sus OR
    como "primeros resultados". Se detectó al revisar el log. **Corrección:** covariables
    estandarizadas, BFGS con hasta 2000 iteraciones y una guarda que **lanza un error** si no
    converge, en lugar de devolver un número dudoso. Los valores convergidos son casi idénticos a
    los del prototipo (R = 1 ajustado: 2,113 vs. 2,108), pero solo se reportan los convergidos.
29. *Propensión sin converger dentro de DoWhy (detectado por el asistente al leer los outputs).* El
    estimador IPW de DoWhy ajusta por defecto una regresión logística sin escalar las covariables, y
    emitió `ConvergenceWarning` (lbfgs). El primer valor de IPW (0,0896) venía de una propensión
    sin converger. **Corrección:** `propensity_score_model` = estandarización + logística con 3000
    iteraciones, con una aserción que detiene el notebook si la propensión no converge.
30. *Resultado anticipado como hecho (error del asistente).* Antes de ver los outputs, el asistente
    redactó que condicionar en el colisionador simulado C "asocia la exposición con las estrellas"
    (Berkson). Con los datos reales, la correlación T–estrellas casi no cambia (0,0247 → 0,0284):
    las estrellas apenas causan la reserva con orden aleatorio, y Berkson requiere que la variable
    cause el colisionador a través de Y. **Corrección:** el contraste usa `prop_location_score2`
    (el atributo más predictivo de la reserva), se agrega la simulación con verdad conocida, y el
    texto informa lo que muestran los datos. **Prevención:** los textos interpretativos se escriben
    después de leer el output, nunca antes.

### Revisión humana que cambió el análisis: robustez de la conclusión central (12/11/2026)

**Prompt (Verónica, resumen).** Revisión punto por punto del cierre de las Fases 3-5, con
prioridad en blindar la conclusión causal de la sección 4.3. Es el ejemplo más claro del proyecto
de **revisión humana que cambió el análisis**: cada pedido se convirtió en un análisis nuevo con
evidencia propia.

| Pedido de Verónica | Qué faltaba en la versión del asistente | Qué se agregó (propuesto por Claude Code según el pedido) |
|---|---|---|
| Balance entre las búsquedas aleatorias y las del algoritmo, con SMD, y re-ponderación si difieren | La comparación R = 0 vs. R = 1 asumía poblaciones comparables sin verificarlo | Tabla de SMD sobre las búsquedas que entran al estimador, pesos de odds R0 → R1, OR y efectos marginales re-estimados con los pesos (notebook, 4.3 bis a) |
| No colapsabilidad del OR: repetir en diferencia de riesgos | Solo había escala OR (no colapsable) | Efecto marginal promedio del logit de elección y MPL con efectos fijos, restringidos a búsquedas con reserva (tasa base comparable) (4.3 bis b) |
| IC del cociente 3,38/2,11 por bootstrap clusterizado (≥ 1000 réplicas) y test formal de la diferencia de log-OR | La conclusión se apoyaba en IC que no se superponían, y el "adelanto" del asistente usaba solo 24 réplicas | Test de Wald y bootstrap percentil por búsqueda con 1000 réplicas, paralelizado, también para los efectos marginales y la diferencia de riesgos (4.3 bis c) |
| Redacción: no "se rechaza Backdoor"; nombrar candidatos de U | Formulación más fuerte que lo que permite el diseño | "Los datos son incompatibles con que el ajuste por lo observable elimine el sesgo", con candidatos concretos (conversión histórica, comisiones, inventario, contenido de la ficha, sesión) |
| El clic no es un mediador sustantivo: reserva ⇒ clic | Se presentaba el 95 % de "mediación" como hallazgo, cuando es casi mecánico | Descomposición exacta del embudo RR = RR_clic × RR_conversión, con método e IC bootstrap; el residuo de 0,004 se explica como diferencia de conversión entre quienes hacen clic (4.4) |
| Valuación en dólares como rango con supuestos explícitos | Un único número (≈ 10,5 reservas por cada 1.000 exposiciones) | Tres escenarios (conservador, central y optimista) con Δp, gasto y desplazamiento explícitos (4.9) |
| "Solo agregados" al 100 % en el notebook | Dispersiones con puntos individuales; mínimos y máximos en tablas | Hexbin con celdas de al menos 10 observaciones; percentiles 1-99 en lugar de mínimo y máximo |

**Errores / alucinaciones detectados.**
31. *Imprecisión de un optimizador de biblioteca (detectada al validar el estimador nuevo).* El
    `ConditionalLogit` de statsmodels con BFGS se detiene antes del máximo: en el test, su
    log-verosimilitud (−4267,51309) es menor que la del Newton exacto (−4267,51304), con un error
    relativo en el OR de ≈ 5·10⁻⁴. **Corrección:** el logit condicional usa el estimador exacto
    por Newton-Raphson (una reserva por búsqueda) con errores sándwich por búsqueda. Los OR
    reportados cambian en la cuarta cifra (3,3778 → 3,3777). **Test:**
    `test_eleccion_logit_coincide_con_statsmodels` (contra el Newton de statsmodels, rel = 1e-6).
32. *Evidencia insuficiente presentada como adelanto (del asistente).* El primer bootstrap del
    cociente de OR usó 24 réplicas. Verónica exigió al menos 1000 réplicas con IC percentil.
    **Corrección:** B = 1000 en la corrida final. **Prevención:** un IC por bootstrap solo se
    reporta con B ≥ 1000.

### Bonus: copiloto de decisión (12/11/2026)

**Prompt (Verónica, resumen).** Estimar el esfuerzo de una versión mínima del copiloto
conversacional y hacerla si hay margen antes del 16/11, sin tocar lo anterior, para compensar la
dimensión Scrum.

**Estimación del asistente:** 3 a 4 horas de desarrollo y tests, más la configuración del secreto
en Streamlit Cloud, que la hace Verónica. Hubo margen y se hizo el mismo 12/11, en código nuevo y
aislado.

**Qué produjo la IA (propuesto por Claude Code, pendiente de revisión de Verónica):**
- `app/copiloto.py`: 5 herramientas de solo lectura sobre los artefactos agregados (efecto causal
  y robustez, CATE por perfil, predicción del RF, refutaciones, escenarios de valuación).
- Loop manual de tool use con el SDK oficial (`anthropic==2.8.0`), modelo `claude-opus-5-5`,
  `fallbacks` del lado del servidor ante rechazos y límite de 6 pasos.
- **Guardarraíl contra cifras inventadas:** si la respuesta final contiene una cifra que no está en
  ningún resultado de herramienta ni en la pregunta, se bloquea y se muestran los resultados
  crudos.
- Pestaña 4 de la app. Sin `ANTHROPIC_API_KEY`, la pestaña explica cómo activarla y el resto de
  la app funciona igual.
- `tests/test_copiloto.py`, con un cliente simulado y sin llamadas reales: formatos numéricos en
  español, el guardarraíl acepta cifras redondeadas y rechaza inventadas, el loop devuelve el
  `tool_result` con el mismo id, bloquea una cifra inventada ("clientes de más de 45 años: 12,7 %"),
  maneja rechazos y las herramientas no devuelven identificadores.

**Limitación declarada:** el guardarraíl verifica que cada cifra *exista* en un resultado de
herramienta, no que esté usada en el sentido correcto. La interpretación sigue requiriendo
criterio humano.
