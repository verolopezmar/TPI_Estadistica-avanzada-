# Registro de uso de IA (insumo de la Entrada 5 de la BITÁCORA)

Asistente: **Claude Code** (modelo Claude Opus 5.5, Anthropic), usado desde VS Code.
Alcance: el asistente propone código, textos y diseños; **todas las decisiones metodológicas las
toma el equipo** y quedan registradas acá. Ningún número del informe se toma del texto del
asistente: todo valor sale de un output ejecutado y versionado.

Formato por fase: prompt (resumido) · qué produjo la IA · qué decidimos · errores/alucinaciones
detectados · test determinístico que los refuta o previene.

---

## Fase 0 — Setup y selección del dataset (11/11/2026)

**Prompt (resumen).** Brief completo del TPI: leer enunciado y rúbrica, armar la estructura del
repo, verificar herramientas (Python, Quarto, TinyTeX), asignar roles Scrum y proponer 3-4
datasets reales para estimar el efecto causal de un recomendador, sin descargar datos hasta la
aprobación del equipo.

**Qué produjo.**
- Estructura de carpetas, `pyproject.toml` (ruff + pytest), `.gitignore`, `src/tpi/config.py`
  (SEED = 42, α = 0,05, rutas relativas), workflow de CI `quality-gate.yml`, plantilla de PR con
  veredicto QA, esqueletos de README y BITÁCORA, este log.
- Verificación del entorno: Python 3.11.9 y 3.13, Quarto 1.11.9 con TinyTeX 2026 OK; `gh` no
  instalado; `git` sin `user.name`/`user.email` global.
- Propuesta de 4 datasets candidatos con recomendación (ver mensaje de Fase 0 y Entrada 1).
- Propuesta de asignación de roles.

**Qué decidimos.** Ver la sección "Decisiones confirmadas por el equipo (12/11/2026)" más abajo.

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
   paper" sin decir cuál. Antes de citarlo verificamos el texto completo: Liu et al. (2013,
   arXiv:1311.7679) dice textualmente "The training data contains 399,344 unique search lists and
   9,917,530 points". El mismo paper **no** define `random_bool` (solo lo lista en una tabla de
   importancia), lo que confirma el punto 1.
   **Test previsto:** `test_data_contract.py` registra N y número de `srch_id` del archivo real y
   los compara con esas cifras.

---

## Fase 0 (continuación) — Render, contrato de datos y DAG borrador (11/11/2026)

**Prompt (resumen).** Dataset aprobado: Expedia (plan B: Online Shoppers semi-sintético). Roles
confirmados. Sin datos todavía: render de prueba de Quarto (¿rompe la ruta de OneDrive con
tilde?), script de descarga, contrato de datos, test de `random_bool` y DAG borrador. Sin
commits hasta tener `gh`.

**Qué produjo.** `report/informe.qmd` + `referencias.bib` (render HTML y PDF OK);
`src/tpi/data_contract.py`, `sampling.py`, `aleatorizacion.py`, `dag.py`;
`data/download_data.py`; `tests/test_environment.py`, `test_data_contract.py`, `test_dag.py`;
`docs/dag_borrador.md` + figura.

**Qué decidimos.** Dataset A (Expedia), roles según la propuesta, informe Quarto independiente
con el notebook como anexo. _(DAG: pendiente de revisión con Lucía.)_

**Errores / alucinaciones detectados.**
3. *Instrucción desactualizada del asistente (autenticación de Kaggle).* En la Fase 0 el
   asistente indicó guardar el token en `%USERPROFILE%\.kaggle\kaggle.json`. Al inspeccionar el
   paquete instalado (`kaggle==2.2.5`), su propio mensaje indica el esquema nuevo:
   `kaggle auth login` (OAuth), un token en `~/.kaggle/access_token` o la variable
   `KAGGLE_API_TOKEN`. **Corrección:** el docstring de `data/download_data.py` documenta el
   esquema vigente. **Prevención:** antes de escribir llamadas a la API inspeccionamos las
   firmas reales (`inspect.signature`) en lugar de confiar en la memoria del asistente.
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

**Prompt (resumen).** La sesión anterior del asistente se cerró. Se pide releer enunciado,
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

8. **Resuelta.** Procedimiento oficial: venv fuera de carpetas sincronizadas
   (`~/.venvs/tpi-eaymr`), instalado desde `requirements.txt`. El README se corrigió en
   consecuencia. Motivo: OneDrive bloquea y corrompe archivos del venv.
9. **Registrada tal cual, sin reescribir el historial.** Los 5 commits de
   `feat/fase0-contrato-datos` (12/11/2026 00:11) se hicieron y pushearon antes de instalar `gh`,
   contra lo planificado. No pasaron por un PR, pero tampoco llegaron a `main`, así que la regla
   de oro (ningún merge a `main` sin firma del QA) no se violó. Decisión del Project Lead: no
   reescribir el historial (hacerlo sobre una rama pública sería peor que el desvío) y abrir el
   PR de la Fase 0 ahora, para que el QA audite los 5 commits juntos antes del merge.

### Decisiones confirmadas por el equipo (12/11/2026)

- **Dataset:** Expedia "Personalize Expedia Hotel Searches — ICDM 2013" (Kaggle). Plan B:
  Online Shoppers con tratamiento semi-sintético.
- **Roles:** Verónica López Marsico = Project Lead & Scrum Master; Lucía Báez = Backend & ML/
  Causal Engineer; Antonella Robiollio = Frontend & UI/UX + QA (Lucía es QA suplente en los PRs
  de la app).
- **Informe:** Quarto independiente, con `TP_Integrador_Final.ipynb` como anexo.
- **DAG:**
  - T binaria = hotel en el top-5 de la búsqueda; la posición continua va como análisis de
    sensibilidad.
  - Y principal = `booking_bool`. `gross_bookings_usd` solo existe si hubo reserva:
    condicionar en "reservó" se usa como ejemplo de la trampa de selección/colisionador.
  - Zc (variables de competidores): la clasifica el asistente con el criterio del equipo
    (si el ranking la usa **y** afecta la reserva, es confusor; si solo afecta Y, es predictor),
    con justificación empírica en la Fase 1.
  - Colisionador: natural si encontramos uno defendible, más el simulado como validación.
  - `click_bool` se evalúa como mediador natural (posición → clic → reserva) para la trampa del
    mediador.
  - `random_bool` como experimento natural: se comparan las estimaciones observacionales (orden
    del algoritmo) con el efecto bajo orden aleatorio, **después** de verificar en los datos y en
    la documentación de la competencia que la variable significa eso.
- **Entorno oficial:** venv en `~/.venvs/tpi-eaymr` (ver punto 8).
- **Historial de git:** no se reescribe (ver punto 9).

---

## Fase 0 (cierre) — Descarga, verificación de `random_bool` y confidencialidad (12/11/2026)

**Prompt (resumen).** Autenticar Kaggle y descargar; verificar `random_bool` y la licencia;
correr los 7 tests de datos. Luego, con la definición de `random_bool` y las reglas copiadas
por el equipo de la pestaña Data/Rules: citar la fuente primaria, analizar el sesgo de selección
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

**Qué decidimos (equipo).**
- `random_bool`: se cita la definición de la fuente primaria. Ante la refutación parcial (error
  10), R = 1 se trata como **cuasi-experimento ajustado**: efectos fijos por búsqueda + ajuste
  por Z, reportando sin ajuste vs. ajustado.
- Comparación observacional vs. experimental: **escala relativa como principal** (risk ratio /
  log-odds dentro de la búsqueda) y, como robustez, ambos estratos restringidos a búsquedas con
  reserva (ATE absoluto). `booking_bool` sigue como Y principal; `click_bool` como secundario.
- Confidencialidad: consulta al profesor (pendiente). Mientras tanto, las mitigaciones de
  `data/README.md` y el plan B evaluado.

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
12. *Discrepancia de transcripción detectada por la API.* El tamaño de `data.zip` informado a
    mano era 434.537.097 bytes; la API y el archivo descargado dicen **434.573.097** (dos dígitos
    traspuestos). **Prevención:** el manifiesto registra tamaños y hashes del archivo real; nunca
    se copian cifras a mano.
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

**Prompt (resumen).** Arrancar la Fase 1 en una rama nueva mientras el QA revisa la Fase 0.

**Qué produjo la IA.** `src/tpi/inferencia.py` (Welch, Mann-Whitney con r biserial y
Hodges-Lehmann, t pareado con Wilcoxon de resguardo, potencia y efecto mínimo detectable, ANOVA
clásico + Levene/Brown-Forsythe + Welch + Kruskal-Wallis, post-hoc Tukey vs. Games-Howell vs.
Bonferroni) y `tests/test_inferencia.py`. `Contraste.texto()` aplica las reglas de reporte del TPI
(estadístico, gl, p exacto, IC 95 %, tamaño de efecto, "no se rechaza H0").

**Diseño propuesto por la IA (a validar por Lucía).** Contrastes sobre unidades independientes:
(1) pareado dentro de la búsqueda con R = 1, CTR del top-5 vs. del resto (sesgo de posición, arco
T → M del DAG); (2) gasto por reserva, usuarios recurrentes vs. nuevos (Welch + Mann-Whitney);
ANOVA: log del precio por noche del hotel reservado según categoría de estrellas.

**Errores / alucinaciones detectados.**
15. *API recordada de una versión vieja.* El asistente usó los nombres de columnas de pingouin
    `p-unc`, `p-val` y `CI95%`; en `pingouin==0.6.1` son `p_unc`, `p_val` y `CI95`. Fallaron 3
    tests con `KeyError`. **Corrección:** se inspeccionaron las columnas reales antes de
    corregir. **Test:** `tests/test_inferencia.py` compara cada resultado contra pingouin, así que
    un cambio de API rompe el test en lugar de producir un número equivocado.
