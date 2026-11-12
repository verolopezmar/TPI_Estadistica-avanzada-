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

**Qué decidimos.** _(a completar por el equipo al aprobar la Fase 0: dataset elegido, roles)._

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
