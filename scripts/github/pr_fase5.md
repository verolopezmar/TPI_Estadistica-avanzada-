## Qué cambia

Cierra la **Fase 5** y el TP: notebook completo ejecutado (Fases 1 a 5), app Streamlit, artefactos agregados, informe Quarto (HTML y PDF), slides (PDF), README editorial, BITÁCORA con las 5 entradas, guía de defensa y auto-auditoría. Milestone: Fase 5. **Depende de los PRs de las Fases 0 a 4.**

- `notebooks/TP_Integrador_Final.ipynb` (+ fuente jupytext `.py`): Fases 1 a 5, ejecutado de punta a punta.
- `app/`: `app.py` (3 módulos: Diagnóstico + PCA, What-Is, What-If con semáforo de refutaciones), `logica.py` (testeable), `requirements.txt` mínimo para Streamlit Cloud, tema.
- `models/`: `artefactos_app.json` (solo agregados; celdas con menos de 10 filas suprimidas), `rf_app.joblib` (RF compacto, hojas de ≥ 100 filas), `resultados.json` (fuente única de los números del informe y las slides).
- `src/tpi/artefactos.py` + `tests/test_app.py`: sin identificadores, supresión de celdas, rangos de entrada, la app corre sin excepciones (`AppTest`).
- `report/informe.qmd` → HTML y PDF; `report/slides.qmd` → PDF (beamer). Números inline desde `models/resultados.json`.
- `README.md`, `BITACORA.md` (Entradas 1 a 5), `docs/defensa.md` (Fases 0 a 5), `docs/auditoria_final.md`, `scripts/github/`.

### Confidencialidad (revisar)
- La app no lee datos: solo `models/`.
- `test_ninguna_celda_publicada_describe_menos_de_k_filas` y `test_artefactos_sin_identificadores`.
- El RF serializado son "resulting model parameters" en sentido estricto: es el riesgo residual que Verónica decidió asumir (BITÁCORA, Entrada 1).

### Deploy
Pasos en el README (sección "Deploy en Streamlit Community Cloud"). El login y el deploy los hace Verónica.

## Quién hizo y quién revisa
- **Autoría:** Verónica López Marsico, con asistencia de Claude Code (Claude Opus 5.5). Decisiones "propuestas por Claude Code" en `docs/ia_log.md`.
- **Revisión QA:** Verónica (rol nominal de QA: Antonella Robiollio; Lucía es QA suplente nominal en los PRs de la app).

## Evidencia
- [x] `ruff check .` y `ruff format --check .` sin errores
- [x] `pytest` en verde: 116 passed, 1 skipped, 2 xfailed (con la muestra real; en CI los tests de datos se saltean)
- [x] Sin datos en el diff
- [x] Notebook ejecutado de punta a punta (`jupyter nbconvert --execute`)
- [x] `quarto render` del informe (HTML y PDF) y de las slides (PDF)
- [x] `streamlit run app/app.py` local y `AppTest` sin excepciones
- [x] Uso de IA registrado en `docs/ia_log.md`

## Veredicto QA
**Revisora:** Verónica López Marsico (rol nominal de QA: Antonella Robiollio).
**Veredicto:** _pendiente_

🤖 Generated with [Claude Code](https://claude.com/claude-code)
