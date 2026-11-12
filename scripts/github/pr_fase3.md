## Qué cambia

Cierra la **Fase 3** (torneo predictivo, calibración y explicabilidad). Milestone: Fase 3. **Depende de los PRs de las Fases 0 a 2.**

- `src/tpi/modelos.py`: features sin fuga (excluye `click_bool`, gasto y respuesta), CV `StratifiedGroupKFold(5)` por búsqueda, métricas, umbral por F1 elegido en Train, métrica de ranking.
- `src/tpi/preparacion.py`: pipeline de la Fase 1 en una función, verificada en el notebook contra los pasos expositivos.
> El notebook es un único artefacto ejecutado de punta a punta; sus secciones 3 a 5 ejecutadas llegan en el PR de la Fase 5. Este PR trae el código y los tests de la Fase 3.

- Notebook, Fase 3 (en el PR de la Fase 5): logística L1/L2 (grilla) con OR e IC 95 % (logit sin penalizar, errores por clúster); Naive Bayes gaussiano con auditoría de independencia condicional; CART sin restricciones vs. podado por `ccp_alpha`; Random Forest con `RandomizedSearchCV`, OOB y curva de convergencia; evaluación en Test (ROC, PR, Brier, F1 macro/ponderado, reliability diagrams); MDI vs. permutación; holdout cronológico; ranking vs. Expedia; PCA como features.
- `compN_inv = −1` como faltante con indicador (aprobado por Verónica el 12/11).

### Resultados principales
RESULTADOS_F3

### A revisar con atención
1. Búsqueda de hiperparámetros en una submuestra del 25 % de las búsquedas de Train (justificación de cómputo) con reentrenamiento en todo Train.
2. `position` como feature predictiva (legítima para predecir; es el centro del ensayo causal de la Fase 4).
3. El umbral de clasificación elegido en Train, no en Test.

## Quién hizo y quién revisa
- **Autoría:** Verónica López Marsico, con asistencia de Claude Code (Claude Opus 5.5). Decisiones "propuestas por Claude Code" en `docs/ia_log.md`.
- **Revisión QA:** Verónica (rol nominal de QA: Antonella Robiollio).

## Evidencia
- [x] `ruff check .` y `ruff format --check .` sin errores
- [x] `pytest` en verde (local, con datos): TESTS
- [x] Sin datos en el diff (`test_ningun_archivo_de_datos_trackeado`)
- [x] Notebook ejecutado de punta a punta
- [x] Uso de IA registrado en `docs/ia_log.md`

## Veredicto QA
**Revisora:** Verónica López Marsico (rol nominal de QA: Antonella Robiollio).
**Veredicto:** _pendiente_

🤖 Generated with [Claude Code](https://claude.com/claude-code)
