## Qué cambia

Cierra la **Fase 0** (setup, contrato de datos y DAG borrador). Milestone: Fase 0.

- Estructura del repo, `ruff` + `pytest` + CI (`quality-gate.yml`), plantilla de PR, `SEED = 42`, α = 0,05.
- **Datos (Expedia ICDM 2013):** `data/download_data.py` descarga `data.zip` (Deflate64, se extrae con `inflate64` verificando CRC-32) y muestrea 30.000 búsquedas completas estratificadas por `random_bool`. El crudo queda **fuera del repo y de OneDrive** (`~/data/tpi-eaymr/raw`, configurable con `TPI_RAW_DIR`); solo se versiona `data/manifiesto.json` (SHA-256 y tamaños, sin registros).
- **Contrato de datos** con la definición de `random_bool` citada de la fuente primaria (Kaggle, pestaña Data).
- **Contrato causal genérico** (`tpi.contrato`) y **plan B** semisintético (Online Shoppers, UCI #468, CC BY 4.0) activable con `TPI_DATASET=online_shoppers`.
- **DAG borrador** con verificación de d-separación (`tests/test_dag.py`), esqueleto Quarto con bibliografía verificada.
- BITÁCORA: Entrada 1 y amenazas de validez A y B en la Entrada 4 (firmas: las pone Verónica). `docs/ia_log.md`: errores 1 a 14, con autoría de cada decisión. `docs/defensa.md`: guía de defensa por rol (Fase 0).

### Quién hizo y quién revisa (trazabilidad real)

- **Autoría:** Verónica López Marsico, con asistencia de Claude Code (Claude Opus 5.5). Quién propuso y quién aprobó cada decisión está en `docs/ia_log.md`.
- **Revisión QA:** la hace **Verónica**, leyendo este PR. Desde el 12/11/2026 Verónica ejecuta y valida todo el trabajo; los roles Scrum del README se mantienen para la defensa oral, pero Lucía Báez y Antonella Robiollio no revisan ni validan PRs (BITÁCORA, "Reorganización").

### Hallazgos a revisar con atención

1. **`random_bool` no es un RCT puro** (ia_log, error 10). Con R = 1 la asociación intra-búsqueda posición–atributo cae entre 71 % y 87 % respecto del algoritmo, pero sigue siendo distinta de 0 para `prop_starrating` (ρ̄ = −0,0743) y `prop_location_score2` (ρ̄ = −0,0651). El test de equivalencia original queda como `xfail(strict=True)` en esas dos variables. El nuevo test exige una reducción ≥ 50 %, con **umbral post hoc** declarado como tal. ¿Es aceptable esta forma de registrar una refutación?
2. **Selección del dataset** (ia_log, error 14; BITÁCORA, Entrada 4): las 399.344 búsquedas tienen ≥ 1 clic, ninguna tiene > 1 reserva, y hay reserva en el 93,86 % de las búsquedas R = 0 contra el 13,04 % de las R = 1.
3. **Confidencialidad:** Verónica decidió (12/11) no consultar a la cátedra y seguir con Expedia con las mitigaciones implementadas; el plan B queda disponible y no se activa (BITÁCORA, Entradas 1 y 4). Comprobar que `tests/test_confidencialidad.py` cubre las extensiones de datos relevantes.

### Desvío de proceso (ia_log, punto 9)

Los primeros 5 commits de esta rama se pushearon antes de tener `gh` y sin PR. No llegaron a `main`, así que la regla de oro no se violó. Decidimos no reescribir el historial: este PR los somete a auditoría junto con el resto.

## Fase y rol
- Fase: 0
- Autora: Verónica López Marsico (Project Lead & Scrum Master), con asistencia de Claude Code (registrada en `docs/ia_log.md`)

## Evidencia
- [x] `ruff check .` sin errores
- [x] `ruff format --check .` sin cambios
- [x] `pytest` en verde (resumen abajo)
- [x] Sin datos crudos ni credenciales en el diff (`test_ningun_archivo_de_datos_trackeado`)
- [ ] Si toca el notebook: "Restart & Run All" sin errores (no aplica: el notebook arranca en la Fase 1)
- [x] Uso de IA registrado en `docs/ia_log.md`

Local, con la muestra real (Windows, Python 3.11.9, venv `~/.venvs/tpi-eaymr`):
```
67 passed, 1 skipped, 2 xfailed in 7.60s
XFAIL test_random_bool_1_orden_independiente_de_atributos[prop_starrating]       (refutación registrada, error 10)
XFAIL test_random_bool_1_orden_independiente_de_atributos[prop_location_score2]  (refutación registrada, error 10)
SKIPPED test_random_bool_1_reduce_la_asociacion_del_algoritmo[price_usd]          (el algoritmo no la usa: rho=-0.0182)
```
En CI (sin datos) los tests marcados `data` se saltean.

**Para reproducir la parte de datos:** autenticar Kaggle y aceptar las reglas de la competencia, y después `python data/download_data.py` y `pytest -m data`.

## Veredicto QA
<!-- Lo emite Verónica López Marsico (revisora real; ver "Quién hizo y quién revisa"): [APROBADO] o [BLOQUEADO] + motivo. Sin este veredicto no se mergea. -->

**Revisora:** Verónica López Marsico (rol nominal de QA: Antonella Robiollio).
**Veredicto:** _pendiente_

🤖 Generated with [Claude Code](https://claude.com/claude-code)
