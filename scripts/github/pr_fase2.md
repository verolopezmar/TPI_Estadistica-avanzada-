## Qué cambia

Cierra la **Fase 2** (Gauss-Markov, puntos influyentes, VIF y PCA sin fuga). Milestone: Fase 2. **Depende de los PRs de las Fases 0 y 1**: mergearlos antes.

- `src/tpi/particion.py`: partición única 80/20 agrupada por búsqueda y estratificada por `random_bool` (Fases 2-4).
- `src/tpi/regresion.py`: Breusch-Pagan, White, Jarque-Bera, Durbin-Watson; residuo estudentizado, leverage y Cook con los umbrales del enunciado; VIF; PCA ajustado solo en Train.
- Notebook, Fase 2: MCO de log(precio por noche) sobre reservas de Train, HC3, Huber, RANSAC (por defecto vs. umbral 3σ̂), VIF y PCA con proyección ciega de Test. Figuras `report/figures/f2_*.png`.
- BITÁCORA: Entrada 2 (Fase 2) y Entrada 3 (tres experimentos fallidos reales). `docs/ia_log.md`: errores 23-25 y el bloqueo de `gh`. `docs/defensa.md`: Fase 2.

### Resultados principales (outputs del notebook)
- MCO: R² = 0,604; F(17; 12840) = 1150,64, p < 1e-300.
- BP p = 4,96e-99 y White p = 1,71e-118 → HC3. JB rechaza (curtosis 5,80). DW = 1,9515.
- 746 influyentes con D > 4/n, ninguno con D > 0,5; más frecuentes con la escala de precio incompatible (9,51 % vs. 4,85 %).
- VIF máx 1,74 (modelo) y 2,98 (conjunto de la Fase 3): sin multicolinealidad.
- PCA: Kaiser 4 componentes (69,7 %), codo en 2; reservados con PC2 medio 0,270 vs. −0,004.

### A revisar con atención
1. La partición (agrupada, estratificada, aleatoria y no cronológica).
2. La decisión de no eliminar influyentes (HC3 + Huber).
3. La retención de componentes (2 para la app, 4 como features alternativas).

## Quién hizo y quién revisa
- **Autoría:** Verónica López Marsico, con asistencia de Claude Code (Claude Opus 5.5); autoría de cada decisión en `docs/ia_log.md`.
- **Revisión QA:** Verónica (rol nominal de QA: Antonella Robiollio).

## Evidencia
- [x] `ruff check .` y `ruff format --check .` sin errores
- [x] `pytest` en verde (local, con datos): `97 passed, 1 skipped, 2 xfailed`
- [x] Sin datos en el diff; el notebook solo muestra agregados
- [x] Notebook ejecutado de punta a punta (`jupyter nbconvert --execute`)
- [x] Uso de IA registrado en `docs/ia_log.md`

## Veredicto QA
**Revisora:** Verónica López Marsico (rol nominal de QA: Antonella Robiollio).
**Veredicto:** _pendiente_

🤖 Generated with [Claude Code](https://claude.com/claude-code)
