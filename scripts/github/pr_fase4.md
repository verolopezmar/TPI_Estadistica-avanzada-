## Qué cambia

Cierra la **Fase 4** (inferencia causal, Causal ML y el choque causal vs. predictivo): el módulo causal y sus tests. Milestone: Fase 4. **Depende de los PRs de las Fases 0 a 3.**

> El notebook es un único artefacto ejecutado de punta a punta (Restart & Run All). Para no ejecutarlo tres veces, las secciones 3, 4 y 5 ejecutadas, las figuras y los artefactos llegan en el PR de la Fase 5. Este PR trae el código y los tests causales.

- `src/tpi/causal.py`: logit condicional por búsqueda (OR intra-búsqueda, con guarda que falla si no converge), MPL con efectos fijos y errores por clúster, flujo DoWhy con DAG de columnas reales y mediador explícito, simulaciones con verdad conocida (filtro de selección S y colisionador C).
- `tests/test_causal.py`: el OR condicional recupera la verdad y **sobrevive al filtro S**; la diferencia absoluta se infla con S; condicionar en C induce Berkson; DoWhy identifica Backdoor sin el mediador y recupera el efecto.
- `requirements.txt`: `networkx==3.4.2` (dowhy 0.12 rompe con networkx ≥ 3.5; ia_log, error 26).

### Resultados principales (notebook, sección 4; llegan con el PR de la Fase 5)
- OR intra-búsqueda del top-5: **benchmark (orden aleatorio, ajustado) 2,11 [1,90; 2,35]**; observacional ajustado 3,38 [3,25; 3,51]; ingenuo 5,38. El observacional sobreestima un 60 % aun ajustado: el Backdoor con Z observado se rechaza empíricamente.
- DoWhy (R = 0): ingenuo 0,1055; regresión 0,0961; IPW 0,0890. Las 3 refutaciones pasan; un U simulado moderado reduce el ATE un 48 %.
- Mediador: el clic absorbe el 94,7 % del efecto. Selección S: el OR se conserva y la Δp se duplica (simulación con verdad conocida).
- DML: ATE 0,083 (R = 0) y 0,007 (R = 1). CATE mayor con precio relativo bajo y buena ubicación.

### A revisar con atención
1. Que la conclusión causal (benchmark con orden aleatorio vs. observacional ajustado) esté bien planteada y declarada en sus límites.
2. El logit condicional como estimador principal (escala relativa) y la lectura de las refutaciones.
3. Los errores 26 a 28 del log (dependencias, colinealidad intra-búsqueda, no convergencia).

## Quién hizo y quién revisa
- **Autoría:** Verónica López Marsico, con asistencia de Claude Code (Claude Opus 5.5). Decisiones "propuestas por Claude Code" en `docs/ia_log.md`.
- **Revisión QA:** Verónica (rol nominal de QA: Antonella Robiollio).

## Evidencia
- [x] `ruff check .` y `ruff format --check .` sin errores
- [x] `pytest` en verde: 116 passed, 1 skipped, 2 xfailed (con la muestra real; en CI los tests de datos se saltean)
- [x] Sin datos en el diff
- [x] Uso de IA registrado en `docs/ia_log.md`

## Veredicto QA
**Revisora:** Verónica López Marsico (rol nominal de QA: Antonella Robiollio).
**Veredicto:** _pendiente_

🤖 Generated with [Claude Code](https://claude.com/claude-code)
