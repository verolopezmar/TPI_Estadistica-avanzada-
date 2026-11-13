## Qué cambia

Cierra la **Fase 1** (inferencia clásica, auditoría forense, faltantes, $Z_c$, contrastes y ANOVA). Milestone: Fase 1. **Depende del PR de la Fase 0**: mergear la Fase 0 antes.

- `src/tpi/inferencia.py`: Welch, Mann-Whitney, t pareado + Wilcoxon, potencia y efecto mínimo detectable, ANOVA clásico + Levene + Welch + Kruskal-Wallis, post-hoc Tukey vs. Games-Howell vs. Bonferroni. `Contraste.texto()` aplica las reglas de reporte del TPI.
- `src/tpi/auditoria.py`: centinelas, escala del precio, outliers (Tukey en log + relativo a la búsqueda), contraste de MCAR, imputación con indicador, agregados de competidores.
- Notebook `notebooks/TP_Integrador_Final.ipynb` (Fase 1), ejecutado de punta a punta; fuente jupytext versionada (`.py`). Figuras en `report/figures/f1_*.png`.
- DAG: Zc → T, Zc → M y nodo de selección S, con tests de d-separación nuevos. T de sensibilidad `top4_dataset`.
- BITÁCORA: Entrada 2 y amenazas/limitaciones en la Entrada 4. `docs/ia_log.md`: errores 15 a 22. `docs/defensa.md`: Fase 1.

### Resultados principales (todos de outputs del notebook)

- **Contraste 1 (pareado, R = 1):** t(14982) = 37,47, p = 7,74e-294, IC 95 % de la diferencia de CTR [0,0531; 0,0590], d_z = 0,306. Cociente de CTR top-5/resto = 2,16.
- **Contraste 2 (Welch, log gasto):** t(1158,6) = 1,303, p = 0,193 → no se rechaza H0. Efecto mínimo detectable g = 0,090.
- **ANOVA (Welch, Levene rechaza):** F(3; 4432,0) = 1807, η² = 0,254; los tres post-hoc coinciden.
- **$Z_c$:** `comp_saldo` y `comp_n_info` son confusoras (interacción $Z_c \times R$ sobre T: p = 2,7e-18 y 2,8e-05; efecto sobre la reserva con R = 1); `comp_diff_max` se excluye.

### A revisar con atención
1. La clasificación de $Z_c$ con el contraste de Z → T (pendiente de tu revisión).
2. Slots fijos 5/11/17/23 y la limitación declarada de R = 1 (Entrada 4, B y C).
3. `compN_inv = -1` (valor no documentado, 1,27 % de los no nulos).

## Quién hizo y quién revisa
- **Autoría:** Verónica López Marsico, con asistencia de Claude Code (Claude Opus 5.5); autoría de cada decisión en `docs/ia_log.md`.
- **Revisión QA:** Verónica, leyendo este PR (rol nominal de QA: Antonella Robiollio; desde el 12/11 Verónica hace todas las validaciones).

## Evidencia
- [x] `ruff check .` sin errores
- [x] `ruff format --check .` sin cambios
- [x] `pytest` en verde
- [x] Sin datos crudos ni credenciales en el diff (`test_ningun_archivo_de_datos_trackeado`); el notebook solo muestra agregados
- [x] Notebook ejecutado de punta a punta (`jupyter nbconvert --execute`, equivalente a Restart & Run All)
- [x] Uso de IA registrado en `docs/ia_log.md`

Local, con la muestra real:
```
90 passed, 1 skipped, 2 xfailed
```
En CI (sin datos) los tests marcados `data` se saltean.

## Veredicto QA
**Revisora:** Verónica López Marsico (rol nominal de QA: Antonella Robiollio).
**Veredicto:** _pendiente_

🤖 Generated with [Claude Code](https://claude.com/claude-code)
