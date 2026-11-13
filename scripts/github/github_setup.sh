#!/usr/bin/env bash
# Crea milestones (uno por fase), labels, issues por tarea y el PR de la Fase 0.
# Requiere: gh autenticado (gh auth status OK). Correr desde la raíz del repo.
set -euo pipefail
GH="${GH:-gh}"
SCRATCH="$(dirname "$0")"

milestone() { # titulo, fecha_limite, descripcion
  $GH api "repos/{owner}/{repo}/milestones" -f title="$1" -f due_on="$2T23:59:00-03:00" \
    -f description="$3" --jq .number
}
label() { $GH label create "$1" --color "$2" --description "$3" --force >/dev/null; }

label "rol:lead"     "5319e7" "Project Lead & Scrum Master"
label "rol:backend"  "0e8a16" "Backend & ML/Causal Engineer"
label "rol:frontend" "1d76db" "Frontend & UI/UX Engineer"
label "rol:qa"       "d93f0b" "QA & Validation Engineer"
label "riesgo"       "b60205" "Puede no llegar al 18/11 o cambia resultados"

M0=$(milestone "Fase 0 - Setup y contrato de datos" 2026-11-12 "Estructura, CI, contrato de datos, DAG borrador, Entrada 1.")
M1=$(milestone "Fase 1 - Inferencia clásica y ANOVA" 2026-11-13 "EDA, MCAR/MAR/MNAR, 2 contrastes, ANOVA/Welch/Kruskal + post-hoc. 15 pts.")
M2=$(milestone "Fase 2 - Gauss-Markov y PCA" 2026-11-14 "MCO, BP/JB/DW, Cook/leverage, VIF, PCA solo en Train. 15 pts.")
M3=$(milestone "Fase 3 - Torneo ML y calibración" 2026-11-15 "Logit/NB/CART/RF, CV agrupada, calibración, MDI vs permutación. 20 pts.")
M4=$(milestone "Fase 4 - Inferencia causal" 2026-11-16 "DAG, Backdoor, DoWhy IPW/regresión, DML/CATE, 3 refutaciones, ensayo. 25 pts.")
M5=$(milestone "Fase 5 - Scrum, bitácora y deploy" 2026-11-17 "App Streamlit (3 módulos), BITÁCORA completa, slides, README final. 25 pts.")

issue() { # milestone_num, titulo, labels, cuerpo
  local t; t=$($GH api "repos/{owner}/{repo}/milestones/$1" --jq .title)
  $GH issue create --title "$2" --label "$3" --milestone "$t" --body "$4" >/dev/null
  echo "  + $2"
}

echo "Issues Fase 0"
issue $M0 "F0: Contrato de datos y descarga reproducible" "rol:backend" "Hecho en el PR de la Fase 0."
issue $M0 "F0: Auditoría QA del PR de la Fase 0" "rol:qa" "Revisora real: Verónica (desde el 12/11 hace todas las validaciones; rol nominal de QA: Antonella). Veredicto [APROBADO]/[BLOQUEADO] en el PR."
issue $M0 "F0: Riesgo de licencia de Expedia (decisión documentada)" "rol:lead,riesgo" "Reglas de Kaggle: datos y resulting model parameters confidenciales. Decisión de Verónica (12/11): no consultar a la cátedra, seguir con Expedia con mitigaciones testeadas; plan B disponible, no activado. Ver BITÁCORA, Entradas 1 y 4."

echo "Issues Fase 1"
issue $M1 "F1: EDA univariado/bivariado (KDE, boxplots, Spearman/Pearson)" "rol:backend" "Figuras de nivel publicación. Unidad: búsqueda×hotel; resúmenes por búsqueda."
issue $M1 "F1: Auditoría forense (centinelas, faltantes, outliers de precio) con antes/después" "rol:backend" "Rúbrica 1.3: cuantificar la distorsión antes vs después."
issue $M1 "F1: Mecanismo de faltantes MCAR/MAR/MNAR (Little + tests de dependencia)" "rol:backend" "Justificar la imputación elegida."
issue $M1 "F1: Clasificar Zc (competidores): confusor vs predictor" "rol:backend" "Criterio del equipo: confusor si el ranking la usa Y afecta la reserva."
issue $M1 "F1: Dos contrastes de hipótesis con IC 95 %, tamaño de efecto y potencia" "rol:backend" "Estadístico, gl, p exacto, IC 95 %, d/r. α = 0,05 a priori."
issue $M1 "F1: ANOVA de 1 factor + Levene + Welch/Kruskal + post-hoc (Games-Howell/Bonferroni)" "rol:backend" "Shapiro no sirve con N grande: skewness/curtosis/Q-Q."
issue $M1 "F1: Tests de no-leakage y de supuestos de la Fase 1" "rol:qa" ""
issue $M1 "F1: Entrada 2 de la BITÁCORA (matriz de decisiones estadísticas)" "rol:backend,rol:qa" ""

echo "Issues Fase 2"
issue $M2 "F2: MCO de referencia (log precio / gasto) con R², F y t" "rol:backend" ""
issue $M2 "F2: Breusch-Pagan, Jarque-Bera, Durbin-Watson + HC3/cluster" "rol:backend" ""
issue $M2 "F2: Cook, leverage, residuos estudentizados + Huber/RANSAC" "rol:backend" ""
issue $M2 "F2: VIF y estrategia (eliminación / Ridge)" "rol:backend" ""
issue $M2 "F2: PCA solo en Train (scree, Kaiser, loadings)" "rol:backend" "Test de no-leakage: el scaler/PCA se ajustan solo con Train."

echo "Issues Fase 3"
issue $M3 "F3: Split 80/20 agrupado por srch_id + StratifiedGroupKFold(5)" "rol:backend,rol:qa" "Test: ningún srch_id en Train y Test a la vez."
issue $M3 "F3: Logit regularizada con OR e IC 95 %" "rol:backend" ""
issue $M3 "F3: Naive Bayes y violación de independencia condicional" "rol:backend" ""
issue $M3 "F3: CART sin poda vs poda ccp_alpha por CV" "rol:backend" ""
issue $M3 "F3: Random Forest (Randomized/GridSearch) + OOB vs CV" "rol:backend" ""
issue $M3 "F3: Métricas de test, reliability diagrams, Brier, MDI vs permutación" "rol:backend" ""
issue $M3 "F3: Entrada 3 (autopsia de >= 2 experimentos fallidos)" "rol:backend" ""

echo "Issues Fase 4"
issue $M4 "F4: DAG final (DOT/networkx) y demostración Backdoor" "rol:backend,rol:lead" ""
issue $M4 "F4: DoWhy regresión + IPW en R=0 vs cuasi-experimento R=1 ajustado" "rol:backend" "Escala relativa principal; robustez en búsquedas con reserva."
issue $M4 "F4: Trampas del mediador (click) y del colisionador (selección S + simulado)" "rol:backend" ""
issue $M4 "F4: DML (LinearDML/CausalForestDML) y curvas CATE" "rol:backend" ""
issue $M4 "F4: Refutaciones (random common cause, placebo, subset)" "rol:backend,rol:qa" ""
issue $M4 "F4: Simulación del filtro de selección (risk ratio intra-búsqueda)" "rol:backend,rol:qa" "BITÁCORA, Entrada 4, amenaza A (c)."
issue $M4 "F4: Ensayo crítico predictivo vs causal + Entrada 4" "rol:lead,rol:backend" ""

echo "Issues Fase 5"
issue $M5 "F5: App Streamlit - Diagnóstico + PCA (solo agregados)" "rol:frontend" "Confidencialidad: nunca registros individuales."
issue $M5 "F5: App - What-Is (predicción calibrada + importancia local)" "rol:frontend" ""
issue $M5 "F5: App - What-If (DAG, ATE, CATE, semáforo de refutaciones)" "rol:frontend" ""
issue $M5 "F5: Deploy en Streamlit Community Cloud + tests de rangos de la app" "rol:frontend,rol:qa" "QA suplente: Lucía."
issue $M5 "F5: Informe Quarto final + notebook Restart & Run All" "rol:lead" ""
issue $M5 "F5: Slides (10-12) y ensayo de defensa por rol" "rol:lead" ""
issue $M5 "F5: README editorial (badges, diagrama, link al deploy) + Entrada 5" "rol:lead,rol:qa" ""

echo "PR Fase 0"
$GH pr create --base main --head feat/fase0-contrato-datos \
  --title "Fase 0: setup, contrato de datos, confidencialidad y DAG borrador" \
  --body-file "$SCRATCH/pr_fase0.md" --label "rol:lead" --milestone "Fase 0 - Setup y contrato de datos"
