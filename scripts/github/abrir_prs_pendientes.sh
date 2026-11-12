#!/usr/bin/env bash
# Abre, en orden, los PRs pendientes de las Fases 1 a 5 hacia main (el de la Fase 0 es el #38).
#
# Uso (Windows, Git Bash, desde la raíz del repo, con `gh auth status` OK):
#   & "C:\Program Files\Git\bin\bash.exe" scripts/github/abrir_prs_pendientes.sh
#
# Cada rama sale de la anterior, así que cada PR muestra también los commits de las fases previas
# hasta que esas se mergeen. Mergear en orden con "Create a merge commit" (no squash): al mergear
# la Fase N, el PR de la Fase N+1 queda mostrando solo sus propios commits.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
GH="${GH:-gh}"
if ! command -v "$GH" >/dev/null 2>&1; then GH="/c/Program Files/GitHub CLI/gh.exe"; fi
"$GH" auth status >/dev/null

abrir() { # rama, título, cuerpo, etiqueta, milestone
  if "$GH" pr view "$1" --json number >/dev/null 2>&1; then
    echo "[=] Ya existe un PR para $1: $("$GH" pr view "$1" --json url --jq .url)"
  else
    "$GH" pr create --base main --head "$1" --title "$2" --body-file "$DIR/$3" --label "$4" --milestone "$5"
  fi
}

abrir feat/fase1-inferencia "Fase 1: inferencia clásica, auditoría forense, Zc y ANOVA" \
  pr_fase1.md "rol:backend" "Fase 1 - Inferencia clásica y ANOVA"
abrir feat/fase2-gauss-markov "Fase 2: Gauss-Markov, influyentes, VIF y PCA sin fuga" \
  pr_fase2.md "rol:backend" "Fase 2 - Gauss-Markov y PCA"
abrir feat/fase3-ml "Fase 3: torneo predictivo, calibración y explicabilidad" \
  pr_fase3.md "rol:backend" "Fase 3 - Torneo ML y calibración"
abrir feat/fase4-causal "Fase 4: inferencia causal, DML/CATE, refutaciones y ensayo" \
  pr_fase4.md "rol:backend" "Fase 4 - Inferencia causal"
abrir feat/fase5-entrega "Fase 5: app, informe, slides, README y bitácora completa" \
  pr_fase5.md "rol:frontend" "Fase 5 - Scrum, bitácora y deploy"
echo "Listo. PRs abiertos:"
"$GH" pr list --state open --json number,title,url --jq '.[] | "#\(.number) \(.title) \(.url)"'
