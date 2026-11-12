# Retorno causal de un algoritmo de recomendación

**TPI Decision Intelligence Capstone — Estadística Aplicada y Modelos de Regresión (EAyMR 2026, UCA)**

> Estado: **Fase 0 (setup)**. Este README se completa en la Fase 5 con badges, diagrama de la solución,
> enlace al deploy y resultados.

## Pregunta central

¿Cuál es el efecto causal de que el algoritmo de recomendación **exponga** un ítem (T) sobre la
conversión y el gasto del usuario (Y), una vez aislada la autoselección de los usuarios y de los
ítems que el propio algoritmo ya favorece (Z)? Contrastamos tres supuestos: el criterio Backdoor
(ausencia de confusión no observada), la trampa del mediador y la trampa del colisionador.

## Equipo y roles Scrum

| Integrante | Rol Scrum | Justificación |
|---|---|---|
| Verónica López Marsico | 👑 Project Lead & Scrum Master (Product Owner) | Es dueña del repositorio remoto: administra issues, milestones y merges a `main`, siempre con la firma previa del QA. |
| Lucía Báez | ⚙️ Backend & ML / Causal Engineer | Concentra el mayor volumen técnico (Fases 1 a 4: statsmodels, scikit-learn, DoWhy, EconML), por eso tiene un solo rol. |
| Antonella Robiollio | 🎨 Frontend & UI/UX Engineer + 🔍 QA & Validation Engineer | La app se construye recién en la Fase 5, así que durante las Fases 1 a 4 su carga es la auditoría de los PRs del backend (pytest, leakage, ruff). |

**Conflicto de interés del rol doble:** en los PRs que escribe Antonella (app), el veredicto
`[APROBADO]`/`[BLOQUEADO]` lo emite Lucía como QA suplente, y queda registrado en el PR.

## Estructura del repositorio

```
├── app/                 # Web app Streamlit (Diagnóstico + PCA, What-Is, What-If)
├── data/                # Script de descarga; los datos crudos no se versionan
├── docs/                # Enunciado, rúbrica, log de uso de IA
├── models/              # Modelos y artefactos serializados (joblib/json)
├── notebooks/           # TP_Integrador_Final.ipynb (notebook maestro)
├── report/              # Informe Quarto (HTML/PDF), figuras y bibliografía
├── src/tpi/             # Código modular reutilizado por notebook, app y tests
├── tests/               # pytest: leakage, dimensiones, supuestos, rangos de la app
├── BITACORA.md          # Cuaderno de ingeniería (5 entradas obligatorias)
└── requirements.txt     # Dependencias con versiones fijadas
```

## Reproducción (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ruff check .
pytest
```

Semilla global: `SEED = 42` (`src/tpi/config.py`). Nivel de significancia fijado a priori: α = 0,05.

## Flujo de trabajo

Ramas por feature → Pull Request a `main` → CI (`ruff` + `pytest`) → veredicto del QA → merge del
Project Lead. Ver `.github/pull_request_template.md`.
