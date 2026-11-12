# Retorno causal de un algoritmo de recomendación

[![Quality Gate](https://github.com/verolopezmar/TPI_Estadistica-avanzada-/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/verolopezmar/TPI_Estadistica-avanzada-/actions/workflows/quality-gate.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-2a78d6)
![Ruff](https://img.shields.io/badge/lint-ruff-1baf7a)
![Tests](https://img.shields.io/badge/tests-pytest-eb6834)
[![App](https://img.shields.io/badge/app-Streamlit-ff4b4b)](PENDIENTE_URL_DEPLOY)

**TPI Decision Intelligence Capstone — Estadística Aplicada y Modelos de Regresión (EAyMR 2026, UCA)**

> **¿El top-5 vende, o solo muestra lo que se vendería igual?** Medimos el efecto causal de que el
> recomendador de Expedia exponga un hotel en los primeros cinco lugares sobre su probabilidad de
> reserva, y lo contrastamos con un experimento natural: un subconjunto de búsquedas mostró los
> hoteles en orden aleatorio.

### Resultado principal

| Estimación del efecto del top-5 sobre la reserva (OR dentro de la búsqueda) | OR | IC 95 % |
|---|---|---|
| **Orden aleatorio, ajustado (benchmark causal)** | **2,11** | [1,90; 2,35] |
| Orden del algoritmo, ajustado por todos los confusores observados | 3,38 | [3,25; 3,51] |
| Orden del algoritmo, comparación ingenua | 5,38 | [5,20; 5,57] |

- **El top-5 sí causa reservas.**
- **Con datos del algoritmo, el efecto se sobreestima aun después de ajustar:**
  - cociente de OR 1,60 (IC 95 % bootstrap por búsqueda [1,42; 1,80], 1000 réplicas);
  - en diferencia de riesgos, cociente de efectos marginales 1,65 [1,41; 1,98];
  - se mantiene re-ponderando a la composición de la muestra aleatoria.

  Los datos son incompatibles con que el ajuste por lo observable elimine el sesgo: el ranker usa
  información que no observamos.
- Con orden aleatorio, el top-5 casi triplica la probabilidad de clic: el clic es el paso
  obligatorio del embudo. El efecto es mayor para hoteles baratos para su búsqueda y bien ubicados.
- Detalle en el [informe](report/informe.qmd) y en el notebook (Fase 4).

## Entregables

| Entregable | Dónde |
|---|---|
| App en producción (Diagnóstico + PCA, What-Is, What-If y copiloto de decisión como bonus) | PENDIENTE_URL_DEPLOY · código en [`app/`](app/) |
| Notebook maestro (Restart & Run All) | [`notebooks/TP_Integrador_Final.ipynb`](notebooks/TP_Integrador_Final.ipynb) |
| Informe técnico (Quarto, HTML y PDF) | [`report/informe.qmd`](report/informe.qmd) → `report/informe.html`, `report/informe.pdf` |
| Diapositivas de la defensa (PDF) | [`report/slides.pdf`](report/slides.pdf) |
| Bitácora de ingeniería (5 entradas) | [`BITACORA.md`](BITACORA.md) |
| Registro de uso de IA | [`docs/ia_log.md`](docs/ia_log.md) |
| Guía de defensa por rol | [`docs/defensa.md`](docs/defensa.md) |

## Diagrama de la solución

```
 Kaggle (data.zip) ──► data/download_data.py ──► crudo fuera del repo (~/data/tpi-eaymr/raw)
                                                   │  muestra estratificada por random_bool
                                                   ▼
                          data/processed/expedia_muestra.parquet  (ignorado por git)
                                                   │
          ┌────────────────────── notebooks/TP_Integrador_Final.ipynb ──────────────────────┐
          │ Fase 1  auditoría forense · MCAR/MAR/MNAR · contrastes · ANOVA de Welch          │
          │ Fase 2  MCO + Gauss-Markov · Cook/leverage · VIF · PCA solo en Train            │
          │ Fase 3  Logística · Naive Bayes · CART podado · Random Forest · calibración     │
          │ Fase 4  DAG · DoWhy (regresión, IPW) · logit condicional R=0 vs R=1 · DML/CATE  │
          │         trampas del mediador y del colisionador · refutaciones                   │
          └──────────────────────────────────────┬──────────────────────────────────────────┘
                                                 ▼
                     models/  (solo agregados + RF compacto, versionados)
                                                 ▼
                         app/app.py (Streamlit Community Cloud)
```

Código reutilizable en [`src/tpi/`](src/tpi/): contrato de datos, auditoría, inferencia,
regresión/PCA, modelos, causal, DAG y artefactos. Cada módulo tiene sus tests en [`tests/`](tests/).

## Equipo y roles Scrum

| Integrante | Rol Scrum |
|---|---|
| Verónica López Marsico | 👑 Project Lead & Scrum Master (Product Owner) |
| Lucía Báez | ⚙️ Backend & ML / Causal Engineer |
| Antonella Robiollio | 🎨 Frontend & UI/UX Engineer + 🔍 QA & Validation Engineer |

**Organización real del trabajo (desde el 12/11/2026):** Verónica ejecuta y valida todo el trabajo
con asistencia de Claude Code, incluidos los veredictos `[APROBADO]`/`[BLOQUEADO]` de los PRs. Los
roles de la tabla se mantienen para la defensa oral. Cada PR registra quién lo revisó realmente
(ver [`BITACORA.md`](BITACORA.md) y [`docs/ia_log.md`](docs/ia_log.md)).

## Reproducción paso a paso (Windows / PowerShell)

> Creá el entorno virtual **fuera** de carpetas sincronizadas (OneDrive, Dropbox, Google Drive):
> la sincronización bloquea y corrompe archivos del venv.

```powershell
# 1. Entorno (Python 3.11)
py -3.11 -m venv $HOME\.venvs\tpi-eaymr
& $HOME\.venvs\tpi-eaymr\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Datos (requiere cuenta de Kaggle, aceptar las reglas de la competencia y `kaggle auth login`)
python data/download_data.py          # descarga data.zip, extrae train.csv fuera del repo y muestrea

# 3. Calidad
ruff check .
ruff format --check .
pytest                                # incluye tests de datos si existe la muestra

# 4. Notebook maestro (Restart & Run All) y artefactos de la app
jupyter nbconvert --to notebook --execute --inplace notebooks/TP_Integrador_Final.ipynb

# 5. Informe y diapositivas
quarto render report/informe.qmd      # HTML y PDF
quarto render report/slides.qmd       # PDF (beamer)

# 6. App local
streamlit run app/app.py
```

Semilla global `SEED = 42` (`src/tpi/config.py`); α = 0,05 fijado a priori. Detalle de los datos y
del plan B (Online Shoppers, CC BY 4.0) en [`data/README.md`](data/README.md).

## Deploy en Streamlit Community Cloud

La app solo necesita lo que está versionado en el repo: `app/` y `models/` (artefactos agregados y
RF compacto). No lee datos.

1. Que la rama con la app (`main`, después de mergear la Fase 5) esté en GitHub.
2. Entrar a <https://share.streamlit.io> con la cuenta de GitHub y autorizar el acceso al repo.
3. **Create app** → *Deploy a public app from GitHub* y completar:
   - **Repository:** `verolopezmar/TPI_Estadistica-avanzada-`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
   - **App URL:** opcional, por ejemplo `tpi-eaymr-top5`.
4. **Advanced settings** → **Python version: 3.11**. Opcional (copiloto, bonus): en **Secrets** cargar
   `ANTHROPIC_API_KEY = "..."`. Sin la clave, la pestaña del copiloto queda desactivada y el resto funciona.
5. **Deploy.** Streamlit Cloud instala `app/requirements.txt` (el que está junto a `app.py`, mínimo
   y con las mismas versiones con que se serializó el modelo).
6. Copiar la URL pública en el badge y en la tabla de entregables de este README (reemplazar
   `PENDIENTE_URL_DEPLOY`).

Local: `streamlit run app/app.py`.

## Confidencialidad de los datos

Las reglas de la competencia declaran confidenciales los datos y los "resulting model parameters".
Decisión de Verónica (12/11/2026): seguir con Expedia con mitigaciones verificadas por tests.

- Ningún archivo de datos en el repo (`tests/test_confidencialidad.py`).
- El crudo queda fuera del repo y de OneDrive.
- La app solo usa agregados con supresión de celdas menores a 10 filas, y un Random Forest con
  hojas de al menos 100 filas (`tests/test_app.py`).

El riesgo residual está documentado en la BITÁCORA (Entradas 1 y 4).

## Flujo de trabajo

Ramas por fase → Pull Request a `main` → CI (`ruff` + `pytest`) → veredicto QA → merge con
"Create a merge commit". Scripts de GitHub en [`scripts/github/`](scripts/github/).
