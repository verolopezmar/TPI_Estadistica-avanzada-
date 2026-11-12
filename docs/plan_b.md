# Plan B: Online Shoppers semisintético (evaluado el 12/11/2026)

> Para: el equipo. Se activa si la cátedra no habilita el uso de Expedia con repo y app públicos
> (reglas de Kaggle: datos y "resulting model parameters" confidenciales).

## Fuente verificada

- UCI Machine Learning Repository #468, *Online Shoppers Purchasing Intention Dataset*.
  Sakar, C. & Kastro, Y. (2018), doi:10.24432/C5F88Q. Licencia **CC BY 4.0**, que permite
  repo, app y presentación públicos con atribución.
- Archivo descargado y verificado (`data/manifiesto_plan_b.json`): 12.330 sesiones × 18
  columnas, sin faltantes, `Revenue` = 1 en 1.908 sesiones (15,5 %).

## Diseño

Covariables y tasa base de compra **reales**; tratamiento, mediador y efecto **simulados** con un
mecanismo declarado en `src/tpi/plan_b.py`. El efecto verdadero queda en la columna `tau_real`,
así que cada estimador (DoWhy, IPW, DML) se audita contra la verdad.

| Rol | Columna | Origen |
|---|---|---|
| T | `expuesto` | Simulado: el "recomendador" expone según `PageValues`, `ProductRelated` y visitante recurrente |
| M | `M` | Simulado: interacción con el ítem recomendado |
| Y | `Y` | Semisintético: probabilidad base ajustada sobre `Revenue` real + efecto por M |
| Z | `ProductRelated`, `ProductRelated_Duration`, `PageValues`, `BounceRates`, `returning`, `Weekend`, `SpecialDay` | Reales |

Resultado sobre los datos reales (semilla 42): ATE verdadero = 0,0646; diferencia ingenua de
medias = 0,2906; sesgo de confusión = +0,2260. La confusión es fuerte y conocida, lo que lo hace
didáctico para la Fase 4.

## Costo de activarlo

- **Código:** `$env:TPI_DATASET = "online_shoppers"`. El pipeline usa `tpi.contrato.contrato_activo()`
  y no importa nada de Expedia (`tests/test_contrato.py`).
- **Pierde:** el experimento natural (`random_bool`), el sesgo de selección real del dataset y
  la estructura agrupada por búsqueda. La parte causal pasa a validarse contra una verdad
  simulada en lugar de contra un cuasi-experimento real.
- **Gana:** sin restricción de licencia; N = 12.330 (≥ 1.500) y 17 variables explicativas que
  combinan continuas, discretas y categóricas, como pide el enunciado.
- **Fases 1 a 3:** se rehacen sobre las variables reales (`Revenue` como Y para el torneo
  predictivo). Estimamos 1 día de trabajo si se activa antes del 14/11.
