# Datos

Fuente: competencia de Kaggle **Personalize Expedia Hotel Searches — ICDM 2013**
(`expedia-personalized-sort`). Kaggle publica `data.zip` (con `train.csv` y `test.csv`) y tres zips
de benchmarks que no usamos. `test.csv` no trae `position`, `click_bool`, `booking_bool` ni
`gross_bookings_usd`, así que **solo usamos `train.csv`**.

## Confidencialidad (reglas de la competencia)

> "The data set and resulting model parameters are considered confidential information that cannot
> be shared with people who are not participating in the contest."

Mitigaciones (decisión de Verónica, 12/11/2026: seguir con Expedia sin consultar a la cátedra;
riesgo residual documentado en la BITÁCORA, Entradas 1 y 4):

| Qué | Dónde | ¿Se versiona? |
|---|---|---|
| `data.zip` y `train.csv` (crudo) | `~/data/tpi-eaymr/raw` (fuera del repo y de OneDrive; configurable con `TPI_RAW_DIR`) | No |
| Muestra de trabajo (`expedia_muestra.parquet`) | `data/processed/` | No (`.gitignore`) |
| Manifiesto (SHA-256, tamaños, semilla; sin registros) | `data/manifiesto.json` | Sí |

`tests/test_confidencialidad.py` falla si git trackea cualquier archivo con formato de datos
(`.csv`, `.parquet`, `.zip`, `.pkl`, …). La app publica solo agregados, nunca registros
individuales.

## Requisitos (una vez)

1. Cuenta en Kaggle y reglas aceptadas en <https://www.kaggle.com/c/expedia-personalized-sort/rules>.
2. Autenticación de la API (kaggle ≥ 2): `kaggle auth login`, o el token de
   <https://www.kaggle.com/settings/api> guardado en `~/.kaggle/access_token`.

## Uso

```powershell
python data/download_data.py                       # descarga data.zip, extrae train.csv, muestrea
python data/download_data.py --solo-muestra        # rehace la muestra sin descargar
pytest -m data                                     # contrato de datos y premisa de random_bool
```

`data.zip` usa compresión **Deflate64** (método 9), que el módulo `zipfile` de Python no
implementa: el script la descomprime con `inflate64` y verifica tamaño y CRC-32.

## Plan B (Online Shoppers, CC BY 4.0)

```powershell
python data/download_plan_b.py
$env:TPI_DATASET = "online_shoppers"   # todo el pipeline pasa a usar el contrato del plan B
```

Ver [`docs/plan_b.md`](../docs/plan_b.md).
