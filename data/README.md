# Datos

Fuente: competencia de Kaggle **Personalize Expedia Hotel Searches — ICDM 2013**
(`expedia-personalized-sort`), archivo `train.csv`. Los datos **no se versionan** (reglas de la
competencia y tamaño): se regeneran con el script.

## Requisitos (una vez)

1. Cuenta en Kaggle y reglas aceptadas en <https://www.kaggle.com/c/expedia-personalized-sort/rules>.
2. Autenticación de la API (kaggle ≥ 2): `kaggle auth login`, o el token de
   <https://www.kaggle.com/settings/api> guardado en `%USERPROFILE%\.kaggle\access_token`.

## Uso

```powershell
python data/download_data.py                       # descarga + muestra (15.000 búsquedas por estrato)
python data/download_data.py --solo-muestra        # rehace la muestra sin descargar
pytest -m data                                     # contrato de datos y test de random_bool
```

- `raw/train.csv`: archivo original.
- `processed/expedia_muestra.parquet`: búsquedas **completas**, estratificadas por `random_bool`.
- `processed/manifiesto.json`: SHA-256 del crudo, semilla, tamaños y violaciones del contrato.
