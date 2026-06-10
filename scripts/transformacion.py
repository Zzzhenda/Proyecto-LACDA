"""Etapa 4 — Transformacion (feature engineering).

Lee data/loan_data_clean.csv, agrega 3 features derivadas deterministicas
y escribe data/loan_data_transformed.csv (el dataset final que la etapa
de validacion audita y carga).

Features derivadas (justificadas por correlacion con loan_status en el EDA):

  rate_x_pct_income   loan_int_rate * loan_percent_income.
                      Riesgo combinado tasa-carga de ingreso. |corr| ~0.46.

  loan_burden         (loan_amnt * (1 + loan_int_rate/100)) / person_income.
                      Costo total del prestamo como fraccion del ingreso
                      anual. |corr| ~0.37.

  has_prev_defaults   Encoding binario de previous_loan_defaults_on_file
                      (Yes->1, No->0). El predictor mas fuerte. |corr| ~0.54.

No se crean features sobre credit_score ni person_age: el EDA mostro
|corr| < 0.03 con loan_status en este dataset. El encoding del resto de
categoricas y el escalado de numericas se delegan al pipeline de sklearn
de la fase de modelado (hacerlo aqui causaria data leakage).
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transformacion")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_CLEAN = DATA_DIR / "loan_data_clean.csv"
CSV_TRANSFORMED = DATA_DIR / "loan_data_transformed.csv"

# Features que esta etapa agrega (validacion.py las audita).
FEATURES_DERIVADAS = ["rate_x_pct_income", "loan_burden", "has_prev_defaults"]


def main() -> None:
    log.info("=== INICIO TRANSFORMACION ===")

    if not CSV_CLEAN.exists():
        log.error(f"No existe {CSV_CLEAN}. Corre la limpieza primero.")
        sys.exit(1)

    df = pd.read_csv(CSV_CLEAN)
    log.info(f"Filas leidas del dataset limpio: {len(df)}")

    df["rate_x_pct_income"] = (df["loan_int_rate"] * df["loan_percent_income"]).round(4)
    df["loan_burden"] = (
        (df["loan_amnt"] * (1 + df["loan_int_rate"] / 100))
        / df["person_income"].clip(lower=1)
    ).round(4)
    df["has_prev_defaults"] = (df["previous_loan_defaults_on_file"] == "Yes").astype("int64")

    if df[FEATURES_DERIVADAS].isnull().any().any():
        log.error("Features derivadas contienen NaN.")
        sys.exit(1)

    log.info(f"Features agregadas: {', '.join(FEATURES_DERIVADAS)}")
    log.info(f"  rate_x_pct_income  mean={df['rate_x_pct_income'].mean():.4f}  "
             f"max={df['rate_x_pct_income'].max():.4f}")
    log.info(f"  loan_burden        mean={df['loan_burden'].mean():.4f}  "
             f"max={df['loan_burden'].max():.4f}")
    log.info(f"  has_prev_defaults  dist="
             f"{df['has_prev_defaults'].value_counts().sort_index().to_dict()}")

    df.to_csv(CSV_TRANSFORMED, index=False)
    log.info(f"Dataset transformado escrito: {CSV_TRANSFORMED.name} ({len(df)} filas)")
    log.info("=== TRANSFORMACION OK ===")


if __name__ == "__main__":
    main()
