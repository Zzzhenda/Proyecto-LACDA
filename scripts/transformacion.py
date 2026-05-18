"""Modulo de transformacion: feature engineering sobre las entidades limpias.

Lee solicitantes_clean y prestamos_clean, agrega 3 features derivadas
deterministicas y escribe:

  * solicitantes_transformed  (sin features extra; mantiene el contrato del schema)
  * prestamos_transformed     (rate_x_pct_income, loan_burden, has_prev_defaults)
  * data/loan_data_transformed.csv (join plano para compatibilidad ML)

Features derivadas (todas a nivel prestamo, justificadas en
notebooks/features.ipynb):

  rate_x_pct_income  -> prestamos_transformed
    loan_int_rate * loan_percent_income.
    Captura riesgo combinado tasa-ingreso. |corr| 0.46.

  loan_burden        -> prestamos_transformed
    (loan_amnt * (1 + loan_int_rate/100)) / person_income.
    Costo total del prestamo como fraccion del ingreso anual. |corr| 0.40.

  has_prev_defaults  -> prestamos_transformed
    Encoding binario de previous_loan_defaults_on_file (Yes->1, No->0).
    El predictor mas fuerte del dataset. |corr| 0.54.

Por que no hay features a nivel solicitante: el EDA mostro que credit_score
y person_age no correlacionan con loan_status en este dataset (|corr| < 0.03),
asi que fico_band y age_group no se justifican por evidencia.

El encoding del resto de categoricas y el escalado de numericas se delegan
al pipeline de sklearn en la fase de modelado (evita data leakage).
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine


CSV_OUT = Path(__file__).resolve().parent.parent / "data" / "loan_data_transformed.csv"

COLS_SOLICITANTE = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "cb_person_cred_hist_length",
    "credit_score",
]

COLS_PRESTAMO = [
    "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income",
    "previous_loan_defaults_on_file", "loan_status",
]

FEATURES_PRESTAMO = ["rate_x_pct_income", "loan_burden", "has_prev_defaults"]


def main() -> None:
    engine = get_engine()

    query = """
        SELECT
            s.person_age, s.person_gender, s.person_education, s.person_income,
            s.person_emp_exp, s.person_home_ownership, s.cb_person_cred_hist_length,
            s.credit_score,
            p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income,
            p.previous_loan_defaults_on_file, p.loan_status
        FROM solicitantes_clean s
        JOIN prestamos_clean p ON p.solicitante_id = s.id
        ORDER BY s.id
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        sys.exit("[transformacion] Las tablas clean estan vacias. Corre la limpieza primero.")

    print(f"[transformacion] Filas leidas (join): {len(df)}")

    # Features derivadas (todas a nivel prestamo)
    df["rate_x_pct_income"] = (df["loan_int_rate"] * df["loan_percent_income"]).round(4)
    df["loan_burden"] = (
        (df["loan_amnt"] * (1 + df["loan_int_rate"] / 100))
        / df["person_income"].clip(lower=1)
    ).round(4)
    df["has_prev_defaults"] = (df["previous_loan_defaults_on_file"] == "Yes").astype("int64")

    if df[FEATURES_PRESTAMO].isnull().any().any():
        sys.exit("[transformacion] FALLA: features derivadas contienen NaN")

    print(f"[transformacion] Features agregadas: {', '.join(FEATURES_PRESTAMO)}")
    print(f"[transformacion] rate_x_pct_income  mean={df['rate_x_pct_income'].mean():.4f}  "
          f"max={df['rate_x_pct_income'].max():.4f}")
    print(f"[transformacion] loan_burden        mean={df['loan_burden'].mean():.4f}  "
          f"max={df['loan_burden'].max():.4f}")
    print(f"[transformacion] has_prev_defaults  dist="
          f"{df['has_prev_defaults'].value_counts().sort_index().to_dict()}")

    # Vaciar en orden inverso por FK
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamos_transformed RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE solicitantes_transformed RESTART IDENTITY CASCADE"))

    # Cargar solicitantes_transformed (sin features extra)
    df_sol = df[COLS_SOLICITANTE].copy()
    df_sol.to_sql("solicitantes_transformed", engine, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        ids = pd.read_sql("SELECT id FROM solicitantes_transformed ORDER BY id", conn)

    # Cargar prestamos_transformed (con las 3 features derivadas)
    df_pre = df[COLS_PRESTAMO + FEATURES_PRESTAMO].copy()
    df_pre.insert(0, "solicitante_id", ids["id"].values)
    df_pre.to_sql("prestamos_transformed", engine, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        n_sol = conn.execute(text("SELECT COUNT(*) FROM solicitantes_transformed")).scalar()
        n_pre = conn.execute(text("SELECT COUNT(*) FROM prestamos_transformed")).scalar()

    print(f"[transformacion] Filas en solicitantes_transformed: {n_sol}")
    print(f"[transformacion] Filas en prestamos_transformed:    {n_pre}")

    # Export CSV (join plano para ML)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"[transformacion] CSV transformado guardado en {CSV_OUT}")


if __name__ == "__main__":
    main()
