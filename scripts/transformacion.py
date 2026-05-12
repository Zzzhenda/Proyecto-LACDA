"""Modulo de transformacion: feature engineering sobre las entidades limpias.

Lee solicitantes_clean y prestamos_clean, agrega tres features derivadas
deterministicas y escribe:

  * solicitantes_transformed  (fico_band, age_group)
  * prestamos_transformed     (rate_x_pct_income)
  * data/loan_data_transformed.csv (join plano para compatibilidad ML)

Features derivadas:

  fico_band         -> solicitantes_transformed
    Banda FICO oficial segun credit_score (entero 1..5).

  age_group         -> solicitantes_transformed
    Segmento etario (entero 1=joven, 2=adulto, 3=senior).

  rate_x_pct_income -> prestamos_transformed
    loan_int_rate * loan_percent_income (captura riesgo combinado).

El encoding de categoricas y el escalado de numericas se delegan al
pipeline de sklearn en la fase de modelado (evitar data leakage).
"""

import sys
from pathlib import Path

import numpy as np
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

FICO_BANDS = [
    (300, 579, 1),  # Poor
    (580, 669, 2),  # Fair
    (670, 739, 3),  # Good
    (740, 799, 4),  # Very Good
    (800, 850, 5),  # Exceptional
]

AGE_GROUPS = [
    (18, 29, 1),   # Joven
    (30, 54, 2),   # Adulto
    (55, 100, 3),  # Senior
]


def _bandify(series: pd.Series, bandas: list[tuple[int, int, int]]) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    for lo, hi, valor in bandas:
        out.loc[series.between(lo, hi, inclusive="both")] = valor
    return out.astype("Int64")


def main() -> None:
    engine = get_engine()

    # Leer join completo desde clean
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

    # Features derivadas
    df["fico_band"] = _bandify(df["credit_score"], FICO_BANDS)
    df["age_group"] = _bandify(df["person_age"], AGE_GROUPS)
    df["rate_x_pct_income"] = (df["loan_int_rate"] * df["loan_percent_income"]).round(4)

    nuevos = ["fico_band", "age_group", "rate_x_pct_income"]
    if df[nuevos].isnull().any().any():
        sys.exit("[transformacion] FALLA: features derivadas contienen NaN")

    print(f"[transformacion] Features agregadas: {', '.join(nuevos)}")
    print(f"[transformacion] Distribucion fico_band: "
          f"{df['fico_band'].value_counts().sort_index().to_dict()}")
    print(f"[transformacion] Distribucion age_group: "
          f"{df['age_group'].value_counts().sort_index().to_dict()}")

    # Vaciar en orden inverso por FK
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamos_transformed RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE solicitantes_transformed RESTART IDENTITY CASCADE"))

    # Cargar solicitantes_transformed (con sus dos features)
    df_sol = df[COLS_SOLICITANTE + ["fico_band", "age_group"]].copy()
    df_sol.to_sql("solicitantes_transformed", engine, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        ids = pd.read_sql("SELECT id FROM solicitantes_transformed ORDER BY id", conn)

    # Cargar prestamos_transformed (con su feature)
    df_pre = df[COLS_PRESTAMO + ["rate_x_pct_income"]].copy()
    df_pre.insert(0, "solicitante_id", ids["id"].values)
    df_pre.to_sql("prestamos_transformed", engine, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        n_sol = conn.execute(text("SELECT COUNT(*) FROM solicitantes_transformed")).scalar()
        n_pre = conn.execute(text("SELECT COUNT(*) FROM prestamos_transformed")).scalar()

    print(f"[transformacion] Filas en solicitantes_transformed: {n_sol}")
    print(f"[transformacion] Filas en prestamos_transformed:    {n_pre}")

    # Export CSV (join plano)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"[transformacion] CSV transformado guardado en {CSV_OUT}")


if __name__ == "__main__":
    main()