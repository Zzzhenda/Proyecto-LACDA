"""Modulo de ingesta: carga `data/loan_data.csv` separando las dos entidades
del diseño tecnico en sus tablas correspondientes.

  solicitantes_raw  <- datos demograficos y financieros del solicitante.
  prestamos_raw     <- caracteristicas del prestamo y resultado (FK a solicitante).

Idempotente: cada corrida vacia ambas tablas antes de insertar.
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine


CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "loan_data.csv"

COLS_SOLICITANTE = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "cb_person_cred_hist_length",
    "credit_score",
]

COLS_PRESTAMO = [
    "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income",
    "previous_loan_defaults_on_file", "loan_status",
]


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"[ingesta] No se encontro {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"[ingesta] CSV leido: {df.shape[0]} filas, {df.shape[1]} columnas")

    engine = get_engine()

    # Vaciar en orden inverso (primero hijos, luego padres) por FK
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamos_raw RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE solicitantes_raw RESTART IDENTITY CASCADE"))

    # Cargar solicitantes y recuperar los IDs generados
    df_sol = df[COLS_SOLICITANTE].copy()
    df_sol.to_sql("solicitantes_raw", engine, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        ids = pd.read_sql("SELECT id FROM solicitantes_raw ORDER BY id", conn)

    # Cargar prestamos con FK al solicitante correspondiente (mismo orden)
    df_pre = df[COLS_PRESTAMO].copy()
    df_pre.insert(0, "solicitante_id", ids["id"].values)
    df_pre.to_sql("prestamos_raw", engine, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        n_sol = conn.execute(text("SELECT COUNT(*) FROM solicitantes_raw")).scalar()
        n_pre = conn.execute(text("SELECT COUNT(*) FROM prestamos_raw")).scalar()

    print(f"[ingesta] Filas en solicitantes_raw: {n_sol}")
    print(f"[ingesta] Filas en prestamos_raw:    {n_pre}")


if __name__ == "__main__":
    main()