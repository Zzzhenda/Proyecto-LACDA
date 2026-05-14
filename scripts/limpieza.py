"""Modulo de procesamiento: limpieza de las entidades Solicitante y Prestamo.

Lee solicitantes_raw y prestamos_raw (joined por solicitante_id), aplica las
reglas del diseño tecnico (cap. 9) y escribe:

  * solicitantes_clean + prestamos_clean  en Postgres
  * data/loan_data_clean.csv              (join plano para compatibilidad ML)

Reglas aplicadas:

  Numericas (Solicitante)
    person_age                  : 18 <= x <= 100
    person_emp_exp              : 0 <= x <= person_age - 18
    person_income               : x >= 0
    credit_score                : 300 <= x <= 850
    cb_person_cred_hist_length  : 0 <= x <= person_age

  Numericas (Prestamo)
    loan_amnt                   : x > 0
    loan_int_rate               : 5 <= x <= 30
    loan_percent_income         : 0 <= x <= 1

  Categoricas (dominios cerrados)
    person_gender                  : male, female
    person_education               : High School, Bachelor, Master, Associate, Doctorate
    person_home_ownership          : RENT, OWN, MORTGAGE, OTHER
    loan_intent                    : PERSONAL, EDUCATION, MEDICAL, VENTURE,
                                     DEBTCONSOLIDATION, HOMEIMPROVEMENT
    previous_loan_defaults_on_file : Yes, No
    loan_status                    : 0, 1
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine


"""CSV_OUT = Path(__file__).resolve().parent.parent / "data" / "loan_data_clean.csv" """
#para crear 2 csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_OUT_SOL = DATA_DIR / "solicitantes_clean.csv"
CSV_OUT_PRE = DATA_DIR / "prestamos_clean.csv"


COLS_SOLICITANTE = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "cb_person_cred_hist_length",
    "credit_score",
]

COLS_PRESTAMO = [
    "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income",
    "previous_loan_defaults_on_file", "loan_status",
]

CAT_DOMAINS = {
    "person_gender": {"male", "female"},
    "person_education": {"High School", "Bachelor", "Master", "Associate", "Doctorate"},
    "person_home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "loan_intent": {
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "DEBTCONSOLIDATION", "HOMEIMPROVEMENT",
    },
    "previous_loan_defaults_on_file": {"Yes", "No"},
}

## REGLAS A APLICAR 
def aplicar_reglas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recibe el join completo, devuelve el df limpio y estadisticas."""
    stats: dict[str, int] = {}

    n0 = len(df)
    df = df.drop_duplicates()
    stats["duplicados"] = n0 - len(df)

    n1 = len(df)
    df = df.dropna()
    stats["nulos"] = n1 - len(df)

    n2 = len(df)
    df = df[df["person_age"].between(18, 100)]
    df = df[df["person_emp_exp"].between(0, df["person_age"] - 18)]
    df = df[df["person_income"] >= 0]
    df = df[df["credit_score"].between(300, 850)]
    df = df[df["loan_amnt"] > 0]
    df = df[df["loan_int_rate"].between(5, 30)]
    df = df[df["loan_percent_income"].between(0, 1)]
    df = df[df["cb_person_cred_hist_length"].between(0, df["person_age"])]
    stats["fuera_de_rango"] = n2 - len(df)

    n3 = len(df)
    for col, dominio in CAT_DOMAINS.items():
        df = df[df[col].isin(dominio)]
    df = df[df["loan_status"].isin([0, 1])]
    stats["categoricos_invalidos"] = n3 - len(df)

    return df.reset_index(drop=True), stats


def main() -> None:
    engine = get_engine()

    # Leer el join completo desde raw
    query = """
        SELECT
            s.person_age, s.person_gender, s.person_education, s.person_income,
            s.person_emp_exp, s.person_home_ownership, s.cb_person_cred_hist_length,
            s.credit_score,
            p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income,
            p.previous_loan_defaults_on_file, p.loan_status
        FROM solicitantes_raw s
        JOIN prestamos_raw p ON p.solicitante_id = s.id
        ORDER BY s.id
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        sys.exit("[limpieza] Las tablas raw estan vacias. Corre la ingesta primero.")

    n_inicial = len(df)
    print(f"[limpieza] Filas leidas (join): {n_inicial}")

    df, stats = aplicar_reglas(df)

    print(f"[limpieza] Removidas por duplicados:     {stats['duplicados']}")
    print(f"[limpieza] Removidas por nulos:          {stats['nulos']}")
    print(f"[limpieza] Removidas por fuera de rango: {stats['fuera_de_rango']}")
    print(f"[limpieza] Removidas por categoricos:    {stats['categoricos_invalidos']}")
    print(f"[limpieza] Filas finales: {len(df)} (de {n_inicial})")

    # Vaciar en orden inverso por FK
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamos_clean RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE solicitantes_clean RESTART IDENTITY CASCADE"))

    # Cargar solicitantes_clean
    df[COLS_SOLICITANTE].to_sql(
        "solicitantes_clean", engine, if_exists="append", index=False, chunksize=1000
    )

    with engine.connect() as conn:
        ids = pd.read_sql("SELECT id FROM solicitantes_clean ORDER BY id", conn)

    # Cargar prestamos_clean con FK
    df_pre = df[COLS_PRESTAMO].copy()
    df_pre.insert(0, "solicitante_id", ids["id"].values)
    df_pre.to_sql("prestamos_clean", engine, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        n_sol = conn.execute(text("SELECT COUNT(*) FROM solicitantes_clean")).scalar()
        n_pre = conn.execute(text("SELECT COUNT(*) FROM prestamos_clean")).scalar()

    
    # --- NUEVA LÓGICA DE EXPORTACIÓN CSV ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Preparar el DataFrame de solicitantes con su ID para el CSV
    df_sol = df[COLS_SOLICITANTE].copy()
    df_sol.insert(0, "id", ids["id"].values)
    
    # Guardar ambos archivos
    df_sol.to_csv(CSV_OUT_SOL, index=False)
    df_pre.to_csv(CSV_OUT_PRE, index=False) # df_pre ya tiene "solicitante_id" de la línea 81
    
    print(f"[limpieza] CSV de solicitantes guardado en {CSV_OUT_SOL}")
    print(f"[limpieza] CSV de préstamos guardado en {CSV_OUT_PRE}")

if __name__ == "__main__":
    main()