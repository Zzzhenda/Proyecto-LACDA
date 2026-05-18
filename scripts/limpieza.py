"""
Modulo de procesamiento: limpieza de las entidades Solicitante y Prestamo.

Estrategia (alineada con metodologia DataOps de la asignatura):

  1. Carga del join solicitantes_raw + prestamos_raw desde Postgres.
  2. Eliminacion de duplicados exactos (unica operacion que reduce filas).
  3. Imputacion de nulos (mediana / moda) para no perder filas por celdas
     vacias aisladas.
  4. Reglas duras del cap. 9 del diseño tecnico, aplicadas en dos bloques:
       a) Variables individuales con rango cerrado -> clip por columna.
       b) Reglas cruzadas (emp_exp <= age-18, cred_hist <= age) -> clip
          usando el valor FINAL de person_age, asi la consistencia se
          mantiene aunque hayan otras transformaciones.
  5. Dominios categoricos cerrados -> valor invalido se reemplaza por la moda.
  6. Winsorizacion al 5% solo sobre columnas SIN rango duro definido en el
     cap. 9 (person_income, loan_amnt). Las variables con contrato estricto
     (person_age, credit_score, loan_int_rate, loan_percent_income) no se
     winsorizan porque eso destruiria informacion legitima dentro del rango
     valido.

El orden importa: si winsorizaramos person_age despues de fijar emp_exp,
la regla cruzada se romperia (bug detectado en la version anterior).

Salidas:
  * tabla solicitantes_clean en Postgres (con id autogenerado).
  * tabla prestamos_clean en Postgres (con FK a solicitantes_clean).
  * data/solicitantes_clean.csv
  * data/prestamos_clean.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import get_engine



class Winsorizer:
    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).astype("float64")
        for col in self.columns_:
            lo = X[col].quantile(self.limits[0])
            hi = X[col].quantile(1 - self.limits[1])
            X[col] = np.clip(X[col], lo, hi)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------
# Logging y constantes
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[limpieza] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

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

# Columnas a las que SI aplicar Winsorizer al 5%.
# Excluye variables con rango duro definido en el cap. 9 (person_age,
# credit_score, loan_int_rate, loan_percent_income) para no destruir
# informacion valida dentro de esos rangos.
WINSORIZE_COLS = ["person_income", "loan_amnt"]


# ---------------------------------------------------------
# Componentes
# ---------------------------------------------------------
def extraer_datos_raw(engine) -> pd.DataFrame:
    """Devuelve el join completo solicitantes_raw + prestamos_raw."""
    logging.info("Leyendo join raw desde Postgres...")
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
    logging.info(f"Filas leidas: {len(df)}")
    return df


def remover_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Duplicados removidos: {n0 - len(df)}")
    return df


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Mediana para numericas, moda para categoricas. Evita perder filas."""
    df = df.copy()
    total = int(df.isnull().sum().sum())
    if total == 0:
        logging.info("Sin nulos que imputar.")
        return df

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    logging.info(f"Nulos imputados: {total}")
    return df


def aplicar_reglas_rango(df: pd.DataFrame) -> pd.DataFrame:
    """Reglas duras del cap. 9 sobre variables individuales (clip)."""
    df = df.copy()

    df["person_age"] = df["person_age"].clip(18, 100)
    df["person_income"] = df["person_income"].clip(lower=0)
    df["credit_score"] = df["credit_score"].clip(300, 850)
    df["loan_int_rate"] = df["loan_int_rate"].clip(5, 30)
    df["loan_percent_income"] = df["loan_percent_income"].clip(0, 1)

    # loan_amnt: regla es x > 0; los 0 se imputan a la mediana del resto
    if (df["loan_amnt"] <= 0).any():
        mediana = df.loc[df["loan_amnt"] > 0, "loan_amnt"].median()
        df.loc[df["loan_amnt"] <= 0, "loan_amnt"] = mediana

    logging.info("Reglas de rango aplicadas (cap. 9).")
    return df


def aplicar_reglas_cruzadas(df: pd.DataFrame) -> pd.DataFrame:
    """Consistencia entre columnas usando el person_age YA fijado."""
    df = df.copy()
    df["person_emp_exp"] = df["person_emp_exp"].clip(lower=0, upper=df["person_age"] - 18)
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].clip(
        lower=0, upper=df["person_age"]
    )
    logging.info("Reglas cruzadas (emp_exp, cred_hist_length) aplicadas.")
    return df


def aplicar_dominios_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """Valores fuera del dominio cerrado se reemplazan por la moda."""
    df = df.copy()
    for col, dominio in CAT_DOMAINS.items():
        if col not in df.columns:
            continue
        invalidos = (~df[col].isin(dominio)).sum()
        if invalidos:
            moda = df.loc[df[col].isin(dominio), col].mode().iloc[0]
            df.loc[~df[col].isin(dominio), col] = moda
            logging.info(f"  {col}: {invalidos} valores fuera de dominio -> {moda!r}")

    # loan_status: binario {0, 1}
    if "loan_status" in df.columns:
        invalidos = (~df["loan_status"].isin([0, 1])).sum()
        if invalidos:
            df.loc[~df["loan_status"].isin([0, 1]), "loan_status"] = 0
            logging.info(f"  loan_status: {invalidos} valores fuera de {{0,1}} -> 0")
    return df


def aplicar_winsorizacion(df: pd.DataFrame) -> pd.DataFrame:
    """Winsoriza al 5% solo columnas sin contrato duro de rango."""
    df = df.copy()
    cols = [c for c in WINSORIZE_COLS if c in df.columns]
    if not cols:
        logging.warning("No hay columnas para Winsorizar.")
        return df

    wins = Winsorizer(limits=(0.05, 0.05))
    df[cols] = wins.fit_transform(df[cols])
    logging.info(f"Winsorizacion 5% aplicada sobre: {cols}")
    return df


def persistir_y_exportar(df: pd.DataFrame, engine) -> None:
    """Persiste en Postgres respetando integridad referencial y exporta CSVs."""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamos_clean RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE solicitantes_clean RESTART IDENTITY CASCADE"))

    df[COLS_SOLICITANTE].to_sql(
        "solicitantes_clean", engine, if_exists="append", index=False, chunksize=1000
    )

    with engine.connect() as conn:
        ids = pd.read_sql("SELECT id FROM solicitantes_clean ORDER BY id", conn)

    df_pre = df[COLS_PRESTAMO].copy()
    df_pre.insert(0, "solicitante_id", ids["id"].values)
    df_pre.to_sql("prestamos_clean", engine, if_exists="append", index=False, chunksize=1000)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_sol_csv = df[COLS_SOLICITANTE].copy()
    df_sol_csv.insert(0, "id", ids["id"].values)
    df_sol_csv.to_csv(CSV_OUT_SOL, index=False)
    df_pre.to_csv(CSV_OUT_PRE, index=False)

    logging.info(f"solicitantes_clean: {len(df)} filas | CSV -> {CSV_OUT_SOL.name}")
    logging.info(f"prestamos_clean:    {len(df)} filas | CSV -> {CSV_OUT_PRE.name}")


# ---------------------------------------------------------
# Controlador
# ---------------------------------------------------------
def main() -> None:
    logging.info("=== INICIO PIPELINE LIMPIEZA ===")
    engine = get_engine()

    df = extraer_datos_raw(engine)
    if df.empty:
        logging.error("Tablas raw vacias. Corre la ingesta primero.")
        sys.exit(1)

    df = remover_duplicados(df)
    df = imputar_nulos(df)
    df = aplicar_reglas_rango(df)        # primero los rangos duros
    df = aplicar_reglas_cruzadas(df)     # despues, consistencia entre columnas
    df = aplicar_dominios_categoricos(df)
    df = aplicar_winsorizacion(df)       # ultimo: atipicos sobre columnas sin rango duro

    persistir_y_exportar(df, engine)
    logging.info("=== LIMPIEZA OK ===")


if __name__ == "__main__":
    main()
