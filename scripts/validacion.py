"""Etapa 5 — Validacion estructural y semantica + carga final.

Audita data/loan_data_transformed.csv contra el contrato de datos
(definido en ingesta.py) y, SOLO si todas las reglas pasan, carga el
dataset en la tabla unica `loan_data` de PostgreSQL (db/init.sql).

Es el gate de calidad del pipeline: si UNA regla falla, sale con exit 1,
la base de datos no se toca y el CI se marca en rojo. La carga es la
consecuencia directa de pasar el gate — por eso viven en el mismo paso.

Validacion estructural:
  * estan todas las columnas (14 del contrato + 3 features derivadas),
  * no hay valores nulos.

Validacion semantica:
  * rangos duros por columna (RANGOS),
  * person_income >= 0, loan_amnt > 0,
  * reglas cruzadas: emp_exp <= age - 18, cred_hist <= age,
  * dominios categoricos cerrados (DOMINIOS), loan_status en {0,1},
  * features derivadas: sin negativos, has_prev_defaults en {0,1}.

Carga (si la validacion paso):
  * TRUNCATE + INSERT dentro de una transaccion -> idempotente: re-correr
    no duplica datos, y un fallo a mitad de camino revierte todo.
  * Verificacion post-carga: el conteo en la tabla debe coincidir con el
    CSV; si no, exit 1.
  * `fecha_carga` (DEFAULT CURRENT_TIMESTAMP en el esquema) deja la
    trazabilidad de cada corrida.

A diferencia de qualitycheck.py (KPI informativo sobre el dato crudo),
aqui el dato YA paso por limpieza: cualquier violacion es un bug del
pipeline y debe romper el build.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ingesta import COLUMNAS, DOMINIOS, RANGOS
from transformacion import FEATURES_DERIVADAS

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validacion")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_TRANSFORMED = DATA_DIR / "loan_data_transformed.csv"
TABLA = "loan_data"


def get_engine() -> Engine:
    """Engine de SQLAlchemy a Postgres. Lee credenciales de variables de
    entorno con defaults consistentes con docker-compose.yml: funciona
    dentro del contenedor `app` (host=db) y localmente con DB_HOST=localhost."""
    user = os.getenv("DB_USER", "lacda")
    pwd = os.getenv("DB_PASSWORD", "lacda_pass")
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "loans")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url, connect_args={"client_encoding": "utf8"})


# ---------------------------------------------------------------------
# Validacion
# ---------------------------------------------------------------------

def validar_estructura(df: pd.DataFrame) -> list[str]:
    fallos = []
    esperadas = set(COLUMNAS) | set(FEATURES_DERIVADAS)
    faltantes = esperadas - set(df.columns)
    if faltantes:
        fallos.append(f"columnas faltantes: {sorted(faltantes)}")

    nulos = int(df.isnull().sum().sum())
    if nulos:
        fallos.append(f"{nulos} valores nulos")
    return fallos


def validar_semantica(df: pd.DataFrame) -> list[str]:
    fallos = []

    for col, (lo, hi) in RANGOS.items():
        fuera = int((~df[col].between(lo, hi)).sum())
        if fuera:
            fallos.append(f"{col}: {fuera} filas fuera de [{lo}, {hi}]")

    if (df["person_income"] < 0).any():
        fallos.append("person_income: tiene valores negativos")
    if (df["loan_amnt"] <= 0).any():
        fallos.append("loan_amnt: tiene valores <= 0")

    if ((df["person_emp_exp"] < 0) | (df["person_emp_exp"] > df["person_age"] - 18)).any():
        fallos.append("person_emp_exp: inconsistente (debe estar entre 0 y person_age-18)")
    if ((df["cb_person_cred_hist_length"] < 0)
            | (df["cb_person_cred_hist_length"] > df["person_age"])).any():
        fallos.append("cb_person_cred_hist_length: inconsistente (debe estar entre 0 y person_age)")

    for col, dominio in DOMINIOS.items():
        invalidos = int((~df[col].isin(dominio)).sum())
        if invalidos:
            fallos.append(f"{col}: {invalidos} valores fuera del dominio")

    if not df["loan_status"].isin([0, 1]).all():
        fallos.append("loan_status: contiene valores distintos de 0/1")

    return fallos


def validar_features(df: pd.DataFrame) -> list[str]:
    fallos = []
    if (df["rate_x_pct_income"] < 0).any():
        fallos.append("rate_x_pct_income: tiene valores negativos")
    if (df["loan_burden"] < 0).any():
        fallos.append("loan_burden: tiene valores negativos")
    if not df["has_prev_defaults"].isin([0, 1]).all():
        fallos.append("has_prev_defaults: valores fuera de {0, 1}")
    return fallos


# ---------------------------------------------------------------------
# Carga (solo se ejecuta si el gate paso)
# ---------------------------------------------------------------------

def cargar(df: pd.DataFrame) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLA} RESTART IDENTITY"))
        df.to_sql(TABLA, conn, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLA}")).scalar()

    if n != len(df):
        log.error(f"Conteo post-carga ({n}) no coincide con el CSV ({len(df)}).")
        sys.exit(1)

    log.info(f"Tabla {TABLA}: {n} filas cargadas y verificadas.")


def main() -> None:
    log.info("=== INICIO VALIDACION (gate de calidad) ===")

    if not CSV_TRANSFORMED.exists():
        log.error(f"No existe {CSV_TRANSFORMED}. Corre el pipeline desde la ingesta.")
        sys.exit(1)

    df = pd.read_csv(CSV_TRANSFORMED)
    log.info(f"Auditando {len(df)} filas de {CSV_TRANSFORMED.name}")

    fallos = validar_estructura(df)
    if not fallos:  # las checks semanticas asumen estructura completa
        fallos += validar_semantica(df)
        fallos += validar_features(df)

    if fallos:
        log.error(f"VALIDACION FALLIDA: {len(fallos)} regla(s) violada(s). La BD no se toca.")
        for f in fallos:
            log.error(f"  - {f}")
        sys.exit(1)

    defaults = int(df["loan_status"].sum())
    pagados = int((df["loan_status"] == 0).sum())
    log.info(f"Todas las reglas pasan. {len(df)} prestamos | "
             f"{defaults} defaults / {pagados} pagados")
    log.info("=== VALIDACION OK -> ejecutando carga ===")

    cargar(df)
    log.info("=== CARGA OK ===")


if __name__ == "__main__":
    main()
