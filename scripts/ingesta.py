"""Etapa 1 — Ingesta.

Define el CONTRATO DE DATOS del pipeline y lo exige en la puerta de
entrada: lee el CSV fuente (data/loan_data.csv), verifica su estructura
y deja un snapshot en la zona de staging (data/loan_data_raw.csv) para
que el resto del pipeline trabaje desacoplado de la fuente.

El contrato (columnas, rangos, dominios; basado en data/Metadata.txt)
vive aqui y las demas etapas lo importan:

  * limpieza.py    -> APLICA rangos y dominios (clip / imputacion).
  * validacion.py  -> AUDITA el dataset final contra las mismas reglas.

Una sola fuente de verdad: un cambio de regla se hace en un solo lugar
y limpieza/validacion no pueden desincronizarse.

Controles de entrada (fallan con exit 1; las etapas siguientes no corren):
  * el archivo fuente existe y no esta vacio,
  * estan exactamente las 14 columnas esperadas (COLUMNAS).

En un escenario productivo esta etapa extraeria desde la fuente real
(API, bucket, base transaccional); aqui la fuente es el CSV versionado
en el repositorio.
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
log = logging.getLogger("ingesta")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_FUENTE = DATA_DIR / "loan_data.csv"
CSV_RAW = DATA_DIR / "loan_data_raw.csv"

# ---------------------------------------------------------------------
# CONTRATO DE DATOS (fuente unica de verdad para todo el pipeline)
# ---------------------------------------------------------------------

# Las 14 columnas del CSV fuente, en su orden canonico.
COLUMNAS = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file", "loan_status",
]

# Rangos duros [min, max] por columna (ver Metadata.txt).
RANGOS = {
    "person_age": (18, 100),
    "credit_score": (300, 850),       # rango FICO
    "loan_int_rate": (5, 30),         # tasa anual en %
    "loan_percent_income": (0, 1),    # razon prestamo/ingreso
}

# Dominios categoricos cerrados.
DOMINIOS = {
    "person_gender": {"male", "female"},
    "person_education": {"High School", "Bachelor", "Master", "Associate", "Doctorate"},
    "person_home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "loan_intent": {
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "DEBTCONSOLIDATION", "HOMEIMPROVEMENT",
    },
    "previous_loan_defaults_on_file": {"Yes", "No"},
}


def main() -> None:
    log.info("=== INICIO INGESTA ===")

    if not CSV_FUENTE.exists():
        log.error(f"No se encontro la fuente {CSV_FUENTE}")
        sys.exit(1)

    df = pd.read_csv(CSV_FUENTE)
    log.info(f"Fuente leida: {df.shape[0]} filas, {df.shape[1]} columnas")

    if df.empty:
        log.error("La fuente no contiene filas.")
        sys.exit(1)

    faltantes = set(COLUMNAS) - set(df.columns)
    extra = set(df.columns) - set(COLUMNAS)
    if faltantes or extra:
        if faltantes:
            log.error(f"Columnas faltantes en la fuente: {sorted(faltantes)}")
        if extra:
            log.error(f"Columnas inesperadas en la fuente: {sorted(extra)}")
        sys.exit(1)
    log.info("Estructura OK: las 14 columnas del contrato estan presentes.")

    # Snapshot de staging con orden de columnas canonico.
    df = df[COLUMNAS]
    df.to_csv(CSV_RAW, index=False)
    log.info(f"Staging escrito: {CSV_RAW.name} ({len(df)} filas)")
    log.info("=== INGESTA OK ===")


if __name__ == "__main__":
    main()
