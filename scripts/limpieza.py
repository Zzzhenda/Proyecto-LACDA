"""Etapa 3 — Limpieza.

Lee el staging crudo (data/loan_data_raw.csv), APLICA el contrato de
datos definido en ingesta.py (RANGOS, DOMINIOS) y escribe
data/loan_data_clean.csv.

Estrategia (el ORDEN importa):

  1. Eliminacion de duplicados exactos (unica operacion que reduce filas).
  2. Imputacion de nulos (mediana / moda) para no perder filas por celdas
     vacias aisladas.
  3. Rangos duros por columna (RANGOS) -> clip.
  4. Reglas cruzadas (emp_exp <= age-18, cred_hist <= age) -> clip usando
     el valor FINAL de person_age; si se aplicaran antes del paso 3 la
     consistencia podria romperse al ajustar person_age despues.
  5. Dominios categoricos cerrados (DOMINIOS) -> valor invalido se
     reemplaza por la moda.
  6. Winsorizacion al 5% SOLO sobre columnas sin rango duro
     (person_income, loan_amnt): acota atipicos extremos sin destruir
     valores legitimos de las columnas con contrato estricto.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ingesta import DOMINIOS, RANGOS

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("limpieza")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_RAW = DATA_DIR / "loan_data_raw.csv"
CSV_CLEAN = DATA_DIR / "loan_data_clean.csv"

# Columnas a winsorizar al 5%. Solo las que NO tienen rango duro en el
# contrato: winsorizar una columna con contrato estricto destruiria
# valores legitimos dentro del rango valido.
WINSORIZE_COLS = ["person_income", "loan_amnt"]


def remover_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    # Define la función que recibe un DataFrame de pandas y devuelve otro DataFrame ya limpio.

    """Identifica y elimina las filas duplicadas exactas en memoria. 
    """ 

    # Calcula el número total de filas que son copias exactas sumando los valores booleanos (True) de df.duplicated().
    # El método df.duplicated() devuelve una Serie booleana donde True indica que la fila es un duplicado de una fila anterior.
    cantidad_duplicados = int(df.duplicated().sum())

    if cantidad_duplicados > 0:
        # Registra un mensaje informativo indicando la cantidad exacta de filas duplicadas detectadas.
        log.info(f"Se identificaron {cantidad_duplicados} filas duplicadas exactas.")
        # Elimina las filas repetidas (dejando la primera) y reestructura el índice desde 0 hasta N-1 sin conservar el viejo.
        # El método drop_duplicates() elimina las filas duplicadas, y reset_index(drop=True) restablece el índice del DataFrame resultante sin agregar el índice anterior como una columna.
        df = df.drop_duplicates().reset_index(drop=True)
        log.info(f"Duplicados removidos correctamente. Filas restantes: {len(df)}")

    else:
        log.info("No se identificaron filas duplicadas. Nada que remover.")
    return df


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Mediana para numericas, moda para categoricas. Evita perder filas."""
    df = df.copy()
    total = int(df.isnull().sum().sum())
    if total == 0:
        log.info("Sin nulos que imputar.")
        return df

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    log.info(f"Nulos imputados: {total}")
    return df


def aplicar_rangos(df: pd.DataFrame) -> pd.DataFrame:
    """Rangos duros del contrato sobre variables individuales (clip)."""
    df = df.copy()
    for col, (lo, hi) in RANGOS.items():
        fuera = int((~df[col].between(lo, hi)).sum())
        df[col] = df[col].clip(lo, hi)
        if fuera:
            log.info(f"  {col}: {fuera} valores fuera de [{lo}, {hi}] -> clip")

    df["person_income"] = df["person_income"].clip(lower=0)

    # loan_amnt: la regla es x > 0; los <= 0 se imputan a la mediana del resto
    invalidos = int((df["loan_amnt"] <= 0).sum())
    if invalidos:
        mediana = df.loc[df["loan_amnt"] > 0, "loan_amnt"].median()
        df.loc[df["loan_amnt"] <= 0, "loan_amnt"] = mediana
        log.info(f"  loan_amnt: {invalidos} valores <= 0 -> mediana ({mediana})")

    log.info("Rangos duros aplicados.")
    return df


def aplicar_reglas_cruzadas(df: pd.DataFrame) -> pd.DataFrame:
    """Consistencia entre columnas usando el person_age YA fijado."""
    df = df.copy()
    df["person_emp_exp"] = df["person_emp_exp"].clip(lower=0, upper=df["person_age"] - 18)
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].clip(
        lower=0, upper=df["person_age"]
    )
    log.info("Reglas cruzadas (emp_exp, cred_hist_length) aplicadas.")
    return df


def aplicar_dominios(df: pd.DataFrame) -> pd.DataFrame:
    """Valores fuera del dominio cerrado se reemplazan por la moda."""
    df = df.copy()
    for col, dominio in DOMINIOS.items():
        invalidos = int((~df[col].isin(dominio)).sum())
        if invalidos:
            moda = df.loc[df[col].isin(dominio), col].mode().iloc[0]
            df.loc[~df[col].isin(dominio), col] = moda
            log.info(f"  {col}: {invalidos} valores fuera de dominio -> {moda!r}")

    # loan_status: binario {0, 1}
    invalidos = int((~df["loan_status"].isin([0, 1])).sum())
    if invalidos:
        df.loc[~df["loan_status"].isin([0, 1]), "loan_status"] = 0
        log.info(f"  loan_status: {invalidos} valores fuera de {{0,1}} -> 0")

    log.info("Dominios categoricos aplicados.")
    return df


def winsorizar(df: pd.DataFrame) -> pd.DataFrame:
    """Acota cada columna de WINSORIZE_COLS a sus percentiles [5, 95]."""
    df = df.copy()
    for col in WINSORIZE_COLS:
        lo, hi = df[col].quantile(0.05), df[col].quantile(0.95)
        df[col] = np.clip(df[col], lo, hi)
    log.info(f"Winsorizacion 5% aplicada sobre: {WINSORIZE_COLS}")
    return df


def main() -> None:
    log.info("=== INICIO LIMPIEZA ===")

    if not CSV_RAW.exists():
        log.error(f"No existe {CSV_RAW}. Corre la ingesta primero.")
        sys.exit(1)

    df = pd.read_csv(CSV_RAW)
    log.info(f"Filas leidas del staging: {len(df)}")

    df = remover_duplicados(df)
    df = imputar_nulos(df)
    df = aplicar_rangos(df)            # primero los rangos duros
    df = aplicar_reglas_cruzadas(df)   # despues, consistencia entre columnas
    df = aplicar_dominios(df)
    df = winsorizar(df)                # ultimo: atipicos en columnas sin rango duro

    df.to_csv(CSV_CLEAN, index=False)
    log.info(f"Dataset limpio escrito: {CSV_CLEAN.name} ({len(df)} filas)")
    log.info("=== LIMPIEZA OK ===")


if __name__ == "__main__":
    main()
