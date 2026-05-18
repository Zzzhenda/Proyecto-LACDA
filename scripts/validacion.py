"""Validacion estructural y semantica (gate de salida del pipeline).

Verifica que las tablas _clean y _transformed cumplan todas las reglas
duras del cap. 9 del diseño tecnico. Es el contrato de calidad de
salida: si algo no pasa, el build falla y el CI se marca en rojo.

Para el sistema de monitoreo con KPIs y alertas (rubrica EP2 punto 2),
ver qualitycheck.py — corre antes en el pipeline sobre las tablas _raw.

Codigos de salida:
  0 - todas las reglas duras pasan
  1 - una regla dura del cap. 9 fallo (rompe el build / CI)
"""

import sys

import pandas as pd

from db import get_engine
from limpieza import CAT_DOMAINS


CAT_SOLICITANTE = {k: v for k, v in CAT_DOMAINS.items()
                   if k in ("person_gender", "person_education", "person_home_ownership")}

CAT_PRESTAMO = {k: v for k, v in CAT_DOMAINS.items()
                if k in ("loan_intent", "previous_loan_defaults_on_file")}


def auditar_solicitantes(df: pd.DataFrame) -> list[str]:
    fallos: list[str] = []

    if df.isnull().values.any():
        fallos.append(f"solicitantes: {int(df.isnull().sum().sum())} valores nulos")

    rangos = {"person_age": (18, 100), "credit_score": (300, 850)}
    for col, (lo, hi) in rangos.items():
        fuera = (~df[col].between(lo, hi)).sum()
        if fuera:
            fallos.append(f"solicitantes.{col}: {fuera} filas fuera de [{lo}, {hi}]")

    if (df["person_income"] < 0).any():
        fallos.append("solicitantes.person_income: tiene valores negativos")

    cap_emp = df["person_age"] - 18
    if (df["person_emp_exp"] > cap_emp).any() or (df["person_emp_exp"] < 0).any():
        fallos.append(
            "solicitantes.person_emp_exp: inconsistente (debe estar entre 0 y person_age-18)"
        )

    if (df["cb_person_cred_hist_length"] > df["person_age"]).any() or \
       (df["cb_person_cred_hist_length"] < 0).any():
        fallos.append("solicitantes.cb_person_cred_hist_length: inconsistente")

    for col, dominio in CAT_SOLICITANTE.items():
        invalidos = (~df[col].isin(dominio)).sum()
        if invalidos:
            fallos.append(f"solicitantes.{col}: {invalidos} valores fuera del dominio")

    return fallos


def auditar_prestamos(df: pd.DataFrame) -> list[str]:
    fallos: list[str] = []

    if df.isnull().values.any():
        fallos.append(f"prestamos: {int(df.isnull().sum().sum())} valores nulos")

    rangos = {"loan_int_rate": (5, 30), "loan_percent_income": (0, 1)}
    for col, (lo, hi) in rangos.items():
        fuera = (~df[col].between(lo, hi)).sum()
        if fuera:
            fallos.append(f"prestamos.{col}: {fuera} filas fuera de [{lo}, {hi}]")

    if (df["loan_amnt"] <= 0).any():
        fallos.append("prestamos.loan_amnt: tiene valores <= 0")

    for col, dominio in CAT_PRESTAMO.items():
        invalidos = (~df[col].isin(dominio)).sum()
        if invalidos:
            fallos.append(f"prestamos.{col}: {invalidos} valores fuera del dominio")

    if not df["loan_status"].isin([0, 1]).all():
        fallos.append("prestamos.loan_status: contiene valores distintos de 0/1")

    return fallos


def auditar_features_solicitante(df: pd.DataFrame) -> list[str]:
    # solicitantes_transformed no agrega features propias en esta entrega;
    # los predictores significativos de la entidad solicitante son
    # categoricos (previous_loan_defaults_on_file, home_ownership) y se
    # encodean en el pipeline de ML, no aqui.
    return auditar_solicitantes(df)


def auditar_features_prestamo(df: pd.DataFrame) -> list[str]:
    extra = ["rate_x_pct_income", "loan_burden", "has_prev_defaults"]
    fallos = auditar_prestamos(df.drop(columns=extra, errors="ignore"))
    if (df["rate_x_pct_income"] < 0).any():
        fallos.append("prestamos_transformed.rate_x_pct_income: tiene valores negativos")
    if (df["loan_burden"] < 0).any():
        fallos.append("prestamos_transformed.loan_burden: tiene valores negativos")
    if not df["has_prev_defaults"].isin([0, 1]).all():
        fallos.append("prestamos_transformed.has_prev_defaults: valores fuera de {0, 1}")
    return fallos


def _read(table: str, engine) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    if df.empty:
        sys.exit(f"[validacion] {table} esta vacia.")
    return df.drop(columns=[c for c in ("id", "solicitante_id", "fecha_carga") if c in df.columns])


def _check(nombre: str, fallos: list[str]) -> bool:
    if fallos:
        print(f"[validacion] FALLAS en {nombre}:")
        for f in fallos:
            print(f"  - {f}")
        return False
    print(f"[validacion] {nombre} OK")
    return True


def main() -> None:
    engine = get_engine()
    ok = True

    df_sol_c = _read("solicitantes_clean", engine)
    print(f"[validacion] Auditando {len(df_sol_c)} filas de solicitantes_clean")
    ok &= _check("solicitantes_clean", auditar_solicitantes(df_sol_c))

    df_pre_c = _read("prestamos_clean", engine)
    print(f"[validacion] Auditando {len(df_pre_c)} filas de prestamos_clean")
    ok &= _check("prestamos_clean", auditar_prestamos(df_pre_c))

    df_sol_t = _read("solicitantes_transformed", engine)
    print(f"[validacion] Auditando {len(df_sol_t)} filas de solicitantes_transformed")
    ok &= _check("solicitantes_transformed", auditar_features_solicitante(df_sol_t))

    df_pre_t = _read("prestamos_transformed", engine)
    print(f"[validacion] Auditando {len(df_pre_t)} filas de prestamos_transformed")
    ok &= _check("prestamos_transformed", auditar_features_prestamo(df_pre_t))

    if not ok:
        sys.exit(1)

    defaults = int(df_pre_t["loan_status"].sum())
    pagados = int((df_pre_t["loan_status"] == 0).sum())
    print()
    print(f"[validacion] Resumen: {len(df_pre_t)} prestamos | "
          f"{defaults} defaults / {pagados} pagados")


if __name__ == "__main__":
    main()
