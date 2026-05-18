"""Validacion estructural, semantica y KPI de calidad (fase 3 del pipeline).

Fusiona el chequeo binario (reglas duras del cap. 9) con el KPI continuo
(QualityCheck adaptado de PROFESORA/quality_check.py). Cubre dos apartados
de la rubrica EP2:

  * Validacion estructural y semantica (etapa del pipeline).
  * Sistema de monitoreo con KPIs y alertas (calidad de datos).

Codigos de salida:
  0 - todo OK
  1 - una regla dura del cap. 9 fallo (rompe el build / CI)
  2 - quality_score < 50 en alguna tabla (alerta CRITICAL)
"""

import sys
from typing import Iterable, Optional

import pandas as pd

from db import get_engine
from limpieza import CAT_DOMAINS


# ============================================================
# A) Reglas duras del cap. 9 (validacion estructural + semantica)
# ============================================================

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
    fallos = auditar_solicitantes(df.drop(columns=["fico_band", "age_group"], errors="ignore"))
    if not df["fico_band"].isin([1, 2, 3, 4, 5]).all():
        fallos.append("solicitantes_transformed.fico_band: valores fuera de {1..5}")
    if not df["age_group"].isin([1, 2, 3]).all():
        fallos.append("solicitantes_transformed.age_group: valores fuera de {1..3}")
    return fallos


def auditar_features_prestamo(df: pd.DataFrame) -> list[str]:
    fallos = auditar_prestamos(df.drop(columns=["rate_x_pct_income"], errors="ignore"))
    if (df["rate_x_pct_income"] < 0).any():
        fallos.append("prestamos_transformed.rate_x_pct_income: tiene valores negativos")
    return fallos


# ============================================================
# B) KPI de calidad (QualityCheck adaptado del material docente)
# ============================================================

PESOS = {
    "nulos/faltantes": 0.30,
    "duplicados": 0.20,
    "outliers": 0.20,
    "inconsistencias": 0.30,
}

UMBRAL_WARNING = 70.0   # score < 70 -> WARNING (no rompe el build)
UMBRAL_CRITICAL = 50.0  # score < 50 -> CRITICAL (sys.exit 2)


class QualityCheck:
    """Diagnostico cuantitativo del estado de un DataFrame depurado."""

    def __init__(
        self,
        data: pd.DataFrame,
        exclude_inconsistencies: Optional[Iterable[str]] = None,
    ):
        self.data = data
        self.exclude_inconsistencies = list(exclude_inconsistencies or [])

    def has_nulls(self) -> bool:
        return bool(self.data.isnull().values.any())

    def has_duplicates(self) -> bool:
        return bool(self.data.duplicated().any())

    def has_outliers(self) -> bool:
        num = self.data.select_dtypes(include=["number"])
        for col in num.columns:
            q1 = num[col].quantile(0.25)
            q3 = num[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            if ((num[col] < lo) | (num[col] > hi)).any():
                return True
        return False

    def has_negative_values(self) -> bool:
        num = self.data.select_dtypes(include=["number"]).drop(
            columns=[c for c in self.exclude_inconsistencies if c in self.data.columns],
            errors="ignore",
        )
        for col in num.columns:
            if (num[col] < 0).any():
                return True
        return False

    def has_categorical_inconsistencies(self) -> bool:
        cat = self.data.select_dtypes(include=["object"])
        for col in cat.columns:
            valores = cat[col].dropna().astype(str)
            normalizados = valores.str.strip().str.lower()
            if valores.nunique() != normalizados.nunique():
                return True
        return False

    def has_inconsistencies(self) -> bool:
        return self.has_negative_values() or self.has_categorical_inconsistencies()

    def quality_score_weighted(self) -> float:
        checks = {
            "nulos/faltantes": self.has_nulls(),
            "duplicados": self.has_duplicates(),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies(),
        }
        penalizacion = sum(PESOS[k] for k, v in checks.items() if v)
        return round((1 - penalizacion) * 100, 2)

    def quality_report(self) -> dict:
        return {
            "nulos/faltantes": self.has_nulls(),
            "duplicados": self.has_duplicates(),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies(),
            "quality_score": self.quality_score_weighted(),
        }


def nivel_alerta(score: float) -> str:
    if score < UMBRAL_CRITICAL:
        return "CRITICAL"
    if score < UMBRAL_WARNING:
        return "WARNING"
    return "OK"


def imprimir_reporte(nombre: str, qc: QualityCheck) -> str:
    rep = qc.quality_report()
    score = rep["quality_score"]
    nivel = nivel_alerta(score)
    flags = "  ".join(
        f"{k}={'OK' if not rep[k] else 'WARN'}"
        for k in ("nulos/faltantes", "duplicados", "outliers", "inconsistencias")
    )
    marca = {"OK": " ", "WARNING": "!", "CRITICAL": "X"}[nivel]
    print(
        f"[validacion] [{marca}] {nombre:30s}  "
        f"score={score:6.2f}/100  nivel={nivel:8s}  {flags}"
    )
    return nivel


# ============================================================
# Controlador
# ============================================================

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

    print()
    print("[validacion] === KPI de calidad (QualityCheck) ===")
    niveles = [
        imprimir_reporte("solicitantes_clean", QualityCheck(df_sol_c)),
        imprimir_reporte("prestamos_clean",
                         QualityCheck(df_pre_c, exclude_inconsistencies=["loan_status"])),
        imprimir_reporte("solicitantes_transformed",
                         QualityCheck(df_sol_t, exclude_inconsistencies=["fico_band", "age_group"])),
        imprimir_reporte("prestamos_transformed",
                         QualityCheck(df_pre_t, exclude_inconsistencies=["loan_status"])),
    ]
    if any(n == "CRITICAL" for n in niveles):
        print("[validacion] ALERTA CRITICA: quality_score < 50 en alguna tabla.")
        sys.exit(2)

    defaults = int(df_pre_t["loan_status"].sum())
    pagados = int((df_pre_t["loan_status"] == 0).sum())
    print()
    print(f"[validacion] Resumen: {len(df_pre_t)} prestamos | "
          f"{defaults} defaults / {pagados} pagados")


if __name__ == "__main__":
    main()
