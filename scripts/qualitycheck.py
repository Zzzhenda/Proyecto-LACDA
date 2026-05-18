"""KPI de calidad de datos (sistema de monitoreo, fase 2 del pipeline).

Mide la calidad de los datos en su estado crudo (_raw), justo despues
de la ingesta y antes de la limpieza. Materializa el sistema de
monitoreo con KPIs y alertas que pide la rubrica EP2.

A diferencia de validacion.py (gate de salida), aqui no se rompe el
build: el dataset crudo se espera que tenga problemas. El proposito
es:

  * Observabilidad: detectar si la fuente de datos se degrada en el
    tiempo (un score raw que cae entre corridas indica que el
    proveedor de datos cambio algo).
  * Diagnostico: saber con que estamos trabajando antes de tocarlo.

Score 0-100 ponderado por 4 dimensiones:
  * nulos / faltantes        (peso 0.30)
  * duplicados               (peso 0.20)
  * outliers (IQR 1.5x)      (peso 0.20)
  * inconsistencias          (peso 0.30)

Codigo de salida:
  0 - siempre (el KPI es informativo, no un gate)
"""

import sys
from typing import Iterable, Optional

import pandas as pd

from db import get_engine


PESOS = {
    "nulos/faltantes": 0.30,
    "duplicados": 0.20,
    "outliers": 0.20,
    "inconsistencias": 0.30,
}

UMBRAL_WARNING = 70.0
UMBRAL_CRITICAL = 50.0


class QualityCheck:
    """Diagnostico cuantitativo del estado de un DataFrame."""

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
        f"[qualitycheck] [{marca}] {nombre:20s}  "
        f"score={score:6.2f}/100  nivel={nivel:8s}  {flags}"
    )
    return nivel


def _read(table: str, engine) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    if df.empty:
        sys.exit(f"[qualitycheck] {table} esta vacia. Corre la ingesta primero.")
    return df.drop(columns=[c for c in ("id", "solicitante_id", "fecha_carga") if c in df.columns])


def main() -> None:
    engine = get_engine()

    print("[qualitycheck] === KPI baseline sobre datos crudos (raw) ===")
    df_sol = _read("solicitantes_raw", engine)
    df_pre = _read("prestamos_raw", engine)
    imprimir_reporte("solicitantes_raw", QualityCheck(df_sol))
    imprimir_reporte(
        "prestamos_raw",
        QualityCheck(df_pre, exclude_inconsistencies=["loan_status"]),
    )
    print("[qualitycheck] OK (informativo, no rompe build)")


if __name__ == "__main__":
    main()
