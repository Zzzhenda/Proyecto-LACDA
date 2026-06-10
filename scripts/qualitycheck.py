"""Etapa 2 — Monitoreo de calidad (KPIs con alertas).

Mide la calidad del dato crudo (data/loan_data_raw.csv) justo despues de
la ingesta y ANTES de la limpieza. Es el sistema de monitoreo del
pipeline: no rompe el build (el dato crudo se espera sucio), pero deja
un KPI trazable por corrida.

Proposito:
  * Observabilidad: si el score baja entre corridas, la fuente de datos
    se degrado (el proveedor cambio algo) y hay que investigar.
  * Diagnostico: saber con que estamos trabajando antes de tocarlo.

KPI: score 0-100 ponderado por 4 dimensiones de calidad. Cada dimension
penaliza proporcionalmente al % de filas afectadas:
  * nulos / faltantes        (peso 0.30)
  * duplicados               (peso 0.20)
  * outliers (IQR 1.5x)      (peso 0.20)
  * inconsistencias          (peso 0.30)

Alertas por umbral: OK (>= 70), WARNING (50-70), CRITICAL (< 50).

Codigo de salida: 0 siempre (es monitoreo informativo, no un gate;
el gate de salida del pipeline es validacion.py).
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
log = logging.getLogger("qualitycheck")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_RAW = DATA_DIR / "loan_data_raw.csv"

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

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def pct_nulos(self) -> float:
        return float(self.data.isnull().values.mean() * 100)

    def pct_duplicados(self) -> float:
        return float(self.data.duplicated().mean() * 100)

    def pct_outliers(self) -> float:
        """% de filas con al menos un outlier segun IQR 1.5x."""
        num = self.data.select_dtypes(include=["number"])
        q1, q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3 - q1
        mask = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).any(axis=1)
        return float(mask.mean() * 100)

    def pct_inconsistencias(self) -> float:
        """% de filas que violan reglas semanticas basicas del dominio."""
        df = self.data
        mask = (
            (df["person_income"] < 0)
            | (df["loan_amnt"] <= 0)
            | (df["person_emp_exp"] > df["person_age"] - 18)
            | (df["cb_person_cred_hist_length"] > df["person_age"])
        )
        return float(mask.mean() * 100)

    def quality_score(self) -> tuple[float, dict]:
        """Score 0-100: cada dimension penaliza su peso * % de filas afectadas."""
        dims = {
            "nulos/faltantes": self.pct_nulos(),
            "duplicados": self.pct_duplicados(),
            "outliers": self.pct_outliers(),
            "inconsistencias": self.pct_inconsistencias(),
        }
        penalizacion = sum(PESOS[k] * (v / 100) for k, v in dims.items())
        return round((1 - penalizacion) * 100, 2), dims


def nivel_alerta(score: float) -> str:
    if score < UMBRAL_CRITICAL:
        return "CRITICAL"
    if score < UMBRAL_WARNING:
        return "WARNING"
    return "OK"


def main() -> None:
    log.info("=== KPI DE CALIDAD SOBRE DATO CRUDO ===")

    if not CSV_RAW.exists():
        log.error(f"No existe {CSV_RAW}. Corre la ingesta primero.")
        sys.exit(1)

    df = pd.read_csv(CSV_RAW)
    score, dims = QualityCheck(df).quality_score()
    nivel = nivel_alerta(score)

    for dim, pct in dims.items():
        estado = "OK  " if pct == 0 else "WARN"
        log.info(f"  [{estado}] {dim:18s} {pct:6.2f}% de filas afectadas (peso {PESOS[dim]})")

    emitir = log.warning if nivel != "OK" else log.info
    emitir(f"QUALITY SCORE = {score}/100  ->  nivel {nivel}")
    log.info("=== QUALITYCHECK OK (informativo, no rompe build) ===")


if __name__ == "__main__":
    main()
