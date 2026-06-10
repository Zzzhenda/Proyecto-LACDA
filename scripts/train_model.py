"""Entrenamiento del modelo de prediccion de default (Random Forest).

Adaptacion de docs/train_model.py (material del curso) al proyecto LACDA.
Cambios respecto del original y su justificacion:

  * Lee desde la tabla `loan_data` de PostgreSQL (el destino del pipeline
    DataOps); si la BD no esta disponible, cae al CSV transformado —
    integracion capa de datos -> capa ML con tolerancia a fallos.
  * Usa el set de 12 predictores "S3 depurado" seleccionado con evidencia
    en notebooks/estudio_features.ipynb (CV AUC 0.9734): excluye
    rate_x_pct_income y loan_burden (|corr| > 0.9 con sus originales,
    redundantes para arboles) y person_emp_exp (|corr| 0.954 con
    person_age); usa has_prev_defaults en lugar del categorico duplicado.
  * Corrige el bug del original: guarda X_test / y_test en data/ para que
    test_model.py evalue EXACTAMENTE el mismo holdout (en el material del
    curso, test_model.py leia archivos que nunca se guardaban).
  * SMOTE descartado por ahora: con class_weight="balanced" el recall en
    CV ya es ~0.84; queda como experimento si el test final lo refuta.

Salidas:
  * models/modelo_default.pkl       (pipeline OHE + RF entrenado)
  * data/X_test.csv, data/y_test.csv (holdout 20%, nunca visto en el fit)
  * results/distribucion_clases.png
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from validacion import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_model")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

TARGET = "loan_status"

# Set "S3 depurado" — seleccionado por CV en notebooks/estudio_features.ipynb
FEATURES = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_home_ownership", "loan_amnt", "loan_intent", "loan_int_rate",
    "loan_percent_income", "cb_person_cred_hist_length", "credit_score",
    "has_prev_defaults",
]


def cargar_datos() -> pd.DataFrame:
    """Tabla loan_data (Postgres) con fallback al CSV transformado."""
    try:
        df = pd.read_sql("SELECT * FROM loan_data", get_engine())
        log.info(f"Datos leidos desde PostgreSQL (loan_data): {len(df)} filas")
    except Exception as exc:
        log.warning(f"BD no disponible ({type(exc).__name__}); usando CSV transformado.")
        df = pd.read_csv(DATA_DIR / "loan_data_transformed.csv")
        log.info(f"Datos leidos desde CSV: {len(df)} filas")
    return df


def main() -> None:
    log.info("=== INICIO ENTRENAMIENTO ===")
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = cargar_datos()
    faltantes = set(FEATURES + [TARGET]) - set(df.columns)
    if faltantes:
        log.error(f"Faltan columnas en los datos: {sorted(faltantes)}")
        sys.exit(1)

    X = df[FEATURES]
    y = df[TARGET]

    # Distribucion de la variable objetivo (desbalance 78/22)
    ax = y.value_counts().rename({0: "pagado", 1: "default"}).plot.pie(
        autopct="%1.1f%%", figsize=(5, 5), ylabel="",
        title="Distribución de la variable objetivo",
    )
    ax.figure.savefig(RESULTS_DIR / "distribucion_clases.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    log.info(f"Distribucion de clases: {y.value_counts().to_dict()} "
             f"({y.mean():.1%} default) -> class_weight='balanced'")

    # Mismo split del estudio de features: el holdout queda fuera del fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=29, stratify=y
    )
    log.info(f"Split estratificado: train={len(X_train)} test={len(X_test)} (random_state=29)")

    categoricas = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(X[c])]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)],
        remainder="passthrough",
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=29, n_jobs=-1, class_weight="balanced",
        )),
    ])

    pipeline.fit(X_train, y_train)
    log.info("Modelo entrenado (RandomForest 200 arboles, class_weight=balanced).")

    # Guardar modelo y holdout (bug del material original: nunca los guardaba)
    with open(MODELS_DIR / "modelo_default.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

    log.info(f"Modelo guardado en {MODELS_DIR / 'modelo_default.pkl'}")
    log.info(f"Holdout guardado: data/X_test.csv ({len(X_test)} filas), data/y_test.csv")
    log.info("=== ENTRENAMIENTO OK ===")


if __name__ == "__main__":
    main()
