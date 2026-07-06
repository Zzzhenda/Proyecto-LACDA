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
  * Agregada la curva de aprendizaje para diagnosticar rendimiento vs. volumen.

Salidas:
  * models/modelo_default.pkl       (pipeline OHE + RF entrenado)
  * data/X_test.csv, data/y_test.csv (holdout 20%, nunca visto en el fit)
  * results/distribucion_clases.png
  * results/curva_aprendizaje.png, results/curva_aprendizaje.csv
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from validacion import get_engine

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_model")

# Rutas del proyecto
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
        log.info(f"Datos leídos desde PostgreSQL (loan_data): {len(df)} filas")
    except Exception as exc:
        log.warning(f"BD no disponible ({type(exc).__name__}); usando CSV transformado.")
        df = pd.read_csv(DATA_DIR / "loan_data_transformed.csv")
        log.info(f"Datos leídos desde CSV: {len(df)} filas")
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

    # ----- Gráfico: Distribución de Clases (Protegido) -----
    try:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        y.value_counts().rename({0: "pagado", 1: "default"}).plot.pie(
            autopct="%1.1f%%", ax=ax_pie, ylabel="",
            title="Distribución de la variable objetivo",
        )
        fig_pie.savefig(RESULTS_DIR / "distribucion_clases.png", dpi=300, bbox_inches="tight")
        plt.close(fig_pie)
        log.info("  [OK] Gráfico de distribución de clases guardado.")
    except Exception as e:
        log.error(f"  [ERROR] No se pudo generar el gráfico de distribución de clases: {e}")
        plt.close('all')

    log.info(f"Distribución de clases: {y.value_counts().to_dict()} "
             f"({y.mean():.1%} default) -> class_weight='balanced'")

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=29, stratify=y
    )
    log.info(f"Split estratificado: train={len(X_train)} test={len(X_test)} (random_state=29)")

    # Preprocesamiento y Pipeline
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
    log.info("Modelo entrenado (RandomForest 200 árboles, class_weight=balanced).")

    # ----- Cálculo de la Curva de Aprendizaje -----
    log.info("Generando curva de aprendizaje (rendimiento vs. volumen de datos)...")
    fracciones = [0.1, 0.25, 0.5, 0.75, 1.0]
    filas = []
    
    for fr in fracciones:
        if fr < 1.0:
            Xi, _, yi, _ = train_test_split(
                X_train, y_train, train_size=fr, random_state=29, stratify=y_train
            )
        else:
            Xi, yi = X_train, y_train
            
        m = clone(pipeline)
        m.fit(Xi, yi)
        p = m.predict(X_test)
        pr = m.predict_proba(X_test)[:, 1]
        
        filas.append({
            "n_entrenamiento": len(Xi),
            "accuracy": accuracy_score(y_test, p),
            "roc_auc": roc_auc_score(y_test, pr),
            "f1_score": f1_score(y_test, p),
        })

    curva = pd.DataFrame(filas)
    curva.to_csv(RESULTS_DIR / "curva_aprendizaje.csv", index=False)
    print("\n" + curva.round(4).to_string(index=False) + "\n")

    # ----- Gráfico: Curva de Aprendizaje (Protegido) -----
    try:
        AZUL = "#1f65b6"
        VERDE = "#2a9d8f"

        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.plot(curva["n_entrenamiento"], curva["accuracy"], marker="o", color=AZUL, linewidth=2, label="Accuracy")
        ax.plot(curva["n_entrenamiento"], curva["roc_auc"], marker="s", color=VERDE, linewidth=2, label="ROC-AUC")
        
        ax.set_xlabel("Tamaño del conjunto de entrenamiento (registros)", fontsize=10, color="#333333")
        ax.set_ylabel("Métrica", fontsize=10, color="#333333")
        ax.set_title("Curva de aprendizaje — rendimiento vs. volumen de datos", fontweight="bold", fontsize=12, pad=15)
        
        # Grid sutil horizontal y limpieza estricta de bordes (Flat Design)
        ax.grid(axis="y", linestyle="-", alpha=0.2, color="grey")
        ax.grid(axis="x", visible=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
        ax.legend(frameon=False, loc="lower right")

        # Anotaciones numéricas del valor ROC-AUC arriba de cada marcador verde
        for _, r in curva.iterrows():
            ax.annotate(f"{r['roc_auc']:.3f}", (r["n_entrenamiento"], r["roc_auc"]),
                        textcoords="offset points", xytext=(0, 8), ha="center", 
                        fontsize=8.5, color=VERDE, fontweight="semibold")

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "curva_aprendizaje.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("  [OK] Gráfico de curva de aprendizaje guardado.")
    except Exception as e:
        log.error(f"  [ERROR] No se pudo generar el gráfico de la curva de aprendizaje: {e}")
        plt.close('all')

    # Guardar modelo y holdout (Garantizado post-gráficos)
    with open(MODELS_DIR / "modelo_default.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

    log.info(f"Modelo guardado en {MODELS_DIR / 'modelo_default.pkl'}")
    log.info(f"Holdout guardado: data/X_test.csv ({len(X_test)} filas), data/y_test.csv")
    log.info("=== ENTRENAMIENTO OK ===")


if __name__ == "__main__":
    main()