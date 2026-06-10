"""Evaluacion del modelo de default sobre el holdout (gate de calidad ML).

Adaptacion de docs/test_model.py (material del curso) al proyecto LACDA.
Cambios respecto del original y su justificacion:

  * Evalua sobre data/X_test.csv / y_test.csv guardados por train_model.py
    (en el original esos archivos nunca se generaban — bug corregido).
  * Se elimino la matriz de correlacion del script original: es analisis
    exploratorio y vive en notebooks/estudio_features.ipynb, no en el test.
  * Gate de calidad del modelo (mismo patron que validacion.py con los
    datos): si ROC-AUC < UMBRAL_AUC el script sale con exit 1 y el CI se
    marca en rojo. Un modelo degradado no pasa desapercibido.
  * Importancia de variables re-agregada por variable de origen (el
    OneHotEncoder expande las categoricas y el original comparaba
    categorias sueltas contra variables numericas completas).

Salidas en results/: metricas.json, matriz_confusion.png, curva_roc.png,
curva_precision_recall.png, importancia_variables.png/.csv,
distribucion_probabilidades.png.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_model")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

# Gate: en CV el modelo rindio AUC ~0.973; bajo 0.93 algo se rompio
# (datos degradados, fuga en el split, bug de features).
UMBRAL_AUC = 0.93


def main() -> None:
    log.info("=== INICIO EVALUACION DEL MODELO ===")
    RESULTS_DIR.mkdir(exist_ok=True)

    modelo_path = MODELS_DIR / "modelo_default.pkl"
    if not modelo_path.exists():
        log.error(f"No existe {modelo_path}. Corre train_model.py primero.")
        sys.exit(1)
    for f in ("X_test.csv", "y_test.csv"):
        if not (DATA_DIR / f).exists():
            log.error(f"No existe data/{f}. Corre train_model.py primero.")
            sys.exit(1)

    with open(modelo_path, "rb") as f:
        model = pickle.load(f)
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    log.info(f"Modelo cargado; holdout de {len(X_test)} filas (nunca visto en el fit).")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ----- Metricas -----
    metricas = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    for k, v in metricas.items():
        log.info(f"  {k:10s} = {v:.4f}")

    metricas["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )
    with open(RESULTS_DIR / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)

    # ----- Matriz de confusion -----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pagado", "default"], yticklabels=["pagado", "default"])
    plt.title("Matriz de Confusión (holdout)", fontweight="bold")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "matriz_confusion.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----- Curva ROC -----
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, label=f"AUC = {metricas['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC", fontweight="bold")
    plt.legend()
    plt.savefig(RESULTS_DIR / "curva_roc.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----- Curva Precision-Recall -----
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_c, prec_c)
    plt.figure(figsize=(7, 5.5))
    plt.plot(rec_c, prec_c, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall", fontweight="bold")
    plt.legend()
    plt.savefig(RESULTS_DIR / "curva_precision_recall.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----- Importancia por variable de origen -----
    rf = model.named_steps["classifier"]
    nombres = model.named_steps["preprocessor"].get_feature_names_out()
    features = list(X_test.columns)
    candidatas = sorted(features, key=len, reverse=True)

    def col_base(n: str) -> str:
        n = n.split("__", 1)[1]
        return next(c for c in candidatas if n == c or n.startswith(c + "_"))

    imp = (pd.Series(rf.feature_importances_, index=[col_base(n) for n in nombres])
           .groupby(level=0).sum().sort_values(ascending=False))
    imp.to_frame("importancia").to_csv(RESULTS_DIR / "importancia_variables.csv")
    log.info(f"Top 5 variables: {', '.join(imp.head(5).index)}")

    plt.figure(figsize=(8, 5.5))
    sns.barplot(x=imp.values, y=imp.index, orient="h")
    plt.title("Importancia agregada por variable", fontweight="bold")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "importancia_variables.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----- Distribucion de probabilidades -----
    plt.figure(figsize=(7, 4.5))
    plt.hist(y_prob, bins=30)
    plt.title("Distribución de probabilidades predichas", fontweight="bold")
    plt.xlabel("Probabilidad de default")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribucion_probabilidades.png", dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"Resultados guardados en {RESULTS_DIR}/")

    # ----- Gate de calidad del modelo -----
    if metricas["roc_auc"] < UMBRAL_AUC:
        log.error(f"GATE FALLIDO: ROC-AUC {metricas['roc_auc']:.4f} < umbral {UMBRAL_AUC}")
        sys.exit(1)
    log.info(f"Gate de calidad OK: ROC-AUC {metricas['roc_auc']:.4f} >= {UMBRAL_AUC}")
    log.info("=== EVALUACION OK ===")


if __name__ == "__main__":
    main()
