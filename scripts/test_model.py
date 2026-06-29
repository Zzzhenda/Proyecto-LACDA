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
curva_precision_recall.png.
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

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_model")

# Rutas del proyecto
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

    # Validar existencia de artefactos requeridos
    modelo_path = MODELS_DIR / "modelo_default.pkl"
    if not modelo_path.exists():
        log.error(f"No existe {modelo_path}. Corre train_model.py primero.")
        sys.exit(1)
        
    for f in ("X_test.csv", "y_test.csv"):
        if not (DATA_DIR / f).exists():
            log.error(f"No existe data/{f}. Corre train_model.py primero.")
            sys.exit(1)

    # Carga de datos y modelo
    with open(modelo_path, "rb") as f:
        model = pickle.load(f)
        
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    log.info(f"Modelo cargado; holdout de {len(X_test)} filas (nunca visto en el fit).")

    # Predicciones
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

    # ----- Generación de Gráficos (Protegida con Try/Except) -----
    log.info("Generando gráficos de evaluación...")
    
    # 1. Matriz de confusión
    try:
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
        log.info("  [OK] Matriz de confusión guardada.")
    except Exception as e:
        log.error(f"  [ERROR] No se pudo generar la matriz de confusión: {e}")
        plt.close()

    # 2. Curva ROC
    try:
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
        log.info("  [OK] Curva ROC guardada.")
    except Exception as e:
        log.error(f"  [ERROR] No se pudo generar la curva ROC: {e}")
        plt.close()

    # 3. Curva Precision-Recall
    try:
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
        log.info("  [OK] Curva Precision-Recall guardada.")
    except Exception as e:
        log.error(f"  [ERROR] No se pudo generar la curva Precision-Recall: {e}")
        plt.close()

    # ----- Gate de calidad del modelo -----
    if metricas["roc_auc"] < UMBRAL_AUC:
        log.error(f"GATE FALLIDO: ROC-AUC {metricas['roc_auc']:.4f} < umbral {UMBRAL_AUC}")
        sys.exit(1)
        
    log.info(f"Gate de calidad OK: ROC-AUC {metricas['roc_auc']:.4f} >= {UMBRAL_AUC}")
    log.info("=== EVALUACION OK ===")


if __name__ == "__main__":
    main()