import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

# Crear carpeta de resultados
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------
# Cargar datos
# -------------------------------------------------
data = pd.read_csv("data/data_fraude.csv")
target = "is_fraud"


# -------------------------------------------------
# Matriz de correlación
# -------------------------------------------------
plt.figure(figsize=(12, 8))

corr = data.corr(numeric_only=True)

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)

plt.title("Matriz de Correlación", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "matriz_correlacion.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# -------------------------------------------------
# Cargar modelo
# -------------------------------------------------
with open("models/modelo_fraude.pkl", "rb") as f:
    model = pickle.load(f)
    print("Modelo descargado exitosamente")

# -------------------------------------------------
# Predicciones
# -------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------
# Métricas
# -------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MÉTRICAS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n========== REPORTE ==========")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# Matriz de confusión visual
# -------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legítima", "Fraude"],  # Predicciones
    yticklabels=["Legítima", "Fraude"]   # Valores reales    
)

plt.title("Matriz de Confusión", fontsize=14, fontweight="bold")
plt.xlabel("Predicción", fontsize=12, fontweight="bold")
plt.ylabel("Valor Real", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "matriz_confusion.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -------------------------------------------------
# Curva ROC
# -------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
plt.title("Curva ROC", fontsize=14, fontweight="bold")
plt.legend()

plt.savefig(
    os.path.join(RESULTS_DIR, "curva_roc.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -------------------------------------------------
# Curva Precision Recall
# -------------------------------------------------
precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_prob
)

pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel("Recall", fontsize=12, fontweight="bold")
plt.ylabel("Precision", fontsize=12, fontweight="bold")
plt.title("Curva Precision-Recall", fontsize=14, fontweight="bold")
plt.legend()

plt.savefig(
    os.path.join(RESULTS_DIR, "curva_precision_recall.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -------------------------------------------------
# Importancia de variables
# -------------------------------------------------
rf_model = model.named_steps["classifier"]

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = rf_model.feature_importances_

importance_data = pd.DataFrame({
    "Variable": feature_names,
    "Importancia": importances
})

importance_data = importance_data.sort_values(
    by="Importancia",
    ascending=False
)

print("\n========== TOP K VARIABLES ==========")
k = 5
print(importance_data.head(k))

plt.figure(figsize=(10, 8))

sns.barplot(
    data=importance_data.head(20),
    x="Importancia",
    y="Variable"
)

plt.title(f"{k} Variables más Importantes", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "importancia_variables.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -------------------------------------------------
# Guarda importancia de variables en un CSV
# -------------------------------------------------
importance_data.to_csv(
    os.path.join(
        RESULTS_DIR,
        "importancia_variables.csv"
    ),
    index=False
)

# -------------------------------------------------
# Distribución de probabilidades
# -------------------------------------------------
plt.figure(figsize=(8, 5))

plt.hist(
    y_prob,
    bins=30
)

plt.title("Distribución de Probabilidades Predichas", fontsize=14, fontweight="bold")
plt.xlabel("Probabilidad de Fraude", fontsize=12, fontweight="bold")
plt.ylabel("Frecuencia", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "distribucion_probabilidades.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Guarda las métricas en archivo JSON
metricas = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "classification_report": classification_report(
        y_test,
        y_pred,
        output_dict=True
    )
}

with open(
    os.path.join(RESULTS_DIR, "metricas.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(metricas, f, indent=4, ensure_ascii=False)    