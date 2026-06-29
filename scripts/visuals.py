import json
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# 2. Análisis y manipulación de datos (Data Science Core)
import numpy as np
import pandas as pd

# 3. Visualización de datos y gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# 4. Machine Learning y Procesamiento (Scikit-Learn)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 5. Módulos locales personalizados (Solo validación)
from validacion import validar_estructura, validar_features, validar_semantica

# ==========================================
# CONFIGURACIÓN DE RUTAS (Adaptada para Docker)
# ==========================================
DATA = Path("data")
RESULTS = Path("results")
MODELS = Path("models")

# Apuntar directamente a la carpeta de resultados existente
output_dir = RESULTS

pd.set_option("display.max_columns", None)

# ==========================================
# CARGA DE DATOS YA TRANSFORMADOS
# ==========================================
df_t = pd.read_csv(DATA / "loan_data_transformed.csv")
y = df_t["loan_status"]

ORIGINALES = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file",
]
TODAS = [c for c in ORIGINALES if c != "previous_loan_defaults_on_file"] + [
    "rate_x_pct_income", "loan_burden", "has_prev_defaults",
]

# ==========================================
# 1. ESCALABILIDAD DE LA VALIDACIÓN DEL PIPELINE
# ==========================================
filas = []
for factor in (1, 2, 4, 8):
    df_n = pd.concat([df_t], ignore_index=True) if factor == 1 else pd.concat([df_t] * factor, ignore_index=True)
    
    t0 = time.perf_counter()
    fallos = (validar_estructura(df_n) + validar_semantica(df_n) + validar_features(df_n))
    t_valida = time.perf_counter() - t0

    filas.append({
        "filas": len(df_n), 
        "validacion (s)": t_valida
    })

bench = pd.DataFrame(filas).set_index("filas")

ax = bench.plot(marker="o", figsize=(7.5, 4.5),
                title="Escalabilidad de las operaciones de validación")
ax.set_ylabel("segundos")
ax.set_xlabel("filas procesadas")
ax.figure.tight_layout()

plt.savefig(output_dir / "escalabilidad_pipeline.png", dpi=300)
plt.close()

# ==========================================
# 2. CORRELACIÓN CON EL TARGET
# ==========================================
corr = (df_t.select_dtypes("number").corr()["loan_status"]
        .drop("loan_status").abs().sort_values(ascending=False))

ax = corr.plot.barh(figsize=(7, 4.5), title="|correlación| con loan_status (dataset final)")
ax.invert_yaxis()
ax.figure.tight_layout()

plt.savefig(output_dir / "correlacion_target.png", dpi=300)
plt.close()

# ==========================================
# 3. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# ==========================================
ax = y.value_counts().rename({0: "pagado", 1: "default"}).plot.bar(
    figsize=(5, 3.5), rot=0, title="Distribución de la variable objetivo")
ax.bar_label(ax.containers[0])
ax.figure.tight_layout()

plt.savefig(output_dir / "distribucion_target.png", dpi=300)
plt.close()

# ==========================================
# 4. MATRIZ DE CORRELACIÓN (HEATMAP)
# ==========================================
num = df_t[TODAS].select_dtypes("number")

plt.figure(figsize=(10, 7.5))
sns.heatmap(num.corr().round(2), cmap="coolwarm", annot=True, fmt=".2f",
            vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
plt.title("Matriz de correlación — variables numéricas", fontweight="bold")
plt.tight_layout()

plt.savefig(output_dir / "matriz_correlacion_train.png", dpi=300)
plt.close()

# ==========================================
# 5. IMPORTANCIA DE VARIABLES (RF)
# ==========================================
try:
    with open(MODELS / "modelo_default.pkl", "rb") as f:
        pipe = pickle.load(f)

    nombres = pipe.named_steps["preprocessor"].get_feature_names_out()
    imps = pipe.named_steps["classifier"].feature_importances_

    candidatas = sorted(TODAS, key=len, reverse=True)
    def col_base(n):
        n = n.split("__", 1)[1]
        return next(c for c in candidatas if n == c or n.startswith(c + "_"))

    imp_var = (pd.Series(imps, index=[col_base(n) for n in nombres])
               .groupby(level=0).sum().sort_values(ascending=False))

    ax = imp_var.plot.barh(figsize=(7, 5), title="Importancia agregada por variable (RF)")
    ax.invert_yaxis()
    ax.figure.tight_layout()

    plt.savefig(output_dir / "importancia_variables_rf.png", dpi=300)
    plt.close()
except FileNotFoundError:
    print("Aviso: No se encontró 'models/modelo_default.pkl', omitiendo gráfico de importancia.")

    # ==========================================
# 6. GRÁFICO: MÉTRICAS DE CALIDAD Y TIEMPO DE EJECUCIÓN (KPI)
# ==========================================
try:
    # 1. Cargar las métricas reales previamente guardadas
    with open(RESULTS / "metricas.json", "r", encoding="utf-8") as f:
        metricas = json.load(f)

    # 2. Cargar X_test para contar las solicitudes y medir inferencia en caliente
    X_test = pd.read_csv(DATA / "X_test.csv")
    
    # Intentar medir el tiempo de inferencia real usando el modelo guardado
    with open(MODELS / "modelo_default.pkl", "rb") as f:
        model = pickle.load(f)
    
    t0 = time.perf_counter()
    _ = model.predict(X_test)
    t_infer = time.perf_counter() - t0
    
    # 3. Configuración de estilos visuales nativos
    AZUL = "#1f77b4"
    GRIS = "#7f7f7f"
    
    nombres = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    claves = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    valores = [metricas[c] * 100 for c in claves]

    fig, (ax, axt) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [3, 1]})

    # Dibujar barras de métricas de calidad
    barras = ax.bar(nombres, valores, color=AZUL)
    ax.axhline(80, color=GRIS, lw=1, ls="--")
    ax.text(len(nombres) - 0.5, 81, "referencia 80%", color=GRIS, fontsize=8.5, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("Métricas de calidad del modelo")
    ax.grid(axis="x", visible=False)
    
    # Reemplazo nativo de etiquetar_barras
    ax.bar_label(barras, fmt="{:.1f}", fontsize=10, color="#333", label_type="edge", padding=3)

    # Tiempo de ejecución como KPI dedicado
    axt.axis("off")
    axt.add_patch(plt.Rectangle((0.08, 0.18), 0.84, 0.64, transform=axt.transAxes,
                                facecolor=GRIS, alpha=0.12, edgecolor=GRIS, lw=1.4))
    axt.text(0.5, 0.62, f"{t_infer*1000:.0f} ms", transform=axt.transAxes,
             ha="center", fontsize=22, fontweight="bold", color="#333")
    axt.text(0.5, 0.40, f"inferencia · {len(X_test):,} solicitudes", transform=axt.transAxes,
             ha="center", fontsize=9.5, color="#555")
    axt.set_title("Tiempo de ejecución")
    
    # Guardar directamente en la carpeta results asignada arriba
    plt.savefig(RESULTS / "metricas_calidad.png", dpi=300, bbox_inches="tight")
    plt.close()

except FileNotFoundError as e:
    print(f"Aviso: No se pudo generar el KPI de métricas debido a la falta de archivos: {e}")
except Exception as e:
    print(f"Aviso: Error inesperado al procesar el gráfico de calidad: {e}")