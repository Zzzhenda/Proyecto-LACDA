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

# ==========================================
# 7. GRÁFICO: INDICADORES CLAVE DEL MODELO (KPIS DASHBOARD) & PERSISTENCIA JSON
# ==========================================
try:
    # 1. Asegurar la carga de dependencias previas necesarias si no se han cargado antes
    with open(RESULTS / "metricas.json", "r", encoding="utf-8") as f:
        metricas = json.load(f)

    X_test = pd.read_csv(DATA / "X_test.csv")
    
    with open(MODELS / "modelo_default.pkl", "rb") as f:
        model = pickle.load(f)
    
    t0 = time.perf_counter()
    _ = model.predict(X_test)
    t_infer = time.perf_counter() - t0

    # 2. Definición de colores para la paleta de KPIs
    AZUL = "#1f77b4"
    VERDE = "#2ca02c"
    GRIS = "#7f7f7f"

    # Estructura de los indicadores
    kpis = [
        ("Accuracy",  f"{metricas['accuracy']*100:.1f}%",  AZUL),
        ("Recall (default)", f"{metricas['recall']*100:.1f}%", VERDE),
        ("F1 (default)", f"{metricas['f1_score']*100:.1f}%", AZUL),
        ("ROC-AUC",  f"{metricas['roc_auc']:.3f}", VERDE),
        ("Tiempo infer.", f"{t_infer*1000:.0f} ms", GRIS),
    ]

    # Crear la cuadrícula de KPIs en horizontal
    fig, axes = plt.subplots(1, len(kpis), figsize=(2.5 * len(kpis), 1.9))
    
    for ax, (etq, val, color) in zip(axes, kpis):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.05, 0.10), 0.90, 0.80, transform=ax.transAxes,
                                   facecolor=color, alpha=0.10, edgecolor=color, lw=1.4, zorder=0))
        ax.text(0.5, 0.60, val, transform=ax.transAxes, ha="center", va="center",
                fontsize=19, fontweight="bold", color=color)
        ax.text(0.5, 0.26, etq, transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#333")
                
    fig.suptitle("Indicadores clave del modelo (holdout de {:,} solicitudes)".format(len(X_test)),
                 fontsize=12, fontweight="bold", y=1.05)
    
    # Guardar gráfico de KPIs en la carpeta de resultados
    plt.savefig(RESULTS / "01_kpis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Gráfico '01_kpis.png' guardado exitosamente en results/")

    # 3. Persistir KPIs como resultado estructurado JSON
    resumen = {k: round(metricas[k], 4) for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']}
    resumen["tiempo_inferencia_ms"] = round(t_infer * 1000, 2)
    resumen["n_holdout"] = int(len(X_test))
    resumen["generado"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Guardar el JSON en la carpeta results asignada arriba
    (RESULTS / "resumen_kpis.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("KPIs guardados exitosamente en results/resumen_kpis.json")

except FileNotFoundError as e:
    print(f"Aviso: No se pudo generar el cuadro de mando de KPIs por archivos faltantes: {e}")
except Exception as e:
    print(f"Aviso: Error inesperado al procesar el cuadro de mando de KPIs: {e}")



# ==========================================
# 10. GRÁFICO: VOLUMEN DE DATOS POR ETAPA DEL PIPELINE
# ==========================================
try:
    # 1. Función auxiliar local para contar registros
    def contar_csv(ruta: Path):
        return len(pd.read_csv(ruta)) if ruta.exists() else None

    # Carga de variables necesarias para el conteo
    n_ingesta = contar_csv(DATA / "loan_data_raw.csv")
    n_limpieza = contar_csv(DATA / "loan_data_clean.csv")
    n_transf = contar_csv(DATA / "loan_data_transformed.csv")
    
    # Adaptación a las variables ya existentes en visuals.py
    n_bd = int(len(df_t))                    # filas finales transformadas
    n_test = int(len(X_test))                # holdout de evaluación
    n_train = n_transf - n_test if n_transf else None

    # Creación del DataFrame de volúmenes
    etapas = pd.DataFrame({
        "etapa": ["1 · Ingesta", "3 · Limpieza", "4 · Transformación",
                  "5 · Carga BD", "ML · Entrenamiento", "ML · Holdout"],
        "filas": [n_ingesta, n_limpieza, n_transf, n_bd, n_train, n_test],
    })
    
    # Guardar CSV de control en la carpeta results/
    etapas.to_csv(RESULTS / "volumen_pipeline.csv", index=False)

    # 2. Paleta de colores extendida
    AZUL = "#1f77b4"
    VERDE = "#2ca02c"
    GRIS = "#7f7f7f"
    NARANJA = "#ff7f0e"
    ROJO = "#d62728"

    colores = [AZUL, AZUL, AZUL, VERDE, GRIS, NARANJA]
    
    fig, ax = plt.subplots(figsize=(9, 4.2))
    barras = ax.barh(etapas["etapa"], etapas["filas"], color=colores)
    ax.invert_yaxis()                      # Flujo estructurado de arriba hacia abajo
    ax.set_xlabel("Registros")
    ax.set_title("Volumen de datos por etapa del pipeline")
    ax.margins(x=0.12)
    ax.grid(axis="y", visible=False)
    
    # Reemplazo nativo de etiquetar_barras para gráficos horizontales
    ax.bar_label(barras, fmt="{:,.0f}", fontsize=10, color="#333", padding=5)

    # Anotar registros descartados en la limpieza de datos
    descartados = (n_ingesta - n_limpieza) if (n_ingesta and n_limpieza) else 0
    tasa_retencion = (n_limpieza / n_ingesta) if (n_ingesta and n_ingesta > 0) else 0
    
    ax.annotate(f"Descartados en limpieza: {descartados:,}  (retención {tasa_retencion:.1%})",
                xy=(0.5, -0.22), xycoords="axes fraction", ha="center",
                fontsize=9.5, color=ROJO if descartados else GRIS)
    
    # Guardar gráfico definitivo en formato PNG
    plt.savefig(RESULTS / "02_volumen_etapas.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Gráfico '02_volumen_etapas.png' y CSV guardados con éxito en results/")
    print(etapas.to_string(index=False))

except FileNotFoundError as e:
    print(f"Aviso: No se pudo calcular el volumen por etapas debido a archivos faltantes: {e}")
except Exception as e:
    print(f"Aviso: Error inesperado al procesar el gráfico de volúmenes: {e}")

 # ==========================================
# 11. GRÁFICO: HISTÓRICO DE RENDIMIENTO POR CORRIDA (PERSISTENCIA ACUMULATIVA)
# ==========================================
try:
    # 1. Configuración del archivo histórico acumulativo en results/
    hist_path = RESULTS / "historico_evaluaciones.csv"
    
    # Generación de la nueva fila de métricas con marca de tiempo
    fila = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_holdout": int(len(X_test)),
        **{k: round(metricas[k], 4) for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
    }
    
    # Leer el archivo histórico actual si existe, de lo contrario inicializar vacío
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([fila])], ignore_index=True)
    hist.to_csv(hist_path, index=False)

    # 2. Paleta de colores para las líneas del gráfico
    AZUL = "#1f77b4"
    VERDE = "#2ca02c"
    NARANJA = "#ff7f0e"
    GRIS = "#7f7f7f"

    # Construcción del gráfico evolutivo
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ejex = range(1, len(hist) + 1)
    
    ax.plot(ejex, hist["accuracy"], marker="o", color=AZUL, label="Accuracy")
    ax.plot(ejex, hist["roc_auc"], marker="s", color=VERDE, label="ROC-AUC")
    ax.plot(ejex, hist["f1_score"], marker="^", color=NARANJA, label="F1-Score")
    
    ax.set_xticks(list(ejex))
    ax.set_xlabel("N.º de corrida")
    ax.set_ylabel("Métrica")
    ax.set_title("Histórico de rendimiento por corrida")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3)
    
    # Mensaje de ayuda visual en caso de ser la primera ejecución del pipeline
    if len(hist) == 1:
        ax.set_ylim(0, 1.05)
        ax.text(1, 0.5, "  1.ª corrida registrada.\n  Se acumula automáticamente\n  en cada ejecución.",
                fontsize=9, color=GRIS, va="center")
                
    # Guardar gráfico definitivo en formato PNG en la carpeta results/
    plt.savefig(RESULTS / "06b_historico_rendimiento.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Gráfico '06b_historico_rendimiento.png' guardado con éxito.")
    print(f"Corridas registradas acumuladas: {len(hist)} en {hist_path}")

except FileNotFoundError as e:
    print(f"Aviso: No se pudo actualizar el histórico por falta de archivos bases: {e}")
except Exception as e:
    print(f"Aviso: Error inesperado al procesar el gráfico histórico: {e}")


# ==========================================
# 12. GRÁFICO: DISTRIBUCIÓN DE CLASES — REAL VS. PREDICHO
# ==========================================
try:
    # 1. Cargar el modelo y los datos necesarios para generar las predicciones
    with open(MODELS / "modelo_default.pkl", "rb") as f:
        model = pickle.load(f)
    
    X_test = pd.read_csv(DATA / "X_test.csv")
    # Nota: Se asume que las etiquetas reales están guardadas en data/y_test.csv
    y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0]  
    
    # Generar predicciones en caliente
    y_pred = model.predict(X_test)

    # 2. Procesar las frecuencias de clases reales y predichas
    real = pd.Series(y_test).value_counts().reindex([0, 1]).fillna(0).astype(int)
    pred = pd.Series(y_pred).value_counts().reindex([0, 1]).fillna(0).astype(int)
    
    clases = ["Pagado (0)", "Default (1)"]
    x = np.arange(len(clases))
    w = 0.38

    # 3. Configuración de estilos visuales consistentes con el script
    AZUL = "#1f77b4"
    GRIS = "#7f7f7f"

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    b1 = ax.bar(x - w / 2, real.values, w, label="Real", color=GRIS)
    b2 = ax.bar(x + w / 2, pred.values, w, label="Predicho", color=AZUL)
    
    ax.set_xticks(x, clases)
    ax.set_ylabel("N.º de solicitudes")
    ax.set_title("Distribución de clases — real vs. predicho", fontweight="bold")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    ax.margins(y=0.15)
    
    # Reemplazo nativo de 'etiquetar_barras' usando la API de Matplotlib
    ax.bar_label(b1, fmt="{:,.0f}", fontsize=9.5, color="#555", padding=3)
    ax.bar_label(b2, fmt="{:,.0f}", fontsize=9.5, color="#333", padding=3)
    
    # Guardar el PNG directamente en la carpeta de resultados estructurada
    plt.savefig(RESULTS / "05_distribucion_clases.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Gráfico '05_distribucion_clases.png' guardado con éxito en results/")

    # 4. Cálculo e impresión del sesgo de estimación
    if real[1] > 0:
        sesgo = (pred[1] - real[1]) / real[1]
        print(f"Defaults reales: {real[1]:,} | predichos: {pred[1]:,} "
              f"({'sobre' if sesgo > 0 else 'sub'}estimación {abs(sesgo):.1%})")
    else:
        print(f"Defaults reales: 0 | predichos: {pred[1]:,}")

except FileNotFoundError as e:
    print(f"Aviso: No se pudo generar el gráfico de distribución de clases por archivos faltantes: {e}")
except Exception as e:
    print(f"Aviso: Error inesperado al procesar la distribución de clases real vs. predicho: {e}")