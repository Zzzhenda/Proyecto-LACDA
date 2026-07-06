import json
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# 1. Análisis y manipulación de datos (Data Science Core)
import numpy as np
import pandas as pd

# 2. Visualización de datos y gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Machine Learning y Procesamiento (Scikit-Learn)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 4. Streamlit e Integración UI
import streamlit as st

import os  #  Necesario para leer variables de entorno de Docker
st.set_page_config(
    page_title="Dashboard de Monitoreo - Pipeline de Crédito",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SISTEMA DE AUTENTICACIÓN
# ==========================================
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.title("Acceso Restringido — Créditos LACDA")
    st.markdown("Por favor, introduce tus credenciales para acceder al sistema.")
    
    usuario_ingresado = st.text_input("Usuario", key="username")
    password_ingresado = st.text_input("Contraseña", type="password", key="password")
    
    # Lee las variables que configuraste en tu archivo .env
    USER_CORRECTO = os.environ.get("APP_USER", "admin_lacda")
    PASSWORD_CORRECTO = os.environ.get("APP_PASSWORD", "password_lacda")

    if st.button("Iniciar Sesión"):
        if usuario_ingresado == USER_CORRECTO and password_ingresado == PASSWORD_CORRECTO:
            st.session_state["authenticated"] = True
            st.success("¡Acceso concedido!")
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")
            
    return False

# Si no está autenticado, detiene la ejecución de todo el dashboard que viene abajo
if not check_password():
    st.stop()

# Módulos locales personalizados (Solo validación)
try:
    from validacion import validar_estructura, validar_features, validar_semantica
except ImportError:
    pass  # Tolerancia si el script de entorno no requiere validaciones nativas directo en UI

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA (STREAMLIT)
# ==========================================
st.set_page_config(
    page_title="Dashboard de Monitoreo - Pipeline de Crédito",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Dashboard de Control Integral — Créditos LACDA")
st.markdown("Monitoreo unificado del ciclo de vida de los datos y el rendimiento del modelo Predictivo.")
st.divider()  # CORRECCIÓN: Reemplaza st.hr() que causaba el AttributeError

# ==========================================
# CONFIGURACIÓN DE RUTAS (Adaptada para Docker)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".")
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Crear directorios si no existen para evitar errores de escritura
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)

# Paleta de colores global
AZUL = "#1f77b4"
VERDE = "#2ca02c"
GRIS = "#7f7f7f"
NARANJA = "#ff7f0e"
ROJO = "#d62728"

# ==========================================
# CARGA DE DATOS BASE Y PROCESAMIENTO ML
# ==========================================
@st.cache_data(show_spinner="Cargando datos transformados...")
def cargar_datos_base():
    if (DATA_DIR / "loan_data_transformed.csv").exists():
        df_t = pd.read_csv(DATA_DIR / "loan_data_transformed.csv")
        y = df_t["loan_status"]
        return df_t, y
    return None, None

df_t, y = cargar_datos_base()

if df_t is None:
    st.error(f"Error crítico: No se encontró el archivo `{DATA_DIR / 'loan_data_transformed.csv'}`. El dashboard no puede inicializarse por completo.")
    st.stop()

# Listas de variables para correlación
ORIGINALES = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file",
]
TODAS = [c for c in ORIGINALES if c != "previous_loan_defaults_on_file"] + [
    "rate_x_pct_income", "loan_burden", "has_prev_defaults",
]

# Inicialización de variables de soporte para inferencia interactiva
metricas = {}
t_infer = 0.0
X_test = pd.DataFrame()
y_test = pd.Series(dtype=int)
y_pred = np.array([])

# Carga segura de artefactos de Machine Learning e Inferencia en Caliente
try:
    if (RESULTS_DIR / "metricas.json").exists():
        with open(RESULTS_DIR / "metricas.json", "r", encoding="utf-8") as f:
            metricas = json.load(f)

    if (DATA_DIR / "X_test.csv").exists():
        X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    
    if (MODELS_DIR / "modelo_default.pkl").exists() and len(X_test) > 0:
        with open(MODELS_DIR / "modelo_default.pkl", "rb") as f:
            model = pickle.load(f)
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        t_infer = time.perf_counter() - t0
    
    if (DATA_DIR / "y_test.csv").exists():
        y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

except Exception as e:
    st.sidebar.warning(f" Algunos componentes de ML no se pudieron cargar: {e}")

# ==========================================
# RE-GENERACIÓN SILENCIOSA DE ARTEFACTOS FALTANTES
# ==========================================
if metricas and len(X_test) > 0:
    try:
        # 1. Re-generar KPI Banner si no existe
        if not (RESULTS_DIR / "01_kpis.png").exists():
            fig_kpi, axes = plt.subplots(1, 5, figsize=(12.5, 1.9))
            kpis_list = [
                ("Accuracy", f"{metricas.get('accuracy', 0)*100:.1f}%", AZUL),
                ("Recall (default)", f"{metricas.get('recall', 0)*100:.1f}%", VERDE),
                ("F1 (default)", f"{metricas.get('f1_score', 0)*100:.1f}%", AZUL),
                ("ROC-AUC", f"{metricas.get('roc_auc', 0):.3f}", VERDE),
                ("Tiempo infer.", f"{t_infer*1000:.0f} ms", GRIS),
            ]
            for ax_k, (etq, val, color) in zip(axes, kpis_list):
                ax_k.axis("off")
                ax_k.add_patch(plt.Rectangle((0.05, 0.10), 0.90, 0.80, transform=ax_k.transAxes, facecolor=color, alpha=0.10, edgecolor=color, lw=1.4))
                ax_k.text(0.5, 0.60, val, transform=ax_k.transAxes, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
                ax_k.text(0.5, 0.26, etq, transform=ax_k.transAxes, ha="center", va="center", fontsize=9, color="#333")
            fig_kpi.suptitle(f"Indicadores clave del modelo (holdout de {len(X_test):,} solicitudes)", fontsize=11, fontweight="bold", y=1.05)
            plt.savefig(RESULTS_DIR / "01_kpis.png", dpi=300, bbox_inches="tight")
            plt.close(fig_kpi)

        # 2. Re-generar Distribución Real vs Predicho si falta
        if not (RESULTS_DIR / "05_distribucion_clases.png").exists() and len(y_test) > 0 and len(y_pred) > 0:
            real = pd.Series(y_test).value_counts().reindex([0, 1]).fillna(0).astype(int)
            pred = pd.Series(y_pred).value_counts().reindex([0, 1]).fillna(0).astype(int)
            clases = ["Pagado (0)", "Default (1)"]
            x = np.arange(len(clases))
            w = 0.38
            fig, ax = plt.subplots(figsize=(7.5, 4.3))
            b1 = ax.bar(x - w / 2, real.values, w, label="Real", color=GRIS)
            b2 = ax.bar(x + w / 2, pred.values, w, label="Predicho", color=AZUL)
            ax.set_xticks(x, clases)
            ax.set_ylabel("N.º de solicitudes")
            ax.set_title("Distribución de clases — real vs. predicho", fontweight="bold")
            ax.grid(axis="x", visible=False)
            ax.legend(frameon=False)
            ax.margins(y=0.15)
            ax.bar_label(b1, fmt="{:,.0f}", fontsize=9.5, color="#555", padding=3)
            ax.bar_label(b2, fmt="{:,.0f}", fontsize=9.5, color="#333", padding=3)
            plt.savefig(RESULTS_DIR / "05_distribucion_clases.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        # 3. Re-generar Matriz de correlación si falta
        if not (RESULTS_DIR / "matriz_correlacion_train.png").exists():
            num_vars = df_t[df_t.columns.intersection(TODAS)].select_dtypes("number")
            if not num_vars.empty:
                fig, ax = plt.subplots(figsize=(10, 8.5))
                sns.heatmap(num_vars.corr().round(2), cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1, cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title("Matriz de correlación — Variables Numéricas", fontweight="bold")
                fig.tight_layout()
                plt.savefig(RESULTS_DIR / "matriz_correlacion_train.png", dpi=300)
                plt.close(fig)

    except Exception as e:
        st.sidebar.error(f"Error generando imágenes de soporte técnico: {e}")

# ==========================================
# BANNER GLOBAL DE KPIs
# ==========================================
if (RESULTS_DIR / "01_kpis.png").exists():
    st.image(str(RESULTS_DIR / "01_kpis.png"), width='stretch')  # CORRECCIÓN: Reemplaza width="stretch"
else:
    st.info("Los indicadores clave se mostrarán aquí una vez que finalice la ejecución del pipeline.")

# ==========================================
# DEFINICIÓN DE PESTAÑAS (Mapeo unificado)
# ==========================================
tab1, tab2, tab3 = st.tabs([
    " Monitoreo del Pipeline (DataOps)", 
    " Fase de Entrenamiento (ML Train)", 
    " Evaluación Ex-Post (ML Test Holdout)"
])

# ==========================================
# PESTAÑA 1: MONITOREO DEL PIPELINE
# ==========================================
with tab1:
    st.header("Flujo, Depuración y Escalabilidad de Datos")
    st.markdown("Métricas operacionales extraídas durante las etapas secuenciales de ingesta y validación.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volumen de Registros por Etapa")
        
        # Calcular dinámicamente y guardar volúmenes del pipeline si falta el archivo
        try:
            def contar_csv(ruta: Path):
                return len(pd.read_csv(ruta)) if ruta.exists() else None

            n_ingesta = contar_csv(DATA_DIR / "loan_data_raw.csv")
            n_limpieza = contar_csv(DATA_DIR / "loan_data_clean.csv")
            n_transf = contar_csv(DATA_DIR / "loan_data_transformed.csv")
            n_bd = int(len(df_t))
            n_test = int(len(X_test)) if len(X_test) > 0 else None
            n_train = n_transf - n_test if (n_transf and n_test) else None

            df_etapas = pd.DataFrame({
                "etapa": ["1 · Ingesta", "3 · Limpieza", "4 · Transformación", "5 · Carga BD", "ML · Entrenamiento", "ML · Holdout"],
                "filas": [n_ingesta, n_limpieza, n_transf, n_bd, n_train, n_test],
            })
            df_etapas.to_csv(RESULTS_DIR / "volumen_pipeline.csv", index=False)
            
            if not (RESULTS_DIR / "02_volumen_etapas.png").exists():
                fig, ax = plt.subplots(figsize=(8.5, 4.5))
                barras = ax.barh(df_etapas["etapa"], df_etapas["filas"], color=[AZUL, AZUL, AZUL, VERDE, GRIS, NARANJA])
                ax.invert_yaxis()
                ax.set_xlabel("Registros")
                ax.set_title("Volumen de datos por etapa del pipeline")
                ax.margins(x=0.12)
                ax.grid(axis="y", visible=False)
                ax.bar_label(barras, fmt="{:,.0f}", fontsize=10, color="#333", padding=5)

                descartados = (n_ingesta - n_limpieza) if (n_ingesta and n_limpieza) else 0
                tasa_retencion = (n_limpieza / n_ingesta) if (n_ingesta and n_ingesta > 0) else 0
                ax.annotate(f"Descartados en limpieza: {descartados:,}  (retención {tasa_retencion:.1%})",
                            xy=(0.5, -0.20), xycoords="axes fraction", ha="center",
                            fontsize=9.5, color=ROJO if descartados else GRIS)
                
                plt.savefig(RESULTS_DIR / "02_volumen_etapas.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            st.error(f"Error procesando volumen del pipeline: {e}")

        if (RESULTS_DIR / "02_volumen_etapas.png").exists():
            st.image(str(RESULTS_DIR / "02_volumen_etapas.png"), width='stretch')
        
        if (RESULTS_DIR / "volumen_pipeline.csv").exists():
            with st.expander("Ver desglose numérico de registros"):
                df_vol = pd.read_csv(RESULTS_DIR / "volumen_pipeline.csv")
                st.dataframe(df_vol, width='stretch')
                
    with col2:
        st.subheader("Análisis de Escalabilidad de Validaciones")
        
        # Simulación controlada del estrés de infraestructura de validaciones locales
        try:
            if 'validar_estructura' in globals() and not (RESULTS_DIR / "escalabilidad_pipeline.png").exists():
                filas_bench = []
                for factor in (1, 2, 4, 8):
                    df_n = pd.concat([df_t], ignore_index=True) if factor == 1 else pd.concat([df_t] * factor, ignore_index=True)
                    t0 = time.perf_counter()
                    _ = (validar_estructura(df_n) + validar_semantica(df_n) + validar_features(df_n))
                    t_valida = time.perf_counter() - t0
                    filas_bench.append({"filas": len(df_n), "validacion (s)": t_valida})

                bench = pd.DataFrame(filas_bench).set_index("filas")
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                bench.plot(marker="o", ax=ax, color=NARANJA, legend=False)
                ax.set_title("Escalabilidad de las operaciones de validación")
                ax.set_ylabel("segundos")
                ax.set_xlabel("filas procesadas")
                fig.tight_layout()
                plt.savefig(RESULTS_DIR / "escalabilidad_pipeline.png", dpi=300)
                plt.close(fig)
        except Exception as e:
            st.error(f"Error generando gráfico de escalabilidad: {e}")

        if (RESULTS_DIR / "escalabilidad_pipeline.png").exists():
            st.image(str(RESULTS_DIR / "escalabilidad_pipeline.png"), width='stretch')
        else:
            st.caption("Gráfico de rendimiento de infraestructura no disponible o saltado.")

    st.divider()
    st.subheader("Estructura de Correlaciones del Dataset")
    if (RESULTS_DIR / "matriz_correlacion_train.png").exists():
        st.image(str(RESULTS_DIR / "matriz_correlacion_train.png"), width='stretch', 
                 caption="Intercorrelación de variables numéricas procesadas en el pipeline.")

# ==========================================
# PESTAÑA 2: ENTRENAMIENTO DEL MODELO
# ==========================================
with tab2:
    st.header("Ajuste y Curvas de Aprendizaje del Modelo")
    st.markdown("Análisis del comportamiento del algoritmo Random Forest según el volumen disponible.")
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        st.subheader("Balance Inicial de Clases")
        
        # Generar gráfico estático si no existe en disco
        if not (RESULTS_DIR / "distribucion_clases.png").exists() and not (RESULTS_DIR / "distribucion_target.png").exists():
            try:
                fig, ax = plt.subplots(figsize=(5, 4.2))
                y.value_counts().rename({0: "pagado", 1: "default"}).plot.bar(ax=ax, rot=0, color=[AZUL, NARANJA])
                ax.set_title("Distribución de la variable objetivo")
                ax.set_ylabel("Cantidad")
                ax.bar_label(ax.containers[0])
                fig.tight_layout()
                plt.savefig(RESULTS_DIR / "distribucion_target.png", dpi=300)
                plt.close(fig)
            except Exception:
                pass

        if (RESULTS_DIR / "distribucion_clases.png").exists():
            st.image(str(RESULTS_DIR / "distribucion_clases.png"), width='stretch', caption="Proporción original Pagados vs Defaults.")
        elif (RESULTS_DIR / "distribucion_target.png").exists():
            st.image(str(RESULTS_DIR / "distribucion_target.png"), width='stretch', caption="Proporción original Pagados vs Defaults (Target).")
            
    with col4:
        st.subheader("Curva de Aprendizaje (Rendimiento vs Volumen)")
        if (RESULTS_DIR / "curva_aprendizaje.png").exists():
            st.image(str(RESULTS_DIR / "curva_aprendizaje.png"), width='stretch',
                     caption="Evolución del Accuracy y ROC-AUC a medida que crece el set de entrenamiento.")
        else:
            st.warning("No se encontró el gráfico 'curva_aprendizaje.png'. Ejecuta el script de entrenamiento para generarlo.")

# ==========================================
# PESTAÑA 3: EVALUACIÓN EX-POST
# ==========================================
with tab3:
    st.header("Puerta de Calidad: Rendimiento sobre Datos de Control")
    st.markdown("Evaluación estratégica realizada estrictamente sobre el 20% de datos en Holdout (nunca vistos por el modelo).")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Matriz de Confusión")
        if (RESULTS_DIR / "matriz_confusion.png").exists():
            st.image(str(RESULTS_DIR / "matriz_confusion.png"), width='stretch',
                     caption="Matriz de aciertos y errores de clasificación en el Holdout.")
        else:
            st.info("Matriz de confusión precalcolada no disponible en directorio 'results'.")
            
        st.subheader("Distribución de Clases: Real vs. Predicho")
        if (RESULTS_DIR / "05_distribucion_clases.png").exists():
            st.image(str(RESULTS_DIR / "05_distribucion_clases.png"), width='stretch',
                     caption="Comparativa del sesgo de estimación del modelo contra la realidad.")
            
            # Despliegue informativo del análisis estadístico de sesgo
            if len(y_test) > 0 and len(y_pred) > 0:
                real_c = pd.Series(y_test).value_counts().reindex([0, 1]).fillna(0)
                pred_c = pd.Series(y_pred).value_counts().reindex([0, 1]).fillna(0)
                if real_c[1] > 0:
                    sesgo = (pred_c[1] - real_c[1]) / real_c[1]
                    tipo_sesgo = "sobreestimación" if sesgo > 0 else "subestimación"
                    st.info(f"**Análisis de Sesgo:** Defaults reales: `{int(real_c[1]):,}` | Predichos: `{int(pred_c[1]):,}` ({tipo_sesgo} del **{abs(sesgo):.1%}**)")

    with col6:
        st.subheader("Análisis Discriminante Probabilístico")
        if (RESULTS_DIR / "curva_roc.png").exists():
            st.image(str(RESULTS_DIR / "curva_roc.png"), width='stretch', caption="Capacidad de discriminación (ROC).")
            
        if (RESULTS_DIR / "curva_precision_recall.png").exists():
            st.image(str(RESULTS_DIR / "curva_precision_recall.png"), width='stretch', caption="Compromiso entre Precisión y Sensibilidad (Recall).")

    st.divider()
    st.subheader("Estabilidad Temporal del Clasificador")
    
    # Manejo del Histórico acumulativo de Evaluaciones por corrida
    try:
        hist_path = RESULTS_DIR / "historico_evaluaciones.csv"
        if metricas and len(X_test) > 0:
            fila_hist = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "n_holdout": int(len(X_test)),
                **{k: round(metricas[k], 4) for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if k in metricas},
            }
            hist_df = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
            
            # Evita registrar duplicados instantáneos al hacer refresh
            if hist_df.empty or hist_df.iloc[-1]["accuracy"] != fila_hist["accuracy"]:
                hist_df = pd.concat([hist_df, pd.DataFrame([fila_hist])], ignore_index=True)
                hist_df.to_csv(hist_path, index=False)

            if not (RESULTS_DIR / "06b_historico_rendimiento.png").exists() or len(hist_df) > 1:
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ejex = range(1, len(hist_df) + 1)
                ax.plot(ejex, hist_df["accuracy"], marker="o", color=AZUL, label="Accuracy")
                if "roc_auc" in hist_df.columns: ax.plot(ejex, hist_df["roc_auc"], marker="s", color=VERDE, label="ROC-AUC")
                if "f1_score" in hist_df.columns: ax.plot(ejex, hist_df["f1_score"], marker="^", color=NARANJA, label="F1-Score")
                
                ax.set_xticks(list(ejex))
                ax.set_xlabel("N.º de corrida")
                ax.set_ylabel("Métrica")
                ax.set_title("Histórico de rendimiento por corrida")
                ax.grid(axis="x", visible=False)
                ax.legend(frameon=False, ncol=3)
                
                if len(hist_df) == 1:
                    ax.set_ylim(0, 1.05)
                    ax.text(1, 0.5, "  1.ª corrida registrada. Se acumulará secuencialmente.", fontsize=9, color=GRIS, va="center")
                            
                plt.savefig(RESULTS_DIR / "06b_historico_rendimiento.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
    except Exception as e:
        st.error(f"Error procesando el gráfico histórico: {e}")

    if (RESULTS_DIR / "06b_historico_rendimiento.png").exists():
        st.image(str(RESULTS_DIR / "06b_historico_rendimiento.png"), width='stretch',
                 caption="Evolución histórica de las métricas clave a través de las sucesivas ejecuciones del pipeline.")