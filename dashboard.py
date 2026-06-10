"""Dashboard de integracion del proyecto LACDA (IL 3.3).

Integra en una sola vista las tres capas del sistema:
  1. Capa de datos: estado de la tabla `loan_data` en PostgreSQL.
  2. Monitoreo del pipeline: KPI de calidad del dato crudo con alertas.
  3. Capa ML: metricas y graficos del modelo Random Forest (results/).

Uso:
  pip install streamlit
  docker compose up -d db          (la seccion 1 consulta la BD en vivo)
  streamlit run dashboard.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "scripts"))
os.environ.setdefault("DB_HOST", "localhost")  # fuera de Docker, la BD esta en localhost

DATA = BASE / "data"
RESULTS = BASE / "results"

st.set_page_config(page_title="LACDA — Dashboard DataOps", layout="wide")
st.title("📊 Proyecto LACDA — Dashboard de integración")
st.caption(
    "Pipeline DataOps → PostgreSQL → Modelo de predicción de default · "
    "ITY1101 Gestión de Datos para IA, DUOC UC — Grupo 4"
)

# ---------------------------------------------------------------------
# 1. Capa de datos (PostgreSQL)
# ---------------------------------------------------------------------
st.header("1 · Capa de datos — tabla `loan_data` (PostgreSQL)")


@st.cache_data(ttl=60)
def resumen_bd() -> pd.Series:
    from validacion import get_engine

    query = """
        SELECT COUNT(*) AS filas,
               SUM(loan_status) AS defaults,
               MAX(fecha_carga) AS ultima_carga
        FROM loan_data
    """
    return pd.read_sql(query, get_engine()).iloc[0]


try:
    r = resumen_bd()
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas cargadas", f"{int(r['filas']):,}")
    c2.metric("Tasa de default", f"{r['defaults'] / r['filas']:.1%}")
    c3.metric("Última carga", str(r["ultima_carga"])[:19])
except Exception as exc:
    st.error(
        f"Base de datos no disponible ({type(exc).__name__}). "
        "Levántala con `docker compose up -d db` y corre el pipeline si está vacía."
    )

# ---------------------------------------------------------------------
# 2. Monitoreo del pipeline (KPI de calidad)
# ---------------------------------------------------------------------
st.header("2 · Monitoreo — KPI de calidad del dato crudo")

csv_raw = DATA / "loan_data_raw.csv"
if csv_raw.exists():
    from qualitycheck import PESOS, QualityCheck, nivel_alerta

    score, dims = QualityCheck(pd.read_csv(csv_raw)).quality_score()
    nivel = nivel_alerta(score)
    icono = {"OK": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}[nivel]

    c1, c2 = st.columns([1, 2])
    c1.metric("Quality score (0-100)", f"{score}", f"nivel {nivel} {icono}", delta_color="off")
    tabla = pd.DataFrame({"% filas afectadas": dims, "peso": PESOS}).round(2)
    c2.bar_chart(tabla["% filas afectadas"])
    st.caption(
        "Score ponderado por 4 dimensiones (nulos 0.30, duplicados 0.20, outliers 0.20, "
        "inconsistencias 0.30). Umbrales de alerta: WARNING < 70, CRITICAL < 50."
    )
else:
    st.warning("No existe data/loan_data_raw.csv — corre el pipeline primero (`docker compose up`).")

# ---------------------------------------------------------------------
# 3. Capa ML (modelo Random Forest)
# ---------------------------------------------------------------------
st.header("3 · Modelo — Random Forest sobre el holdout (9.000 filas)")

metricas_path = RESULTS / "metricas.json"
if metricas_path.exists():
    met = json.loads(metricas_path.read_text(encoding="utf-8"))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{met['roc_auc']:.4f}")
    c2.metric("F1 (default)", f"{met['f1_score']:.4f}")
    c3.metric("Recall", f"{met['recall']:.4f}")
    c4.metric("Precision", f"{met['precision']:.4f}")
    c5.metric("Accuracy", f"{met['accuracy']:.4f}")

    col_izq, col_der = st.columns(2)
    if (RESULTS / "matriz_confusion.png").exists():
        col_izq.image(str(RESULTS / "matriz_confusion.png"), caption="Matriz de confusión")
    if (RESULTS / "curva_roc.png").exists():
        col_der.image(str(RESULTS / "curva_roc.png"), caption="Curva ROC")

    if (RESULTS / "importancia_variables.csv").exists():
        st.subheader("Importancia de variables")
        imp = pd.read_csv(RESULTS / "importancia_variables.csv", index_col=0)
        st.bar_chart(imp["importancia"])
else:
    st.warning(
        "No hay resultados del modelo — corre `python scripts/train_model.py` "
        "y `python scripts/test_model.py`."
    )

# ---------------------------------------------------------------------
# Evidencias
# ---------------------------------------------------------------------
log_path = BASE / "docs" / "evidencias_pipeline.log"
if log_path.exists():
    with st.expander("📜 Logs de la última corrida del pipeline (evidencia)"):
        st.code(
            "\n".join(
                l for l in log_path.read_text(encoding="utf-8").splitlines()
                if "lacda_app" in l
            ),
            language="text",
        )
