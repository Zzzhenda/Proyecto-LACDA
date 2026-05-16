"""
Modulo de procesamiento: limpieza de las entidades Solicitante y Prestamo.

Flujo del pipeline modular (Requisitos Duoc):
1. Carga de datos unificados desde la base de datos Postgres (Raw).
2. Eliminacion de duplicados (unica operacion que reduce filas).
3. Aplicacion de reglas tecnicas del Cap. 9 modificando valores (imputacion y clip).
4. Aplicacion de la tecnica Winsorization al 5% usando la clase interna.
5. Persistencia limpia en Postgres y exportacion paralela a los archivos CSV.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text

# Conexiones internas de tu infraestructura
from db import get_engine

# ---------------------------------------------------------
# CLASE CUSTOM WINSORIZER (Integrada internamente)
# ---------------------------------------------------------
class Winsorizer:
    """
    Tratamiento de atípicos personalizado para evitar librerías externas.
    Descarta el % de los extremos usando cuantiles de pandas y np.clip.
    """
    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits
        self.columns_ = None

    def fit(self, X, y=None):
        # Guardar nombres si es DataFrame, si no generar nombres genéricos
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            self.columns_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        for col in self.columns_:
            lower = X[col].quantile(self.limits[0])
            upper = X[col].quantile(1 - self.limits[1])
            X = X.astype("float64")
            X[col] = np.clip(X[col], lower, upper)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.columns_)
        else:
            return np.array(input_features)


# ---------------------------------------------------------
# CONFIGURACIÓN DE ENTORNO Y CONSTANTES
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[limpieza] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_OUT_SOL = DATA_DIR / "solicitantes_clean.csv"
CSV_OUT_PRE = DATA_DIR / "prestamos_clean.csv"

COLS_SOLICITANTE = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "cb_person_cred_hist_length",
    "credit_score",
]

COLS_PRESTAMO = [
    "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income",
    "previous_loan_defaults_on_file", "loan_status",
]

CAT_DOMAINS = {
    "person_gender": {"male", "female"},
    "person_education": {"High School", "Bachelor", "Master", "Associate", "Doctorate"},
    "person_home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "loan_intent": {
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "DEBTCONSOLIDATION", "HOMEIMPROVEMENT",
    },
    "previous_loan_defaults_on_file": {"Yes", "No"},
}


# ---------------------------------------------------------
# COMPONENTES MODULARES DEL PIPELINE
# ---------------------------------------------------------

def extraer_datos_raw(engine) -> pd.DataFrame:
    """Realiza la consulta SQL para extraer el join plano unificado de las tablas raw."""
    logging.info("Consultando datos iniciales (join plano) desde Postgres...")
    query = """
        SELECT
            s.person_age, s.person_gender, s.person_education, s.person_income,
            s.person_emp_exp, s.person_home_ownership, s.cb_person_cred_hist_length,
            s.credit_score,
            p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income,
            p.previous_loan_defaults_on_file, p.loan_status
        FROM solicitantes_raw s
        JOIN prestamos_raw p ON p.solicitante_id = s.id
        ORDER BY s.id
    """
    df = pd.read_sql(query, engine)
    logging.info(f"Filas leídas desde la base de datos: {len(df)}")
    return df


def remover_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica y remueve duplicados exactos en el conjunto de datos."""
    n_inicial = len(df)
    df_limpio = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Removidas por duplicados: {n_inicial - len(df_limpio)}")
    return df_limpio


def aplicar_reglas_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modifica valores anomalos basandose en las especificaciones del Cap 9.
    No elimina filas adicionales; usa imputacion por estadistica central y limites logicos.
    """
    logging.info("Aplicando correccion de datos segun limites y dominios de negocio...")
    df = df.copy()

    # 1. Tratamiento preventivo de nulos mediante imputacion analitica (mediana/moda)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Reglas estructurales de variables numericas del Solicitante (Uso de .clip)
    if "person_age" in df.columns:
        df["person_age"] = df["person_age"].clip(18, 100)
    if "person_emp_exp" in df.columns and "person_age" in df.columns:
        df["person_emp_exp"] = df["person_emp_exp"].clip(lower=0, upper=df["person_age"] - 18)
    if "person_income" in df.columns:
        df["person_income"] = df["person_income"].clip(lower=0)
    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].clip(300, 850)
    if "cb_person_cred_hist_length" in df.columns and "person_age" in df.columns:
        df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].clip(lower=0, upper=df["person_age"])

    # 3. Reglas estructurales de variables numericas del Prestamo
    if "loan_amnt" in df.columns:
        df["loan_amnt"] = df["loan_amnt"].apply(lambda x: x if x > 0 else 1)
    if "loan_int_rate" in df.columns:
        df["loan_int_rate"] = df["loan_int_rate"].clip(5, 30)
    if "loan_percent_income" in df.columns:
        df["loan_percent_income"] = df["loan_percent_income"].clip(0, 1)

    # 4. Mapeo controlado sobre dominios categoricos cerrados (Valores extraños se reemplazan por la moda)
    for col, dominio in CAT_DOMAINS.items():
        if col in df.columns:
            moda_cat = df[col].mode()[0]
            df[col] = df[col].apply(lambda x: x if x in dominio else moda_cat)
            
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].apply(lambda x: x if x in [0, 1] else 0)

    logging.info("Transformacion de consistencia logica completada.")
    return df


def aplicar_winsorizacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la tecnica de Winsorization con limites fijos de percentil del 5%.
    Utiliza la clase interna declarada al principio de este archivo.
    """
    cols_objetivo = ['person_age', 'credit_score', 'person_income', 'loan_amnt',
                     'Age', 'CreditScore', 'Balance', 'EstimatedSalary']
    
    columnas_presentes = [col for col in cols_objetivo if col in df.columns]

    if not columnas_presentes:
        logging.warning("No se detectaron columnas compatibles para Winsorization. Continuando...")
        return df

    logging.info(f"Aplicando Winsorization (Límite 5% - Ambos Extremos) sobre: {columnas_presentes}")

    # Instanciamos el Winsorizer interno
    winsorizer = Winsorizer(limits=(0.05, 0.05))
    
    # Transformacion inplace controlada del bloque numerico
    df[columnas_presentes] = winsorizer.fit_transform(df[columnas_presentes])
    
    logging.info("Valores outliers ajustados correctamente dentro de los rangos de cuantiles.")
    return df


def persistir_y_exportar_datos(df: pd.DataFrame, engine) -> None:
    """Carga los datos limpios en la Base de Datos e imprime los archivos CSV finales respetando la integridad referencial."""
    
    # 1. Truncar tablas en cascada para evitar duplicaciones residuales
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamos_clean RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE solicitantes_clean RESTART IDENTITY CASCADE"))

    # 2. Insercion de Solicitantes
    df[COLS_SOLICITANTE].to_sql(
        "solicitantes_clean", engine, if_exists="append", index=False, chunksize=1000
    )

    # Recuperacion estructurada de IDs autogenerados por Postgres
    with engine.connect() as conn:
        ids = pd.read_sql("SELECT id FROM solicitantes_clean ORDER BY id", conn)

    # 3. Vinculacion de llave foranea e insercion de Prestamos
    df_pre = df[COLS_PRESTAMO].copy()
    df_pre.insert(0, "solicitante_id", ids["id"].values)
    df_pre.to_sql("prestamos_clean", engine, if_exists="append", index=False, chunksize=1000)

    # 4. Exportacion paralela a sistema de archivos CSV (Formato Duoc)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    df_sol_csv = df[COLS_SOLICITANTE].copy()
    df_sol_csv.insert(0, "id", ids["id"].values)
    
    df_sol_csv.to_csv(CSV_OUT_SOL, index=False)
    df_pre.to_csv(CSV_OUT_PRE, index=False)

    logging.info(f"Estructura relacional guardada en Base de Datos Postgres.")
    logging.info(f"CSV de solicitantes guardado en: {CSV_OUT_SOL}")
    logging.info(f"CSV de préstamos guardado en: {CSV_OUT_PRE}")
    logging.info(f"Volumetría final procesada: {len(df)} registros.")


# ---------------------------------------------------------
# CONTROLADOR PRINCIPAL
# ---------------------------------------------------------

def main() -> None:
    logging.info("=== INICIANDO PIPELINE DE LIMPIEZA ===")
    engine = get_engine()

    # Ejecucion coordinada paso a paso
    df = extraer_datos_raw(engine)
    
    if df.empty:
        logging.error("Las tablas raw estan vacias. Deteniendo operacion.")
        sys.exit(1)

    df = remover_duplicados(df)
    df = aplicar_reglas_negocio(df)
    df = aplicar_winsorizacion(df)
    persistir_y_exportar_datos(df, engine)
    
    logging.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()