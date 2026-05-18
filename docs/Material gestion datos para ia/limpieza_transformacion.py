import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from feature_engineering import FeatureEngineering
from correlation_filter import CorrelationFilter
from winsorizer import Winsorizer

def tratar_duplicados(X : pd.DataFrame, drop = True):
  """
  Tratamiento de duplicados

  Parámetros
  ----------
  X : DataFrame
    Conjunto de datos.
  drop : bool
    Si se deben eliminar los duplicados.

  Retorna
  -------
  DataFrame
    Conjunto de datos sin duplicados.
  """
  return X.drop_duplicates() if drop else X

target = "is_fraud"
# Define variables independientes
X = data_for_preparation.drop(columns=[target], errors="ignore")
# Rescata variable objetivo
y = data_for_preparation[target]
# Separa las características en cuantitativas y cualitativas
num_cols = [] # Acá van sus columnas cuantitativas

cat_cols = [] # Acá van sus columnas cualitativas

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("winsorizer", Winsorizer()),
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

# Esto se debe adaptar de acuerdo a las columnas que vaya a necesitar su tarea de ingenieria de características
fe = FeatureEngineering()

pipeline_preparacion = Pipeline(steps=[
    ("duplicados", FunctionTransformer(tratar_duplicados,
                                       kw_args={"drop": True})),
    ("feature_engineering", fe),
    ("preprocesador", preprocessor),
    ("colinealidad", CorrelationFilter(threshold=0.9))
])

pipeline_preparacion.fit(X)

# Obtener nombres del preprocesador
feature_names = pipeline_preparacion.named_steps["preprocesador"].get_feature_names_out()

# Pasarlos al filtro
pipeline_preparacion.named_steps["colinealidad"].set_feature_names(feature_names)

# Transformar
X_transformada = pipeline_preparacion.transform(X)

# Usar nombres correctos
cols_finales = pipeline_preparacion.named_steps["colinealidad"].get_feature_names_out()

# Arma el dataframe final
data_transformada = pd.DataFrame(X_transformada, columns=cols_finales)

# Guarda la data preparada (limpia y transformada)
data_transformada.to_csv("data_limpia.csv")