import pandas as pd
import pickle
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from correlation_filter import CorrelationFilter

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Crear carpeta de resultados
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------
# Cargar datos
# -------------------------------------------------
data = pd.read_csv("data/data_fraude.csv")

# Variable objetivo
target = "is_fraud"

X = data.drop(columns=[target])
y = data[target]

# Revisa la distribución de la variable objetivo
# En este caso se obtiene un gráfico de torta
data[target].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                 labels=['Legitima', 'Fraude'],
                                 figsize=(6, 6))
plt.title("Distribución de variable objetivo", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "distribucion_clases.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# -------------------------------------------------
# Variables categóricas y numéricas
# -------------------------------------------------
categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# -------------------------------------------------
# Preprocesamiento
# -------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# -------------------------------------------------
# Modelo
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=29,
    n_jobs=-1,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# -------------------------------------------------
# Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=29,
    stratify=y
)

# -------------------------------------------------
# Entrenamiento
# -------------------------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------------------------
# Guardar modelo
# -------------------------------------------------
with open("models/modelo_fraude.pkl", "wb") as f:
    pickle.dump(pipeline, f)
    print("Modelo guardado como modelo_fraude.pkl")

print("Entrenamiento finalizado ...")