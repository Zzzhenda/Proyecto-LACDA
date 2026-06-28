import json
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

def main():
    BASE_DIR = Path('.')
    POWERBI_DIR = BASE_DIR / 'powerbi'
    POWERBI_DIR.mkdir(exist_ok=True)

    print("Generando CSVs para Power BI...")

    # 1. Métricas
    print("- Generando metricas_modelo.csv")
    with open(BASE_DIR / 'results' / 'metricas.json', 'r', encoding='utf-8') as f:
        m = json.load(f)
    gini = 2 * m['roc_auc'] - 1
    metricas_df = pd.DataFrame([
        ('Accuracy', m['accuracy']),
        ('Precision (default)', m['precision']),
        ('Recall (default)', m['recall']),
        ('F1-score (default)', m['f1_score']),
        ('ROC-AUC', m['roc_auc']),
        ('Gini', gini)
    ], columns=['metrica', 'valor']).round(4)
    metricas_df.to_csv(POWERBI_DIR / 'metricas_modelo.csv', index=False, encoding='utf-8')

    # 2. Matriz de confusión
    print("- Generando matriz_confusion.csv")
    # Utilizamos pickle porque train_model.py lo guardó con pickle
    with open(BASE_DIR / 'models' / 'modelo_default.pkl', 'rb') as f:
        mod = pickle.load(f)
        
    X = pd.read_csv(BASE_DIR / 'data' / 'X_test.csv')
    y = pd.read_csv(BASE_DIR / 'data' / 'y_test.csv').squeeze('columns')

    cm = confusion_matrix(y, mod.predict(X))
    etiquetas = {0: 'Pagado (0)', 1: 'Default (1)'}
    matriz_data = [
        (etiquetas[i], etiquetas[j], int(cm[i, j]))
        for i in (0, 1) for j in (0, 1)
    ]
    matriz_df = pd.DataFrame(matriz_data, columns=['clase_real', 'clase_predicha', 'conteo'])
    matriz_df.to_csv(POWERBI_DIR / 'matriz_confusion.csv', index=False, encoding='utf-8')

    # 3. Importancia de variables
    print("- Generando importancia_variables.csv")
    imp = pd.read_csv(BASE_DIR / 'results' / 'importancia_variables.csv')
    imp.columns = ['variable', 'importancia']
    imp.round(4).to_csv(POWERBI_DIR / 'importancia_variables.csv', index=False, encoding='utf-8')

    print("¡Listo! Archivos CSV regenerados en la carpeta /powerbi")

if __name__ == '__main__':
    main()