import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def transformar_datos(ruta_entrada, ruta_salida):
    print(f"Iniciando transformacion de datos desde: {ruta_entrada}...")
    
    try:
        # 1. Cargar datos limpios
        df = pd.read_csv(ruta_entrada)
        
        # 2. Identificar columnas categoricas (texto)
        columnas_texto = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Columnas a convertir: {columnas_texto}")
        
        # 3. Aplicar Label Encoding
        le = LabelEncoder()
        for col in columnas_texto:
            df[col] = le.fit_transform(df[col])
            
        # 4. Escalar variables numericas (Normalizacion)
        scaler = StandardScaler()
        columnas_numericas = df.drop(columns=['loan_status']).columns
        df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])
        
        # 5. Guardar el dataset transformado
        df.to_csv(ruta_salida, index=False)
        
        print("-" * 30)
        print("Transformacion completada con exito.")
        print(f"Archivo guardado en: {ruta_salida}")
        
    except Exception as e:
        print(f"Ocurrio un error durante la transformacion: {e}")

if __name__ == "__main__":
    ruta_clean = os.path.join("data", "loan_data_limpio.csv")
    ruta_final = os.path.join("data", "loan_data_final.csv")
    
    if not os.path.exists(ruta_clean):
        print(f"Error: No se encontro {ruta_clean}.")
    else:
        transformar_datos(ruta_clean, ruta_final)