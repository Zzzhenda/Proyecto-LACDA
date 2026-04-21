import pandas as pd
import os

def cargar_datos(ruta_archivo):
    print(f"Iniciando ingesta de datos desde: {ruta_archivo}...")
    
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encontró el archivo en la ruta {ruta_archivo}")
        print("Asegúrate de que el archivo CSV esté dentro de la carpeta 'data/'.")
        return None

    try:
        df = pd.read_csv(ruta_archivo)
        
        print("\nIngesta exitosa.")
        print("-" * 30)
        print(f"Total de filas: {df.shape[0]}")
        print(f"Total de columnas: {df.shape[1]}")
        
        print("\n--- Primeras 5 filas ---")
        print(df.head())
        
        print("\n--- Información de las columnas ---")
        print(df.info())
        
        return df
        
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        return None

if __name__ == "__main__":
    ruta_csv = os.path.join("data", "loan_data.csv")
    
    dataset = cargar_datos(ruta_csv)