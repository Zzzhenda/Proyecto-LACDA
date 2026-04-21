import pandas as pd
import os

def limpiar_datos(ruta_entrada, ruta_salida):
    print(f"Iniciando limpieza de datos desde: {ruta_entrada}...")
    
    try:
        # Cargar los datos
        df = pd.read_csv(ruta_entrada)
        filas_iniciales = df.shape[0]
        
        # 1. Eliminar duplicados
        duplicados = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"Se encontraron y eliminaron {duplicados} filas duplicadas.")
        

        # edades mayores a 100 años y experiencias laborales mayores a 70 años
        filas_antes_outliers = df.shape[0]
        df = df[(df['person_age'] <= 100) & (df['person_emp_exp'] <= 70)]
        outliers = filas_antes_outliers - df.shape[0]
        print(f"Se eliminaron {outliers} filas por valores atípicos (ej. edad > 100 o experiencia > 70).")
        
        df.to_csv(ruta_salida, index=False)
        
        print("-" * 30)
        print("Limpieza completada con éxito.")
        print(f"Filas originales: {filas_iniciales}")
        print(f"Filas finales (limpias): {df.shape[0]}")
        print(f"Archivo guardado en: {ruta_salida}")
        
    except Exception as e:
        print(f"Ocurrió un error durante la limpieza: {e}")

if __name__ == "__main__":
    ruta_raw = os.path.join("data", "loan_data.csv")
    ruta_clean = os.path.join("data", "loan_data_limpio.csv")
    
    if not os.path.exists(ruta_raw):
        print(f"Error: No se encontró el archivo {ruta_raw}. Asegúrate de tener el dataset original.")
    else:
        limpiar_datos(ruta_raw, ruta_clean)