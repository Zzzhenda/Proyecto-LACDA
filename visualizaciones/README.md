# Visualización de Resultados — Proyecto LACDA

Capa de **visualización** del proyecto: comunica los resultados del pipeline DataOps y del modelo de
predicción de _default_ a quienes toman decisiones. Todo vive en un solo cuaderno,
[`visualizaciones.ipynb`](visualizaciones.ipynb), construido con **Matplotlib** siguiendo las
metodologías de **data-ink ratio** y **simplicidad visual**.

## Cómo ejecutarlo

El cuaderno lee las salidas reales del pipeline, así que primero deben existir. Desde la raíz del
repositorio:

```bash
# 1. Generar los datos y el modelo (si aún no están)
python scripts/ingesta.py
python scripts/limpieza.py
python scripts/transformacion.py
python scripts/train_model.py     # cae al CSV si la BD no está arriba
python scripts/test_model.py      # produce results/metricas.json

# 2. Abrir y ejecutar el cuaderno
jupyter notebook visualizaciones/visualizaciones.ipynb
```

> Si Docker está arriba (`docker compose up -d db`) el cuaderno lee el dataset desde PostgreSQL
> (`loan_data`); si no, usa automáticamente `data/loan_data_transformed.csv`. No hay que cambiar nada.

## Conexión automática (sin alimentación manual)

Ningún dato se escribe a mano: cada gráfico se alimenta de una salida real del pipeline o del modelo.

| Fuente real | Contenido | Generada por |
|---|---|---|
| `loan_data` (PostgreSQL) → *fallback* `data/loan_data_transformed.csv` | Dataset final validado | `validacion.py` |
| `data/loan_data_raw / _clean / _transformed.csv` | Conteos por etapa (flujo del pipeline) | etapas 1–4 |
| `models/modelo_default.pkl` + `data/X_test.csv` / `y_test.csv` | Modelo entrenado y holdout | `train_model.py` |
| `results/metricas.json` | Accuracy, Precision, Recall, F1, ROC-AUC | `test_model.py` |

Reejecutar el cuaderno actualiza figuras y tablas sin intervención humana.

## Qué produce (carpeta `output/`, regenerable)

| Requisito | Gráfico | Archivo |
|---|---|---|
| Matriz de confusión | Heatmap (verde acierto / rojo error) | `04_matriz_confusion.png` |
| Métricas de calidad (Accuracy/Recall/F1/Tiempo) | Barras + KPIs | `01_kpis.png`, `03_metricas_calidad.png` |
| Distribución de clases | Barras (real vs. predicho) | `05_distribucion_clases.png` |
| Evolución del rendimiento | Líneas (curva de aprendizaje + histórico) | `06a_curva_aprendizaje.png`, `06b_historico_rendimiento.png` |
| Casos erróneos | Tabla filtrable (CSV + filtro interactivo) | `casos_erroneos.csv` |
| Volumen por etapa | Barras (embudo) | `02_volumen_etapas.png`, `volumen_pipeline.csv` |

`output/` está en `.gitignore` (reproducible); el cuaderno conserva sus gráficos embebidos para
visualización directa en GitHub/Jupyter.
