# Integración con Power BI — Proyecto LACDA

Esta carpeta deja **lista la conexión** entre el pipeline DataOps y **Power BI Desktop**.
La capa de datos del proyecto (PostgreSQL, tabla `loan_data`) es la fuente; Power BI se
conecta a ella como cualquier herramienta BI organizacional. Solo falta que tú instales
Power BI Desktop y armes los visuales arrastrando campos.

## Contenido de la carpeta

| Archivo | Para qué sirve |
|---|---|
| `lacda.pbids` | Archivo de conexión. Al abrirlo, Power BI Desktop se conecta directo a la tabla `loan_data` en Postgres. |
| `metricas_modelo.csv` | Métricas del modelo en formato plano (Accuracy, Precision, Recall, F1, ROC-AUC, Gini) para tarjetas/barras. |
| `matriz_confusion.csv` | Matriz de confusión en formato largo (clase_real, clase_predicha, conteo) para el visual *Matrix*. |
| `importancia_variables.csv` | Importancia de cada variable del Random Forest, para un gráfico de barras. |

> Los 3 CSV se generan a partir de las salidas reales del modelo (`results/`). Si reentrenas
> el modelo y quieres refrescarlos, vuelve a correr el bloque que los crea (ver el final de
> este README) y luego en Power BI: **Inicio → Actualizar**.

## Requisitos previos

1. **Power BI Desktop** instalado (gratis): https://powerbi.microsoft.com/desktop/
   - Las versiones recientes (2022 en adelante) traen el conector **PostgreSQL** integrado.
   - Si Power BI pidiera el proveedor **Npgsql**, instálalo desde
     https://www.npgsql.org/ (paquete *Npgsql* para .NET) y reinicia Power BI Desktop.
2. **La base de datos arriba.** Desde la raíz del repo:
   ```powershell
   docker compose up -d db
   ```
   Si la tabla está vacía, corre el pipeline una vez (`docker compose up`).
   El puerto **5432** queda expuesto a `localhost` (ver `docker-compose.yml`).

## Paso a paso

### 1. Conectar a PostgreSQL

1. **Doble clic en `lacda.pbids`** (o en Power BI: *Archivo → Abrir*).
   Power BI usa los datos de conexión ya definidos: servidor `localhost:5432`, base `loans`.
2. Cuando pida credenciales, elige **Base de datos** e ingresa:
   - **Usuario:** `lacda`
   - **Contraseña:** `lacda_pass`

   > Son las credenciales de desarrollo definidas en `.env.example` / `docker-compose.yml`.
   > Si creaste un `.env` propio, usa esos valores.
3. En el **Navegador**, marca la tabla **`public.loan_data`** y pulsa **Cargar**.

### 2. Sumar los resultados del modelo (CSV)

La tabla `loan_data` cubre la **capa de datos**. Para mostrar también los **resultados del
modelo** (lo que pide la rúbrica), agrega los CSV de esta carpeta:

- *Inicio → Obtener datos → Texto/CSV* → selecciona `metricas_modelo.csv`,
  `matriz_confusion.csv` e `importancia_variables.csv` (uno por uno) → **Cargar**.

### 3. Armar el dashboard (sugerencia de visuales)

| Sección | Visual | Campos |
|---|---|---|
| KPIs del modelo | **Tarjetas** (Card) | `metricas_modelo[valor]` filtrando por `metrica` (una tarjeta por métrica) |
| Métricas | **Barras** | eje `metricas_modelo[metrica]`, valor `metricas_modelo[valor]` |
| Matriz de confusión | **Matrix** | filas `clase_real`, columnas `clase_predicha`, valores `Suma de conteo` |
| Importancia de variables | **Barras horizontales** | eje `variable`, valor `importancia` (orden descendente) |
| Tasa de default | **Tarjeta / Gauge** | medida `DIVIDE(SUM(loan_status), COUNT(id))` sobre `loan_data` |
| Default por intención | **Barras apiladas** | eje `loan_intent`, valor promedio de `loan_status` |
| Riesgo por ingreso | **Columnas** | eje `person_income` (en rangos), valor promedio de `loan_status` |
| Edad / score | **Histograma** | `person_age`, `credit_score` |

Columnas útiles de `loan_data`: `loan_status` (0 pagado / 1 default), `loan_intent`,
`person_income`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `credit_score`,
`person_home_ownership`, `has_prev_defaults`.

## Import vs. DirectQuery

`lacda.pbids` usa **`mode: "Import"`**: Power BI carga una copia de los datos y el reporte
funciona aunque después apagues la BD (ideal para la demo y para exportar el `.pbix`).
Si prefieres consultar la BD en vivo, edita `lacda.pbids` y cambia `"Import"` por
`"DirectQuery"` (requiere la BD encendida en cada apertura).

## Cómo encaja en el informe (apartado h — integración BI)

- **Patrón de integración:** la BD relacional es el punto único de integración. Cualquier
  herramienta BI (Power BI, Metabase, Grafana, Looker) se conecta vía *connection string*
  estándar a `loan_data`, **sin desarrollo adicional** ni acoplar la herramienta al pipeline.
- **Doble vía implementada:** dashboard local en **Streamlit** (`dashboard.py`) para el equipo
  técnico + **Power BI** (esta carpeta) como capa de reporte organizacional.
- **Seguridad:** Power BI se autentica con un rol/usuario propio de la BD; las credenciales
  viven en `.env` (no versionado), no en el `.pbids`. En producción, Power BI se conectaría a
  una instancia gestionada (Azure Database for PostgreSQL) con un usuario de **solo lectura**.

## Regenerar los CSV (tras reentrenar el modelo)

Los CSV salen de `results/`. Tras reentrenar, vuelve a generarlos con este comando
(desde la raíz del repo; no es un script del pipeline, es un export puntual de BI):

```powershell
docker compose up -d db
$env:DB_HOST="localhost"
python scripts/train_model.py ; python scripts/test_model.py
python -c "import json,joblib,pandas as pd; from pathlib import Path; from sklearn.metrics import confusion_matrix; B=Path('.'); O=B/'powerbi'; m=json.loads((B/'results'/'metricas.json').read_text(encoding='utf-8')); g=2*m['roc_auc']-1; pd.DataFrame([('Accuracy',m['accuracy']),('Precision (default)',m['precision']),('Recall (default)',m['recall']),('F1-score (default)',m['f1_score']),('ROC-AUC',m['roc_auc']),('Gini',g)],columns=['metrica','valor']).round(4).to_csv(O/'metricas_modelo.csv',index=False,encoding='utf-8'); mod=joblib.load(B/'models'/'modelo_default.pkl'); X=pd.read_csv(B/'data'/'X_test.csv'); y=pd.read_csv(B/'data'/'y_test.csv').squeeze('columns'); cm=confusion_matrix(y,mod.predict(X)); e={0:'Pagado (0)',1:'Default (1)'}; pd.DataFrame([(e[i],e[j],int(cm[i,j])) for i in (0,1) for j in (0,1)],columns=['clase_real','clase_predicha','conteo']).to_csv(O/'matriz_confusion.csv',index=False,encoding='utf-8'); imp=pd.read_csv(B/'results'/'importancia_variables.csv'); imp.columns=['variable','importancia']; imp.round(4).to_csv(O/'importancia_variables.csv',index=False,encoding='utf-8'); print('CSV de powerbi/ regenerados')"
```
