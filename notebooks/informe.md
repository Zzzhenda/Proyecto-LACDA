# Informe Tecnico - Evaluacion Parcial N°2

**Asignatura:** ITY1101 - Gestion de Datos para IA
**Seccion:** [completar]
**Proyecto:** Sistema de Clasificacion de Aprobacion de Prestamos
**Equipo - Grupo 4:**
- Nicolas Fernandez Vera - Procesamiento y limpieza
- Bastian Gutierrez - Modelado y entrenamiento
- Victor Gutierrez - Documentacion y arquitectura

**Fecha:** [completar]
**Repositorio:** [completar URL de GitHub]

---

## Indice

1. Resumen ejecutivo
2. Justificacion de la metodologia PMBOK
3. Planificacion del proyecto
4. Explicacion tecnica del pipeline
   1. Ingesta
   2. Quality check (KPI baseline)
   3. Limpieza
   4. Transformacion
   5. Validacion
   6. Carga
5. Documentacion del codigo y evidencias
6. Estrategia de KPI de monitoreo
7. Conclusiones y proximos pasos

---

## 1. Resumen ejecutivo

### 1.1 Problema

Las instituciones financieras enfrentan un problema critico al evaluar solicitudes de prestamo: decidir, en tiempo limitado, si un cliente pagara o entrara en default. La evaluacion manual es lenta, costosa y subjetiva; la evaluacion algoritmica requiere datos limpios, consistentes y bien estructurados. La calidad de los datos determina la calidad del modelo predictivo: **basura entra, basura sale**.

### 1.2 Solucion

Construimos un **pipeline de datos DataOps** que toma el *Loan Approval Classification Dataset* (45.000 solicitudes, 14 variables) y lo deja listo para entrenar un clasificador binario de default. El pipeline opera en 5 etapas automatizadas, dockerizadas, ejecutadas en cada cambio del repositorio via GitHub Actions:

```
ingesta -> quality check -> limpieza -> transformacion -> validacion
```

### 1.3 Valor para la organizacion

| Antes (sin pipeline) | Despues (con pipeline) |
|---|---|
| CSVs procesados a mano, scripts ad-hoc | Pipeline reproducible en un `docker compose up` |
| Calidad de datos no medida | KPI `quality_score` cuantificable por etapa |
| Sin trazabilidad de transformaciones | 6 tablas (raw, clean, transformed × 2 entidades) con historial completo |
| Decisiones de feature engineering por intuicion | Features derivadas justificadas por EDA con correlacion medida |
| Errores en datos detectados en produccion | Gate de validacion automatica rompe el build si las reglas fallan |

**Alcance de esta entrega:** ingesta, limpieza, transformacion, validacion y monitoreo. El entrenamiento del modelo y la API REST estan documentados como proximos pasos.

---

## 2. Justificacion de la metodologia PMBOK

Adoptamos un **enfoque hibrido (mixto)** de PMBOK porque el proyecto combina dos naturalezas distintas:

| Componente | Naturaleza | Por que |
|---|---|---|
| Arquitectura del pipeline, schema de DB, infraestructura Docker, CI | **Predictiva** | Estos artefactos requieren planificacion al inicio. Cambiarlos a mitad del proyecto fuerza recrear todo el volumen de datos. |
| Reglas de limpieza, features derivadas, KPIs de calidad | **Adaptativa** | Estas decisiones dependen de lo que el EDA descubra. No se pueden planificar de antemano sin ver los datos. |

**Evidencia de iteracion adaptativa real:** las features `fico_band` y `age_group` fueron incluidas en una version inicial del pipeline. Tras correr un EDA sistematico (`notebooks/features.ipynb`), descubrimos que `credit_score` y `person_age` tienen correlacion |corr| < 0.03 con el target en este dataset, por lo que las descartamos y las reemplazamos por `loan_burden` (|corr| 0.40) y `has_prev_defaults` (|corr| 0.54), justificadas por evidencia. Esto refleja un ciclo PDCA (Plan-Do-Check-Act) tipico del enfoque adaptativo.

**Por que no PMBOK puramente predictiva:** un plan fijo no permitiria descartar features cuando el EDA contradice las hipotesis iniciales.

**Por que no PMBOK puramente adaptativa:** sin una arquitectura predefinida (schema, contenedores, CI), cada iteracion empezaria de cero.

---

## 3. Planificacion del proyecto

### 3.1 Tecnologia de seguimiento

Utilizamos **GitHub Issues + Projects** como tablero de tareas, integrado al repositorio donde vive el codigo. Razones:

- Cero costo, cero configuracion adicional (el repo ya esta en GitHub).
- Cada issue se linkea a commits y pull requests automaticamente, dando trazabilidad bidireccional.
- El CI (GitHub Actions) corre sobre cada PR, integrando seguimiento y validacion en la misma plataforma.

Alternativas evaluadas: Trello (mas visual pero requiere sincronizar a mano con el repo), Jira (potente pero overkill para 3 personas), Azure DevOps (idem).

### 3.2 Work Breakdown Structure (WBS)

```
1. Setup
   1.1 Configurar repositorio GitHub
   1.2 Definir Dockerfile y docker-compose
   1.3 Configurar CI con GitHub Actions

2. Diseño de datos
   2.1 Modelo logico (2 entidades: Solicitante, Prestamo)
   2.2 Modelo fisico (6 tablas: raw, clean, transformed × 2)
   2.3 Schema SQL (db/init.sql)

3. Pipeline - implementacion
   3.1 Ingesta (scripts/ingesta.py)
   3.2 Quality check / KPI baseline (scripts/qualitycheck.py)
   3.3 Limpieza (scripts/limpieza.py + Winsorizer)
   3.4 Transformacion / feature engineering (scripts/transformacion.py)
   3.5 Validacion / gate (scripts/validacion.py)

4. Analisis exploratorio (EDA)
   4.1 EDA general (notebooks/eda.ipynb)
   4.2 EDA por entidad (notebooks/eda_solicitantes.ipynb, eda_prestamos.ipynb)
   4.3 Busqueda sistematica de features (notebooks/features.ipynb)

5. Documentacion y defensa
   5.1 README.md tecnico
   5.2 CLAUDE.md para asistentes
   5.3 Informe (este documento)
   5.4 Preparacion de la presentacion
```

### 3.3 Carta Gantt (3 semanas)

```
                          Sem 1      Sem 2      Sem 3
1. Setup                  [###]
2. Diseño de datos        [#####]
3. Pipeline                  [#########]
4. EDA                          [######]
5. Documentacion                   [######]
```

---

## 4. Explicacion tecnica del pipeline

El pipeline DataOps consta de 5 etapas encadenadas en el `Dockerfile`. Cada etapa es un script Python independiente, con responsabilidad unica y comunicacion via PostgreSQL.

```
CMD ["sh", "-c", "python scripts/ingesta.py
                  && python scripts/qualitycheck.py
                  && python scripts/limpieza.py
                  && python scripts/transformacion.py
                  && python scripts/validacion.py"]
```

El operador `&&` garantiza **propagacion de errores**: si una etapa falla, las siguientes no se ejecutan.

### 4.1 Ingesta (`scripts/ingesta.py`)

**Objetivo:** cargar `data/loan_data.csv` a PostgreSQL separando las dos entidades del modelo logico.

**Herramientas:**
- `pandas.read_csv` para parsear el CSV.
- `SQLAlchemy` para conexion a Postgres.
- `pandas.DataFrame.to_sql` con `chunksize=1000` para insercion por lotes.

**Decision tecnica:** la ingesta es **idempotente**. Antes de insertar, hace `TRUNCATE TABLE ... RESTART IDENTITY CASCADE` en orden hijo→padre. Asi, ejecutarla dos veces no duplica datos.

**Salida:** tablas `solicitantes_raw` (45.000 filas) y `prestamos_raw` (45.000 filas, FK a solicitantes_raw).

### 4.2 Quality check (`scripts/qualitycheck.py`)

**Objetivo:** medir la calidad de los datos crudos antes de tocarlos. Sistema de **observabilidad de la fuente**.

**Herramientas:**
- Clase `QualityCheck` adaptada de `PROFESORA/quality_check.py`.
- Calcula 4 dimensiones (nulos, duplicados, outliers IQR, inconsistencias) y un `quality_score` ponderado 0-100.

**Decision tecnica:** esta etapa **no rompe el build** (siempre retorna exit 0). El raw se espera que tenga problemas — el proposito es **observar**, no fallar. Si en el futuro el `raw_score` cae entre corridas, eso indica que la fuente de datos se degrado y dispara alerta.

**Salida (stdout):**
```
[qualitycheck] [ ] solicitantes_raw  score= 80.00/100  nivel=OK
[qualitycheck] [!] prestamos_raw     score= 60.00/100  nivel=WARNING
```

### 4.3 Limpieza (`scripts/limpieza.py`)

**Objetivo:** depurar los datos aplicando las reglas del cap. 9 del diseño tecnico.

**Herramientas:**
- pandas para imputacion y clipping.
- Clase `Winsorizer` adaptada de `PROFESORA/winsorizer.py`.

**Orden estricto de 7 pasos:**

1. Eliminar duplicados exactos (`drop_duplicates`).
2. Imputar nulos (mediana para numericas, moda para categoricas).
3. Aplicar reglas duras por columna (clip por rango cerrado del cap. 9).
4. Aplicar reglas cruzadas (`emp_exp ≤ age-18`, `cred_hist ≤ age`), usando la edad **ya fijada** en el paso 3.
5. Aplicar dominios categoricos (reemplazar invalidos por moda).
6. Aplicar Winsorizacion al 5% **solo en** `person_income` y `loan_amnt`.

**Por que Winsorizar solo 2 columnas:** son las unicas que **no tienen rango cerrado** definido en el cap. 9. Las demas (`credit_score`, `loan_int_rate`, `loan_percent_income`, `person_age`) tienen contrato duro — Winsorizarlas destruiria informacion legitima dentro del rango valido.

**Por que el orden importa:** si Winsorizamos `person_age` antes del paso 4, la regla `emp_exp ≤ age-18` se rompe (se detecto en una version anterior y se documento como invariante).

**Salida:** tablas `solicitantes_clean`, `prestamos_clean` + CSVs.

### 4.4 Transformacion (`scripts/transformacion.py`)

**Objetivo:** feature engineering. Crear features derivadas que aumenten el poder predictivo del modelo.

**Decisiones clave (3 features, todas a nivel prestamo):**

| Feature | Formula | \|corr\| | Dominio |
|---|---|---|---|
| `rate_x_pct_income` | `loan_int_rate × loan_percent_income` | 0.46 | Interaccion tasa/ingreso |
| `loan_burden` | `loan_amnt × (1 + loan_int_rate/100) / person_income` | 0.40 | Costo total / ingreso |
| `has_prev_defaults` | `(previous_loan_defaults_on_file == "Yes").astype(int)` | 0.54 | Historial crediticio |

**Justificacion EDA:** ver `notebooks/features.ipynb`. Se probaron **14 features candidatas** de 4 familias (ratios financieros, transformaciones log, encoding de categoricas, interacciones). Las 3 elegidas cubren 3 dominios distintos y su matriz de correlacion mutua confirma que **no son redundantes entre si**.

**Lo que NO esta aqui (por diseño):** encoding one-hot de categoricas y escalado de numericas. Eso se delega al `sklearn.Pipeline` en la fase de modelado, para que se ajuste **solo con datos de entrenamiento** y no introduzca data leakage del set de validacion.

**Salida:** tablas `solicitantes_transformed` (sin features extra), `prestamos_transformed` (con las 3 features) + `data/loan_data_transformed.csv` (join plano para ML).

### 4.5 Validacion (`scripts/validacion.py`)

**Objetivo:** gate de salida del pipeline. Verifica que las tablas `_clean` y `_transformed` cumplan **todas las reglas del cap. 9**. Si algo falla, **rompe el build con exit 1**.

**Por que validar despues de limpiar si `limpieza.py` ya aplico las reglas:**
- **Defensa en profundidad:** `limpieza.py` cree que aplico las reglas. `validacion.py` lo verifica con codigo distinto. Si el verificador usa la misma logica que el limpiador, no esta verificando — esta confiando.
- **Catch de regresiones:** un cambio futuro en `limpieza.py` que rompa una regla queda atrapado por el CI antes de llegar a produccion.
- **Contrato auditable:** la validacion es el contrato de calidad de salida. Otros equipos (ML, BI) consumen los datos sabiendo que pasaron por este gate.

**Checks ejecutados:**
- Rangos: `person_age ∈ [18, 100]`, `credit_score ∈ [300, 850]`, `loan_int_rate ∈ [5, 30]`, `loan_percent_income ∈ [0, 1]`.
- Negativos: `person_income ≥ 0`, `loan_amnt > 0`.
- Consistencia cruzada: `person_emp_exp ∈ [0, age-18]`, `cb_person_cred_hist_length ∈ [0, age]`.
- Dominios categoricos: `person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`, `loan_status`.
- Features derivadas: `rate_x_pct_income ≥ 0`, `loan_burden ≥ 0`, `has_prev_defaults ∈ {0, 1}`.

### 4.6 Carga

La "carga" en este pipeline es bidireccional:

1. **Carga interna a PostgreSQL:** cada etapa escribe sus resultados a las tablas correspondientes (`_raw`, `_clean`, `_transformed`).
2. **Carga al consumidor (ML):** `transformacion.py` exporta `data/loan_data_transformed.csv` — la vista plana lista para `pandas.read_csv` + `sklearn.Pipeline`. Es el "contrato de entrega" entre el equipo de datos y el equipo de modelado.

Las tablas quedan persistidas en el volumen Docker `pg_data`, accesibles para queries ad-hoc:

```bash
docker compose exec db psql -U lacda -d loans -c "SELECT COUNT(*) FROM solicitantes_clean;"
```

---

## 5. Documentacion del codigo y evidencias

### 5.1 Repositorio

- **Codigo:** [link al repo GitHub]
- **Dockerfile:** raiz del repo
- **CI:** `.github/workflows/ci.yml` — corre el pipeline completo en cada push/PR a main, sube CSVs depurados como artefactos.

### 5.2 Estructura

```
Proyecto-LACDA/
├── data/                       CSV crudo + salidas del pipeline
├── db/init.sql                 6 tablas (raw, clean, transformed × 2 entidades)
├── docs/diseño_tecnico.pdf     diseño tecnico (cap. 8-9: modelo logico y reglas)
├── scripts/
│   ├── ingesta.py              etapa 1
│   ├── qualitycheck.py         etapa 2 (KPI baseline)
│   ├── limpieza.py             etapa 3 (con clase Winsorizer)
│   ├── transformacion.py       etapa 4 (feature engineering)
│   └── validacion.py           etapa 5 (gate de salida)
├── notebooks/
│   ├── eda.ipynb               EDA general
│   ├── eda_solicitantes.ipynb  EDA por entidad
│   ├── eda_prestamos.ipynb     EDA por entidad
│   ├── features.ipynb          busqueda sistematica de features
│   └── informe.md              este documento
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

### 5.3 Logs esperados de una corrida exitosa

```
[ingesta]        Filas en solicitantes_raw: 45000
[ingesta]        Filas en prestamos_raw:    45000
[qualitycheck]   solicitantes_raw  score= 80.00/100  nivel=OK
[qualitycheck]   prestamos_raw     score= 60.00/100  nivel=WARNING
[limpieza]       Duplicados removidos: 0
[limpieza]       Reglas de rango aplicadas (cap. 9)
[limpieza]       Reglas cruzadas (emp_exp, cred_hist_length) aplicadas
[limpieza]       Winsorizacion 5% aplicada sobre: ['person_income', 'loan_amnt']
[transformacion] Features agregadas: rate_x_pct_income, loan_burden, has_prev_defaults
[validacion]     solicitantes_clean OK
[validacion]     prestamos_clean OK
[validacion]     solicitantes_transformed OK
[validacion]     prestamos_transformed OK
[validacion]     Resumen: 45000 prestamos | 10000 defaults / 35000 pagados
```

---

## 6. Estrategia de KPI de monitoreo (calidad de datos)

### 6.1 KPI principal: `quality_score`

Score ponderado 0-100 calculado sobre cada tabla por la clase `QualityCheck` (adaptada del material docente).

| Dimension | Peso | Como se mide |
|---|---|---|
| Nulos / faltantes | 0.30 | `df.isnull().any()` |
| Duplicados | 0.20 | `df.duplicated().any()` |
| Outliers | 0.20 | Regla IQR 1.5x por columna numerica |
| Inconsistencias | 0.30 | Negativos en columnas no excluidas + categoricas mal normalizadas |

Formula: `score = (1 − Σ pesos_dimensiones_con_fallo) × 100`

### 6.2 Umbrales de alerta

| Nivel | Rango | Accion |
|---|---|---|
| OK | score ≥ 70 | Sin alerta |
| WARNING | 50 ≤ score < 70 | Loggear; revisar manualmente |
| CRITICAL | score < 50 | Alerta automatica; el pipeline puede romper |

### 6.3 Donde se aplica

- **`qualitycheck.py`** mide `_raw` (antes de limpiar) → observabilidad de la fuente.
- **`validacion.py`** valida `_clean` y `_transformed` con reglas duras binarias (cap. 9) → gate de salida.

La separacion es deliberada: el KPI continuo monitorea, el gate binario decide.

### 6.4 Estado actual del proyecto (medido)

| Tabla | Score | Nivel |
|---|---|---|
| `solicitantes_raw` | 80/100 | OK |
| `prestamos_raw` | 60/100 | WARNING (por duplicados de fila completa + outliers IQR) |
| `_clean`, `_transformed` | n/a en KPI | Auditoria binaria: **todas las reglas pasan** |

---

## 7. Conclusiones y proximos pasos

### 7.1 Lo logrado en esta entrega

- Pipeline de datos completo, dockerizado, con CI automatizado.
- Modelo fisico normalizado (2 entidades × 3 etapas = 6 tablas) con trazabilidad por fila.
- 3 features derivadas justificadas por EDA con respaldo cuantitativo.
- Sistema de monitoreo con KPI ponderado y umbrales de alerta.
- Documentacion tecnica (README, CLAUDE.md, 4 notebooks, este informe).

### 7.2 Proximos pasos (fuera de esta entrega)

| Paso | Descripcion | Riesgo / consideracion |
|---|---|---|
| Entrenamiento del modelo | `sklearn.Pipeline` con encoding + escalado + clasificador (RandomForest / XGBoost) | Dataset desbalanceado (22% positivos) — usar `class_weight='balanced'` o SMOTE |
| Validacion del modelo | Cross-validation estratificada, metricas precision/recall/F1/AUC-ROC | Accuracy no sirve por el desbalance |
| API REST | FastAPI servir el modelo entrenado en endpoint `/predict` | Sanitizacion de input segun las reglas del cap. 9 |
| Dashboard de monitoreo | Grafana o similar para visualizar `quality_score` historico | Requiere persistir los scores en una tabla `kpi_history` |
| Escalabilidad | Particionar tablas por fecha si el volumen supera 1M filas | El pipeline actual carga todo en memoria; revisar si llega a este punto |

### 7.3 Como manejariamos anomalias en produccion

| Anomalia | Estrategia |
|---|---|
| CSV con encoding distinto | `qualitycheck.py` deteccion temprana; logs claros del error |
| Postgres caido / timeout | Healthcheck del compose + retry en CI |
| Columna nueva en el CSV fuente | `validacion.py` detecta esquema cambiado, falla el build |
| Datos degradados (raw_score cae) | Alerta WARNING/CRITICAL desde `qualitycheck.py`; investigar fuente |
| Bug en `limpieza.py` que rompe regla | `validacion.py` la pilla (defensa en profundidad), CI rojo |

---

## Anexo: como ejecutar el proyecto

```bash
# Levantar el pipeline completo
docker compose build
docker compose up

# Inspeccionar las tablas
docker compose exec db psql -U lacda -d loans -c "SELECT COUNT(*) FROM solicitantes_clean;"

# Correr una etapa puntual
docker compose run --rm app python scripts/qualitycheck.py

# Apagar
docker compose down       # mantiene los datos
docker compose down -v    # borra el volumen pg_data (obligatorio si cambia init.sql)
```

**Importante:** si se modifica `db/init.sql`, hay que correr `docker compose down -v` antes del siguiente `up`. PostgreSQL solo ejecuta `init.sql` si el volumen esta vacio — si quedan datos viejos, el schema viejo persiste y las etapas posteriores fallan al intentar usar columnas nuevas que no existen en el schema antiguo.
