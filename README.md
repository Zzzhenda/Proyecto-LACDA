# Sistema de Clasificación de Aprobación de Préstamos

![CI](https://github.com/<usuario>/<repo>/actions/workflows/ci.yml/badge.svg)

> Reemplaza `<usuario>/<repo>` por el path real del repo en GitHub para que el badge muestre el estado del último build.

Proyecto académico — DuocUC, Ingeniería Informática (IA), asignatura *Gestión de Datos para IA*.

Objetivo final: predecir, mediante un modelo de IA, si una solicitud de préstamo terminará en *default* o pago exitoso, sobre el dataset *Loan Approval Classification Dataset* (45.000 solicitudes, 14 variables).

## Alcance de esta entrega

Se implementan los módulos de **ingesta** y **procesamiento** del diseño técnico (`docs/diseño_tecnico.md`):

1. **Ingesta** — carga del CSV crudo a PostgreSQL (`loans_raw`).
2. **Limpieza** — aplica las reglas del diccionario de datos del diseño y produce:
   - tabla `loans_clean` en PostgreSQL,
   - archivo `data/loan_data_clean.csv`.
3. **Transformación (feature engineering)** — agrega features derivadas (`fico_band`, `age_group`, `rate_x_pct_income`) y produce:
   - tabla `loans_transformed` en PostgreSQL,
   - archivo `data/loan_data_transformed.csv`.
4. **Auditoría / quality check** — verifica que `loans_clean` y `loans_transformed` cumplan todas las reglas. Falla si encuentra cualquier violación.

Las etapas posteriores (entrenamiento, predicción, FastAPI) quedan fuera de esta entrega.

> 📖 Para entender el proyecto a fondo (decisiones técnicas, qué hace cada script, cómo se defiende), ver `docs/guia_equipo.md`.

## Estructura del repositorio

```
Proyecto-LACDA/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml                    (CI: corre la pipeline en cada push)
├── data/
│   ├── loan_data.csv                 (entrada — dataset crudo)
│   ├── loan_data_clean.csv           (salida — generada por la limpieza)
│   ├── loan_data_transformed.csv     (salida — generada por la transformación)
│   └── Metadata.txt
├── db/
│   └── init.sql                      (esquema de loans_raw, loans_clean, loans_transformed)
├── docs/
│   ├── diseño_tecnico.md             (documento vigente, fuente de verdad)
│   ├── guia_equipo.md                (documentación interna del equipo)
│   └── diseño_tecnico_v1_obsoleto.pdf
└── scripts/
    ├── db.py                         (helper de conexión a Postgres)
    ├── ingesta.py                    (módulo 1: CSV → loans_raw)
    ├── limpieza.py                   (módulo 2a: loans_raw → loans_clean + CSV)
    ├── transformacion.py             (módulo 2b: loans_clean → loans_transformed + CSV)
    └── auditoria.py                  (quality check: valida loans_clean y loans_transformed)
```

### ¿Qué hace cada script?

| Archivo | Rol |
|---|---|
| `scripts/db.py` | Helper compartido. Construye el `engine` de SQLAlchemy con credenciales tomadas de variables de entorno. Lo usan los otros scripts para no repetir el código de conexión. |
| `scripts/ingesta.py` | Lee `data/loan_data.csv` y lo carga en la tabla `loans_raw`. Idempotente (TRUNCATE + INSERT). |
| `scripts/limpieza.py` | Lee `loans_raw`, aplica las reglas del diccionario de datos, escribe `loans_clean` y exporta `data/loan_data_clean.csv`. |
| `scripts/transformacion.py` | Lee `loans_clean`, agrega features derivadas (`fico_band`, `age_group`, `rate_x_pct_income`), escribe `loans_transformed` y exporta `data/loan_data_transformed.csv`. |
| `scripts/auditoria.py` | Lee `loans_clean` y `loans_transformed` y verifica TODAS las reglas. Sale con código distinto de 0 si encuentra cualquier violación → así el CI marca el commit en rojo. |

## Cómo ejecutarlo localmente

Requisito único: Docker Desktop corriendo.

```bash
docker compose build
docker compose up
```

El contenedor `app` ejecuta la pipeline completa (`ingesta → limpieza → transformación → auditoría`) y sale al terminar. El contenedor `db` queda corriendo con los datos persistidos en el volumen `pg_data`.

Salida esperada (resumida):

```
[ingesta]        Filas en loans_raw: 45000
[limpieza]       Filas finales: 44993 (de 45000)
[limpieza]       CSV limpio guardado en /app/data/loan_data_clean.csv
[transformacion] Features agregadas: fico_band, age_group, rate_x_pct_income
[transformacion] CSV transformado guardado en /app/data/loan_data_transformed.csv
[auditoria]      loans_clean OK
[auditoria]      loans_transformed OK
```

### Verificar la base de datos

```bash
docker compose exec db psql -U lacda -d loans -c "SELECT COUNT(*) FROM loans_clean;"
```

### Apagar

```bash
docker compose down       # mantiene los datos
docker compose down -v    # borra el volumen (requerido si cambias init.sql)
```

### Correr una etapa puntual

```bash
docker compose run --rm app python scripts/limpieza.py
```

## Integración continua (GitHub Actions)

El workflow `.github/workflows/ci.yml` se dispara en cada `push` y `pull request` a `main`. En cada corrida:

1. Levanta un servicio efímero de PostgreSQL 16.
2. Instala las dependencias del proyecto.
3. Aplica `db/init.sql` para crear las tablas.
4. Ejecuta **ingesta → limpieza → auditoría** en orden.
5. Sube el `loan_data_clean.csv` resultante como artefacto descargable de la corrida.

Si la auditoría detecta cualquier violación de las reglas del diseño, el job falla y el commit queda marcado en rojo en GitHub. Esto garantiza que cualquier cambio futuro al CSV o a las reglas no introduzca data inválida sin que se note.

## Reglas de limpieza aplicadas

Tomadas directamente del diccionario de datos (cap. 9 de `docs/diseño_tecnico.md`):

| Variable | Regla |
|---|---|
| `person_age` | 18 ≤ x ≤ 100 |
| `person_emp_exp` | 0 ≤ x ≤ (person_age − 18) |
| `person_income` | x ≥ 0 |
| `credit_score` | 300 ≤ x ≤ 850 |
| `loan_amnt` | x > 0 |
| `loan_int_rate` | 5 ≤ x ≤ 30 |
| `loan_percent_income` | 0 ≤ x ≤ 1 |
| `cb_person_cred_hist_length` | 0 ≤ x ≤ person_age |
| `person_gender` | {male, female} |
| `person_education` | {High School, Bachelor, Master, Associate, Doctorate} |
| `person_home_ownership` | {RENT, OWN, MORTGAGE, OTHER} |
| `loan_intent` | {PERSONAL, EDUCATION, MEDICAL, VENTURE, DEBTCONSOLIDATION, HOMEIMPROVEMENT} |
| `previous_loan_defaults_on_file` | {Yes, No} |
| `loan_status` | {0, 1} |

Adicionalmente: se eliminan duplicados y filas con nulos.

## Features derivadas (transformación)

Después de la limpieza, el módulo de transformación agrega tres features determinísticas:

| Feature | Tipo | Lógica |
|---|---|---|
| `fico_band` | int 1–5 | Banda FICO oficial (Poor/Fair/Good/Very Good/Exceptional) |
| `age_group` | int 1–3 | Joven [18,29] / Adulto [30,54] / Senior [55,100] |
| `rate_x_pct_income` | float | `loan_int_rate × loan_percent_income` |

El **encoding de categóricas** y el **escalado de numéricas** *no* se aplican aquí: se delegan al pipeline de modelado para evitar *data leakage*.

## Decisión de modelado físico

El diseño técnico (cap. 8) propone un modelo lógico normalizado en cuatro entidades (*solicitantes*, *historial_crediticio*, *préstamos*, *resultado*). En esta entrega el **modelo físico se mantiene plano** (`loans_raw` y `loans_clean` como espejos del CSV con `id` y `fecha_carga` técnicas). Razones:

- Consume directo en la pipeline de ML aguas abajo (un solo `SELECT *`).
- La columna `id` ya garantiza trazabilidad por fila.
- Normalizar agrega *joins* sin valor para esta etapa; el modelo lógico del diseño queda disponible para una iteración posterior.

## Stack

- Python 3.12 · pandas · NumPy · SQLAlchemy · psycopg2
- PostgreSQL 16 (alpine)
- Docker Compose
- GitHub Actions (CI)

## Equipo — Grupo 4

- Nicolás Fernández Vera — Procesamiento y limpieza
- Bastián Gutiérrez — Modelado y entrenamiento
- Víctor Gutiérrez — Documentación y arquitectura
