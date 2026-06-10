# Proyecto LACDA — Pipeline DataOps de Aprobación de Préstamos

Pipeline DataOps sobre el **Loan Approval Classification Dataset** (45.000 solicitudes, 14 variables): convierte el dato crudo en un dataset limpio, validado y trazable en PostgreSQL, sobre el que se entrena un modelo **Random Forest** de predicción de default (**ROC-AUC 0.975** en holdout).

> Proyecto de ITY1101 — Gestión de Datos para IA (DUOC UC). Cubre la capa de datos (EP2) y la fase de modelado y evaluación del sistema (Experiencia 3 / RA3). Próximos pasos: API REST de scoring y despliegue.

**📓 Documentación completa del proyecto:**

* [`notebooks/informe.ipynb`](notebooks/informe.ipynb) — el pipeline etapa por etapa: decisiones técnicas, KPIs de monitoreo y evidencias de ejecución (EP2).
* [`notebooks/estudio_features.ipynb`](notebooks/estudio_features.ipynb) — auditoría de la transformación y selección de variables predictoras para Random Forest.
* [`notebooks/evaluacion.ipynb`](notebooks/evaluacion.ipynb) — evaluación RA3: resultados del modelo, rendimiento, auditoría de seguridad, integración, fallas detectadas y mejoras propuestas.

## El pipeline — un script por etapa

```
data/loan_data.csv (fuente)
   │
   ├─ 1. ingesta.py         define el CONTRATO de datos + verifica estructura → staging
   ├─ 2. qualitycheck.py    KPI de calidad 0-100 con alertas (monitoreo, no rompe el build)
   ├─ 3. limpieza.py        aplica el contrato: duplicados, nulos, rangos, dominios
   ├─ 4. transformacion.py  3 features derivadas
   └─ 5. validacion.py      gate estructural + semántico → si TODO pasa, CARGA a PostgreSQL
```

Decisiones de diseño clave:

* **La base de datos es el destino, no el área de trabajo.** Las etapas intermedias intercambian CSVs en `data/`; solo el dataset validado llega a la **tabla única** `loan_data`.
* **El contrato de datos vive en la puerta de entrada** (`ingesta.py`): la limpieza lo aplica y la validación lo audita importando las mismas constantes — una sola fuente de verdad.
* **La carga vive dentro de la validación**: cargar la BD es la consecuencia directa de pasar el gate. Si una regla falla → `exit 1` y la BD no se toca.

## Ejecución

Requiere Docker Desktop.

```bash
docker compose build
docker compose up
```

Eso levanta PostgreSQL, espera su healthcheck y corre las 5 etapas encadenadas; si una falla, las siguientes no se ejecutan. Los datos quedan persistidos en el volumen `pg_data`.

```bash
# Una etapa puntual
docker compose run --rm app python scripts/limpieza.py

# Consultar el resultado
docker compose exec db psql -U lacda -d loans -c "SELECT COUNT(*) FROM loan_data;"
```

Credenciales: se leen de `.env` (ver `.env.example`); sin él se usan defaults de desarrollo.

## Modelo de predicción de default (Random Forest)

Con el pipeline corrido y la BD arriba (`pip install scikit-learn matplotlib seaborn`):

```bash
python scripts/train_model.py    # lee la tabla loan_data, entrena y guarda models/ + holdout
python scripts/test_model.py     # evalúa sobre el holdout: results/ + gate AUC >= 0.93
```

Resultados en holdout (9.000 filas): **ROC-AUC 0.975 · F1 0.832 · recall 0.85**. El set de 12 predictores está justificado en `notebooks/estudio_features.ipynb`.

## Dashboard de integración

```bash
pip install streamlit
streamlit run dashboard.py
```

Una vista que integra: estado de la tabla `loan_data` (consulta en vivo a Postgres), KPI de calidad del dato crudo con alertas, y métricas/gráficos del modelo.

## Estructura del proyecto

```
Proyecto-LACDA/
├── data/                    Fuente (loan_data.csv, Metadata.txt) y salidas del pipeline
├── db/init.sql              Esquema PostgreSQL (tabla única loan_data)
├── scripts/
│   ├── ingesta.py           1. contrato de datos + staging
│   ├── qualitycheck.py      2. KPI de calidad con alertas
│   ├── limpieza.py          3. limpieza según contrato
│   ├── transformacion.py    4. features derivadas
│   ├── validacion.py        5. gate de calidad + carga a PostgreSQL
│   ├── train_model.py       Entrenamiento Random Forest (lee loan_data)
│   └── test_model.py        Evaluación en holdout + gate AUC
├── notebooks/
│   ├── informe.ipynb        Documentación técnica del pipeline (EP2)
│   ├── estudio_features.ipynb  Selección de predictores para el modelo
│   └── evaluacion.ipynb     Evaluación RA3 (rendimiento/seguridad/integración)
├── dashboard.py             Dashboard Streamlit de integración
├── models/ · results/       Salidas del modelo (regeneradas, no versionadas)
├── docs/                    Material de la asignatura y evidencias de ejecución
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml CI: pipeline + modelo en cada push/PR
```

## Stack

Python 3.12 · pandas · SQLAlchemy · PostgreSQL 16 · Docker Compose · GitHub Actions · scikit-learn · Streamlit

## CI/CD

GitHub Actions ejecuta en cada push/PR a `main`, contra un PostgreSQL de servicio: las 5 etapas del pipeline **y** el entrenamiento + evaluación del modelo, con **doble gate de calidad** — datos (`validacion.py`) y modelo (`test_model.py`, AUC ≥ 0.93). CSVs procesados y resultados del modelo se publican como artifacts.

## Equipo — Grupo 4

| Integrante | Rol |
|---|---|
| Nicolás Fernández Vera | Data Engineer |
| Bastián Gutiérrez | Data Analyst |
| Víctor Gutiérrez | ML Engineer |
