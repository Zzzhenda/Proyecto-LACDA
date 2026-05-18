# Sistema de Clasificacion de Aprobacion de Prestamos

Predecir si una solicitud de prestamo terminara en default o pago exitoso, sobre el dataset *Loan Approval Classification Dataset* (45.000 solicitudes, 14 variables).

Esta entrega cubre el pipeline de datos. Entrenamiento, prediccion y API REST quedan fuera de alcance.

## Pipeline

1. **Ingesta** - CSV crudo a PostgreSQL (`solicitantes_raw`, `prestamos_raw`).
2. **Quality check** - KPI baseline sobre `_raw` (sistema de monitoreo, informativo).
3. **Limpieza** - reglas duras del cap. 9, imputacion y Winsorizer 5%. Genera tablas `_clean` y CSVs.
4. **Transformacion** - features derivadas (`rate_x_pct_income`, `loan_burden`, `has_prev_defaults`), todas justificadas por EDA (ver `notebooks/features.ipynb`). Genera tablas `_transformed` y `loan_data_transformed.csv`.
5. **Validacion** - gate de salida: reglas duras cap. 9 sobre `_clean` y `_transformed` (exit 1 si fallan).

## Como ejecutarlo

Requiere Docker Desktop.

```bash
docker compose build
docker compose up
```

El contenedor `app` corre el pipeline completo y sale. Los datos quedan persistidos en el volumen `pg_data`.

## Estructura

```
Proyecto-LACDA/
├── data/                       CSV crudo + salidas del pipeline
├── db/init.sql                 6 tablas (raw, clean, transformed x 2 entidades)
├── docs/diseño_tecnico.pdf     fuente de verdad
├── scripts/                    ingesta, qualitycheck, limpieza, transformacion, validacion
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml    CI con Postgres efimero
```

## Stack

Python 3.12, pandas, SQLAlchemy, PostgreSQL 16, Docker Compose, GitHub Actions.

## Equipo - Grupo 4

- Nicolas Fernandez Vera - Procesamiento y limpieza
- Bastian Gutierrez - Modelado y entrenamiento
- Victor Gutierrez - Documentacion y arquitectura
