# Sistema de Clasificación de Aprobación de Préstamos

Pipeline DataOps para procesamiento y validación de datos sobre el dataset **Loan Approval Classification Dataset** (45.000 solicitudes bancarias, 14 variables).

El objetivo del proyecto es preparar datos limpios, validados y trazables para futuros modelos de IA orientados a predicción de default crediticio.

> Esta entrega cubre únicamente la capa de gestión y procesamiento de datos.  
> Entrenamiento de modelos, API REST y despliegue quedan como próximos pasos.

---

# Pipeline

## 1. Ingesta
Carga del CSV crudo hacia PostgreSQL usando pandas + SQLAlchemy.

Genera:
- `solicitantes_raw`
- `prestamos_raw`

Características:
- Separación por entidades
- Idempotencia con TRUNCATE
- Trazabilidad mediante `fecha_carga`

---

## 2. Limpieza
Aplicación de reglas de negocio y validación de dominios.

Incluye:
- Eliminación de nulos
- Validación de rangos
- Validación semántica
- Filtrado de inconsistencias

Genera:
- `solicitantes_clean`
- `prestamos_clean`
- `loan_data_clean.csv`

---

## 3. Transformación
Creación de features derivadas para futuras etapas de Machine Learning.

Features:
- `rate_x_pct_income`
- `loan_burden`
- `has_prev_defaults`

Genera:
- `solicitantes_transformed`
- `prestamos_transformed`
- `loan_data_transformed.csv`

---

## 4. Validación
Módulo de control de calidad automatizado integrado con CI/CD.

Verifica:
- Nulos
- Duplicados
- Rangos numéricos
- Dominios categóricos
- Coherencia semántica
- Integridad de features derivadas

Si una validación falla:
```bash
exit 1

El workflow de GitHub Actions marca el pipeline como fallido automáticamente.

Ejecución

Requiere:

Docker Desktop
Ejecutar pipeline completo
docker compose build
docker compose up

El pipeline ejecuta automáticamente:

ingesta → limpieza → transformación → validación

Los datos quedan persistidos en el volumen:

pg_data
Estructura del proyecto
Proyecto-LACDA/
├── data/                       Dataset y salidas CSV
├── db/init.sql                 Esquema PostgreSQL
├── scripts/
│   ├── ingesta.py
│   ├── limpieza.py
│   ├── transformacion.py
│   ├── qualitycheck.py
│   └── validacion.py
├── docs/
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
Stack Tecnológico
Python 3.12
pandas
SQLAlchemy
PostgreSQL 16
Docker Compose
GitHub Actions
CI/CD

El proyecto utiliza GitHub Actions para:

ejecutar el pipeline automáticamente,
validar calidad de datos,
detectar errores antes de integrar cambios a main.
Equipo — Grupo 4
Nicolás Fernández Vera

Data Engineer
Procesamiento y limpieza

Bastián Gutiérrez

Data Analyst
Modelado y entrenamiento

Víctor Gutiérrez

ML Engineer
Documentación y arquitectura
