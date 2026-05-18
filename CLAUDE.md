# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto del proyecto

Proyecto académico (DuocUC, *Gestión de Datos para IA*). Objetivo final: predecir default de préstamos sobre el *Loan Approval Classification Dataset* (45.000 filas, 14 columnas). **Esta entrega cubre sólo el pipeline de datos** (ingesta → limpieza → transformación → validación). Entrenamiento, predicción y API REST quedan fuera de alcance.

El README está en español y es la referencia primaria de cara al docente — mantener cualquier cambio sincronizado con él.

## Comandos

Todo corre dentro de Docker. Postgres no se ejecuta nativo en Windows.

```bash
docker compose build
docker compose up                          # corre pipeline completo y sale
docker compose down                        # mantiene volumen pg_data
docker compose down -v                     # OBLIGATORIO si tocas db/init.sql

# correr una sola etapa
docker compose run --rm app python scripts/ingesta.py
docker compose run --rm app python scripts/limpieza.py
docker compose run --rm app python scripts/transformacion.py
docker compose run --rm app python scripts/validacion.py

# inspeccionar la DB
docker compose exec db psql -U lacda -d loans -c "SELECT COUNT(*) FROM solicitantes_clean;"
```

No hay test runner (pytest) ni linter configurado. Las funciones de `limpieza.py` y `transformacion.py` son testeables sin DB si se mockean los IO de Postgres.

## Arquitectura

### Modelo físico: 2 entidades × 3 etapas = 6 tablas

El diseño técnico (cap. 8) normaliza el dataset crudo en dos entidades — **solicitante** y **préstamo** — unidas por FK. Esa separación se mantiene en cada etapa del pipeline:

```
solicitantes_raw   ── FK ──>  prestamos_raw
solicitantes_clean ── FK ──>  prestamos_clean
solicitantes_transformed ── FK ──>  prestamos_transformed
```

Todas las tablas están en `db/init.sql`. Cada `scripts/*.py` que escribe hace `TRUNCATE … RESTART IDENTITY CASCADE` en orden hijo→padre antes de insertar (idempotencia). Cuando ML necesita una vista plana, se hace `JOIN` (ver `extraer_datos_raw` en `limpieza.py` y la query inicial de `transformacion.py`).

**Invariante crítico:** las filas se insertan **ordenadas por `id`** en la entidad padre y luego se asignan FKs por posición (`df_pre.insert(0, "solicitante_id", ids["id"].values)`). Cualquier reordenamiento intermedio rompe la integridad referencial sin que el schema lo note.

### Orden interno de `limpieza.py` (no reordenar)

```
1. extraer_datos_raw       # JOIN raw
2. remover_duplicados       # única operación que reduce filas
3. imputar_nulos            # mediana/moda; nunca elimina filas
4. aplicar_reglas_rango     # clip cap. 9 — fija person_age FINAL
5. aplicar_reglas_cruzadas  # emp_exp ≤ age-18, cred_hist ≤ age  (depende del paso 4)
6. aplicar_dominios_categoricos
7. aplicar_winsorizacion    # SOLO person_income, loan_amnt — al FINAL
```

Winsorización se aplica al final, **nunca** sobre `person_age`, `credit_score`, `loan_int_rate`, `loan_percent_income` (tienen rango duro). Si se winsoriza `person_age` antes del paso 5, las reglas cruzadas se rompen (bug detectado en versión anterior).

### `validacion.py` = etapa del pipeline + sistema de monitoreo

El archivo fusiona **dos responsabilidades** distintas, ambas exigidas por la rúbrica EP2:

| Bloque | Salida | Exit code |
|---|---|---|
| `auditar_*()` — reglas duras cap. 9 | binario (pasa / falla) | 1 si falla |
| `QualityCheck` — KPI ponderado | score 0-100 + nivel OK/WARNING/CRITICAL | 2 si CRITICAL |

Los WARNING (60/100 en este dataset por duplicados semánticos + IQR outliers) son **esperados y no rompen el build**. Sólo CRITICAL lo hace. Si se necesita separar etapa-de-pipeline de monitoreo, dividir en `validacion.py` (exit 1) y `monitoreo.py` (exit 2).

`validacion.py` importa `CAT_DOMAINS` de `limpieza.py` — acoplamiento intencional pero implica que validación no puede correr en un contenedor sin `limpieza.py` importable.

### Features derivadas (transformacion.py)

Sólo tres, **determinísticas** (no dependen de la distribución → no introducen data leakage):

- `fico_band` (1-5): bandas FICO oficiales sobre `credit_score`
- `age_group` (1-3): joven / adulto / senior
- `rate_x_pct_income`: `loan_int_rate × loan_percent_income`

**Encoding de categóricas y escalado de numéricas se delegan al pipeline de sklearn** en la fase de modelado (no aquí), para que se ajusten sólo con datos de entrenamiento.

### Configuración de DB

`scripts/db.py` lee `DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME`. Defaults coinciden con `docker-compose.yml` (`host=db`). El CI usa `host=localhost` porque corre Postgres como service.

**Nota:** `.env.example` declara `POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB` (nombres del image oficial de Postgres) — esas variables **no son leídas** por `db.py`. Si se introduce `.env`, renombrar a `DB_*` o ajustar `db.py`.

## CI

`.github/workflows/ci.yml` levanta Postgres efímero, instala deps, aplica `db/init.sql`, y corre las 4 etapas en orden. Sube los CSVs depurados como artefacto. Cualquier fallo en `validacion.py` (exit 1 o 2) marca el commit en rojo.

## Discrepancias README ↔ repo (pendientes de arreglar)

- README línea 4 tiene placeholder `<usuario>/<repo>` para el badge de CI.
- README líneas 26, 49-50 referencian `docs/estudio.md` y `docs/preguntas_defensa.md` — **no existen**.
- README líneas 51-62 referencian carpeta `PROFESORA/` en la raíz — en el repo real está en `docs/Material gestion datos para ia/`.
