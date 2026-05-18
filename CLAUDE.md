# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto del proyecto

Proyecto académico (DuocUC, **ITY1101 *Gestión de Datos para IA***, Grupo 4). Evaluación Parcial 2 (EP2): informe + presentación + preguntas individuales (70% de la nota). Objetivo final: predecir default de préstamos sobre el *Loan Approval Classification Dataset* (45.000 filas, 14 columnas). **Esta entrega cubre sólo el pipeline de datos** en 5 etapas:

```
ingesta → qualitycheck → limpieza → transformacion → validacion
```

Entrenamiento del modelo, predicción y API REST quedan fuera de alcance (documentados como próximos pasos en `notebooks/informe.md`).

El README está en español, sin tildes y deliberadamente conciso (el usuario rechazó versiones extensas). La documentación técnica detallada vive en `notebooks/informe.md` y este archivo. Mantener cualquier cambio sincronizado entre los tres.

## Comandos

Todo corre dentro de Docker. Postgres no se ejecuta nativo en Windows.

```bash
docker compose build
docker compose up                          # corre pipeline completo y sale
docker compose down                        # mantiene volumen pg_data
docker compose down -v                     # OBLIGATORIO si tocas db/init.sql

# correr una sola etapa
docker compose run --rm app python scripts/ingesta.py
docker compose run --rm app python scripts/qualitycheck.py
docker compose run --rm app python scripts/limpieza.py
docker compose run --rm app python scripts/transformacion.py
docker compose run --rm app python scripts/validacion.py

# inspeccionar la DB
docker compose exec db psql -U lacda -d loans -c "SELECT COUNT(*) FROM solicitantes_clean;"
```

No hay test runner (pytest) ni linter configurado. Las funciones de `limpieza.py` y `transformacion.py` son testeables sin DB si se mockean los IO de Postgres. Los notebooks de `notebooks/` se pueden ejecutar localmente sin Postgres (leen directo de `data/loan_data.csv`).

**Verificación offline rápida** (sin Docker): `python -m py_compile scripts/*.py` para sintaxis. Para validar lógica end-to-end se puede crear un script temporal que aplique las funciones de validacion contra los CSVs en `data/`.

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

### Dos responsabilidades, dos archivos

La rúbrica EP2 exige dos cosas distintas — están separadas a propósito en dos scripts:

| Archivo | Cuándo corre | Sobre qué | Rol | Exit |
|---|---|---|---|---|
| `qualitycheck.py` | después de ingesta, antes de limpieza | tablas `_raw` | KPI / monitoreo (observabilidad de la fuente) | 0 siempre — informativo |
| `validacion.py` | al final | tablas `_clean` y `_transformed` | gate de salida (reglas duras cap. 9) | 1 si falla |

`qualitycheck.py` no rompe el build porque el dataset crudo se espera que tenga problemas — su valor es diagnóstico y de observabilidad (detectar si la fuente se degrada entre corridas). `validacion.py` sí rompe: es el contrato de calidad de salida.

`validacion.py` importa `CAT_DOMAINS` de `limpieza.py` — acoplamiento intencional pero implica que validación no puede correr en un contenedor sin `limpieza.py` importable.

### Features derivadas (transformacion.py)

Tres features **determinísticas** (no dependen de la distribución → no introducen data leakage), **todas a nivel prestamo**, las tres con respaldo EDA en `notebooks/features.ipynb`:

- `rate_x_pct_income` = `loan_int_rate × loan_percent_income` — riesgo combinado tasa/ingreso (|corr| 0.46)
- `loan_burden` = `loan_amnt × (1 + loan_int_rate/100) / person_income` — costo total / ingreso (|corr| 0.40)
- `has_prev_defaults` = `(previous_loan_defaults_on_file == "Yes").astype(int)` — encoding binario (|corr| 0.54)

**Por qué no hay features a nivel solicitante:** el EDA mostró que `credit_score` (|corr| 0.008) y `person_age` (|corr| 0.02) **no correlacionan** con `loan_status` en este dataset. Las features anteriores (`fico_band`, `age_group`) se quitaron al no poder justificarse por datos. `solicitantes_transformed` mantiene el contrato del schema (raw/clean/transformed por entidad) aunque no agregue columnas.

**Encoding de categóricas y escalado de numéricas se delegan al pipeline de sklearn** en la fase de modelado (no aquí), para que se ajusten sólo con datos de entrenamiento.

### Configuración de DB

`scripts/db.py` lee `DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME`. Defaults coinciden con `docker-compose.yml` (`host=db`). El CI usa `host=localhost` porque corre Postgres como service.

**Nota:** `.env.example` declara `POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB` (nombres del image oficial de Postgres) — esas variables **no son leídas** por `db.py`. Si se introduce `.env`, renombrar a `DB_*` o ajustar `db.py`.

## CI

`.github/workflows/ci.yml` levanta Postgres efímero, instala deps, aplica `db/init.sql`, y corre las **5 etapas** en orden (`ingesta → qualitycheck → limpieza → transformacion → validacion`). Sube los CSVs depurados como artefacto. Solo `validacion.py` rompe el build (exit 1); `qualitycheck.py` es informativo (exit 0 siempre).

## Notebooks (`notebooks/`)

| Archivo | Rol |
|---|---|
| `eda_solicitantes.ipynb` | EDA de la entidad Solicitante — markdown mínimo, foco en datos. Confirma que `credit_score` y `person_age` NO predicen default. |
| `eda_prestamos.ipynb` | EDA de la entidad Préstamo — markdown mínimo. Muestra que `loan_percent_income`, `loan_int_rate` y `loan_intent` son los predictores. |
| `features.ipynb` | Justificación detallada de las 3 features con explicaciones en lenguaje natural. Incluye matriz de correlación para verificar no-redundancia. |
| `informe.md` | Informe técnico alineado con la pauta de evaluación EP2 (9 indicadores). Tiene placeholders `[completar]` para sección, fecha y URL de repo. |

Los notebooks corren sin Postgres (leen `data/loan_data.csv` directamente). El usuario tenía 4 notebooks y los redujo a 3 + el informe — no agregar más sin pedirlo.

## Reglas del entorno y de colaboración con el usuario

- **Idioma:** español, **sin tildes** en archivos del proyecto (README, scripts, informe, notebooks). En conversación las tildes están bien.
- **Estilo de docs:** el usuario prefiere documentos cortos y precisos sobre extensos y detallados. El README pasó por dos rondas de simplificación drástica.
- **Defensa académica:** el usuario está preparando la defensa oral del EP2. Cuando pregunte conceptos, responder con la narrativa que usaría frente al docente (no solo la respuesta técnica). El docente puede preguntar por: metodología PMBOK aplicada, justificación de cada feature, estrategias de manejo de anomalías, escalabilidad.
- **EDA-driven decisions:** las features originales (`fico_band`, `age_group`) fueron eliminadas tras descubrir vía EDA que `credit_score` (|corr| 0.008) y `person_age` (|corr| 0.02) no correlacionan con el target. Reemplazadas por `has_prev_defaults`, `loan_burden`, `rate_x_pct_income`. Si surgen ideas de nuevas features, validar con EDA antes de agregar al pipeline.
- **Cambios al schema:** después de tocar `db/init.sql`, el usuario tiene que hacer `docker compose down -v` antes del próximo `up`. Recordárselo cuando aplique.

## Estado actual al cierre de la última sesión

- Pipeline funcional, 5 etapas, sintaxis OK en todos los scripts.
- Schema actualizado con las features nuevas en `prestamos_transformed` (`loan_burden NUMERIC(10,4)`, `has_prev_defaults SMALLINT`).
- `solicitantes_transformed` no tiene features extra — mantiene contrato del schema pero EDA no justificó agregar nada.
- `notebooks/informe.md` listo con todos los apartados de la pauta EP2; faltan datos de portada.
- Docker no se pudo correr en la sesión pasada (daemon no respondía); el pipeline se validó offline contra los CSVs en `data/`. Confirmar con el usuario si necesita probarlo en Docker antes de entregar.
