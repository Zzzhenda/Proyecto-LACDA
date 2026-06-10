# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Pipeline DataOps + modelo de default (ITY1101 Gestión de Datos para IA, DUOC UC) sobre el dataset **Loan Approval Classification** (45.000 solicitudes, `data/loan_data.csv`, versionado en git). La EP2 (capa de datos) y la Experiencia 3/RA3 (modelo + evaluación de rendimiento/seguridad/integración) están implementadas:

- **Modelo**: Random Forest entrenado por `scripts/train_model.py` (lee la tabla `loan_data` de Postgres, con fallback al CSV) y evaluado por `scripts/test_model.py` (gate: exit 1 si ROC-AUC < 0.93). Resultado en holdout: AUC 0.975, F1 0.832. Salidas: `models/` y `results/` (gitignored, regenerables).
- **Decisiones fijadas por evidencia** (`notebooks/estudio_features.ipynb` — no re-litigar sin nueva evidencia): features = set "S3 depurado" de 12 variables (sin `person_emp_exp`, sin el categórico `previous_loan_defaults_on_file`, con `has_prev_defaults`; `rate_x_pct_income` y `loan_burden` FUERA del modelo por multicolinealidad > 0.9 pero SE MANTIENEN en el pipeline de datos). Split 80/20 estratificado `random_state=29`; el holdout solo lo mira `test_model.py`.
- **Evaluación RA3**: `notebooks/evaluacion.ipynb` (rendimiento, auditoría de seguridad, integración, fallas, mejoras). `dashboard.py` (Streamlit) integra BD + KPI + métricas del modelo: `streamlit run dashboard.py`.
- **Seguridad ya aplicada**: credenciales vía `.env` (docker-compose usa `${POSTGRES_*:-default}`), imagen no-root (`USER appuser`) con override `user: root` SOLO en el compose local porque el bind mount de Docker Desktop/Windows no acepta escrituras no-root — no "arreglar" eso quitando el USER del Dockerfile.

Código y comentarios en español; mantener ese idioma. La documentación principal (exigida por la rúbrica, ver `docs/Material gestion datos para ia/ENCARGO.pdf`) es `notebooks/informe.ipynb` — si cambias el pipeline, actualiza también el notebook.

**Preferencia del dueño del repo:** exactamente **5 scripts, uno por etapa** (la estructura que enseñó el profesor). No consolidar en un solo archivo ni agregar scripts auxiliares; los helpers se integran dentro de las etapas.

## Comandos

```bash
# Pipeline completo (requiere Docker Desktop)
docker compose build
docker compose up
# El contenedor app ejecuta: ingesta -> qualitycheck -> limpieza -> transformacion -> validacion (que incluye la carga)

# Una etapa puntual
docker compose run --rm app python scripts/<etapa>.py

# Ejecución local sin contenedor app (Postgres sí en Docker)
docker compose up -d db
pip install -r requirements.txt
$env:DB_HOST="localhost"; python scripts/ingesta.py   # PowerShell; en bash: DB_HOST=localhost python ...
```

- Modelo: `pip install scikit-learn matplotlib seaborn` (NO están en requirements.txt a propósito: la imagen Docker del pipeline se mantiene liviana; el CI los instala en un paso aparte). Luego `python scripts/train_model.py && python scripts/test_model.py` con la BD arriba.
- Los scripts se invocan como `python scripts/<etapa>.py` desde la raíz; los imports entre etapas (`from ingesta import RANGOS`) dependen de que `scripts/` esté en `sys.path` (Python lo agrega por ser el dir del script).
- `get_engine()` vive en `scripts/validacion.py` (único módulo que toca la BD); lee `DB_USER/DB_PASSWORD/DB_HOST/DB_PORT/DB_NAME` con defaults que apuntan a `host=db` (red de docker-compose). Para correr local: `DB_HOST=localhost`.
- No hay tests ni linter. El "test" del proyecto es correr el pipeline completo y que las 5 etapas salgan con código 0 (`validacion.py` es el gate).
- El notebook se ejecuta localmente (no en el contenedor) con la BD arriba: necesita `pandas sqlalchemy psycopg2-binary matplotlib` instalados local.

## Arquitectura

**La BD es el destino del pipeline, no su área de trabajo.** Una sola tabla (`loan_data`, esquema en `db/init.sql`); las etapas intermedias intercambian CSVs en `data/` (`loan_data_raw.csv` → `loan_data_clean.csv` → `loan_data_transformed.csv`, gitignored y regenerados en cada corrida). Solo el dataset validado se carga a Postgres.

Un script por etapa en `scripts/`, en orden de ejecución:

| Etapa | Script | Exit code |
|---|---|---|
| 1 Ingesta | `ingesta.py` — define el **contrato de datos** (COLUMNAS, RANGOS, DOMINIOS), verifica estructura de la fuente, escribe staging | 1 si fuente ausente/estructura mala |
| 2 Monitoreo | `qualitycheck.py` — KPI 0–100 con alertas OK/WARNING/CRITICAL sobre el dato crudo | **siempre 0** (informativo) |
| 3 Limpieza | `limpieza.py` — dedup, imputación, rangos, dominios, winsorización | 1 si falta staging |
| 4 Transformación | `transformacion.py` — 3 features derivadas (define FEATURES_DERIVADAS) | 1 si features con NaN |
| 5 Validación + carga | `validacion.py` — gate estructural+semántico; **si todo pasa**, TRUNCATE+INSERT transaccional a `loan_data` con verificación de conteo | **1 si una regla falla → la BD no se toca, CI rojo** |

Detalles que importan al modificar:

- **El contrato de datos vive en `ingesta.py`** y las demás etapas lo importan (`limpieza` lo aplica, `validacion` lo audita). Cualquier cambio de regla se hace solo ahí — no dupliques constantes de negocio.
- **El orden interno de `limpieza.py` es deliberado**: rangos duros primero, luego reglas cruzadas (`person_emp_exp <= person_age - 18`, `cred_hist <= person_age`) usando el `person_age` ya corregido, winsorización al final y solo sobre columnas sin rango duro (WINSORIZE_COLS, definido en limpieza). Reordenar rompe la consistencia cruzada.
- **La carga es parte de `validacion.py`** a propósito: es la consecuencia de pasar el gate. No separarla en un script aparte.
- **Features derivadas solo a nivel préstamo** (`rate_x_pct_income`, `loan_burden`, `has_prev_defaults`), justificadas por correlación con `loan_status` (~0.46/~0.37/~0.54). Encoding de categóricas y escalado se delegan a la fase futura de ML para evitar data leakage — no agregarlos al pipeline.
- Logging: todos los scripts usan `log = logging.getLogger("<etapa>")` con formato compartido `[%(name)s] ...`; no usar prefijos manuales en el format string (los imports entre etapas harían que gane el basicConfig del módulo importado).
- `db/init.sql` se aplica solo al **crear** el volumen `pg_data`; si cambias el esquema: `docker compose down -v` y volver a levantar.
- Cada etapa es idempotente y re-ejecutable por separado (lee su CSV de entrada; la carga hace TRUNCATE antes de insertar).

## CI

`.github/workflows/ci.yml` corre las 5 etapas en orden contra un service container Postgres 16 en cada push/PR a `main` (con `DB_HOST=localhost`) y sube `loan_data_clean.csv` y `loan_data_transformed.csv` como artifacts. Si el gate de validación sale con 1, el workflow falla y la BD no se toca.
