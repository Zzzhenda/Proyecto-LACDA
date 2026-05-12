# Guía del equipo — Proyecto LACDA

Este documento es para que los tres integrantes del Grupo 4 dominemos el proyecto end-to-end. La idea es que cualquiera pueda explicar, modificar y defender cualquier parte sin depender de quien la escribió.

Si solo necesitas correr el proyecto, lee el `README.md`. Esta guía profundiza en el *por qué* de cada decisión.

---

## 1. ¿Qué hace este sistema?

El sistema toma un CSV con 45.000 solicitudes de préstamo y produce un dataset depurado listo para entrenar un modelo de IA que prediga si la solicitud terminará en *default* (no pago) o pago exitoso.

Esta entrega cubre los **dos primeros módulos** del diseño técnico:

- **Ingesta** — carga del CSV crudo a PostgreSQL.
- **Procesamiento** — limpieza (aplica reglas de validación) + transformación (feature engineering).

Faltan: entrenamiento del modelo, predicción y la API REST con FastAPI. Estos quedan para entregas posteriores.

---

## 2. Mapa mental del proyecto (lo más importante)

```
┌─────────────────────┐
│ data/loan_data.csv  │   ← Punto de partida (45.000 filas)
└──────────┬──────────┘
           │
           ▼
   ┌──────────────┐
   │  ingesta.py  │   carga el CSV tal cual a la tabla loans_raw
   └──────┬───────┘
          ▼
┌──────────────────────┐
│  PostgreSQL          │
│    loans_raw         │   ← Espejo del CSV (45.000 filas)
└──────────┬───────────┘
           │
           ▼
   ┌───────────────┐
   │  limpieza.py  │   aplica las reglas del cap. 9 del diseño técnico
   └───────┬───────┘
           │
           ├──────────────► data/loan_data_clean.csv  (44.993 filas)
           ▼
┌──────────────────────┐
│  PostgreSQL          │
│    loans_clean       │   ← Mismo dataset depurado
└──────────┬───────────┘
           │
           ▼
   ┌──────────────────┐
   │ transformacion.py│   agrega features derivadas (fico_band, age_group, rate_x_pct_income)
   └────────┬─────────┘
            │
            ├─────────────► data/loan_data_transformed.csv
            ▼
┌──────────────────────┐
│  PostgreSQL          │
│  loans_transformed   │   ← loans_clean + 3 columnas nuevas
└──────────┬───────────┘
           │
           ▼
   ┌─────────────────┐
   │  auditoria.py   │   valida loans_clean Y loans_transformed contra el diseño
   └─────────────────┘
           │
           ▼
   ✅ OK (continúa)   /   ❌ Falla (CI rojo, hay que investigar)
```

**Regla mental clave:** los datos siempre fluyen en una dirección, sin saltos. Cada etapa toma como entrada lo que produjo la anterior. Si algo falla, no se sigue adelante.

---

## 3. Anatomía del repositorio

```
Proyecto-LACDA/
├── README.md                      ← guía operativa (cómo correrlo)
├── requirements.txt               ← dependencias Python
├── Dockerfile                     ← cómo construir el contenedor de la app
├── docker-compose.yml             ← define los servicios (db + app)
├── .env.example                   ← plantilla de variables de entorno
│
├── .github/
│   └── workflows/
│       └── ci.yml                 ← GitHub Actions (CI)
│
├── data/
│   ├── loan_data.csv              ← ENTRADA: dataset crudo original
│   ├── loan_data_clean.csv        ← SALIDA: dataset depurado
│   └── Metadata.txt               ← descripción de las columnas
│
├── db/
│   └── init.sql                   ← crea las tablas loans_raw y loans_clean
│
├── docs/
│   ├── diseño_tecnico.md          ← documento de diseño (FUENTE DE VERDAD)
│   ├── diseño_tecnico_v1_obsoleto.pdf
│   └── guia_equipo.md             ← este archivo
│
└── scripts/
    ├── db.py                      ← helper de conexión a Postgres
    ├── ingesta.py                 ← módulo 1
    ├── limpieza.py                ← módulo 2a
    ├── transformacion.py          ← módulo 2b (feature engineering)
    └── auditoria.py               ← quality check
```

### ¿Por qué cada archivo existe?

| Archivo | Razón de ser |
|---|---|
| `requirements.txt` | Lista exacta de versiones de librerías Python. Garantiza que en cualquier máquina se instalen las mismas versiones. Sin esto, lo que funciona en un PC puede romperse en otro. |
| `Dockerfile` | Receta para construir la imagen del contenedor que ejecuta los scripts. Define Python 3.12, copia el código e instala dependencias. |
| `docker-compose.yml` | Orquesta los dos contenedores: `db` (Postgres) y `app` (Python). Define la red interna, el volumen persistente y el orden de arranque (la app espera a que la DB esté lista). |
| `.env.example` | Plantilla de credenciales. La idea es que cada integrante copie a `.env` y ponga sus propios valores. En este proyecto los valores son fijos porque es académico, pero la convención queda. |
| `.github/workflows/ci.yml` | Workflow de GitHub Actions que corre la pipeline en cada push. Si alguien rompe algo, GitHub avisa. |
| `db/init.sql` | Postgres ejecuta este archivo automáticamente la PRIMERA vez que arranca con un volumen vacío. Crea las dos tablas. Si modificas este archivo, hay que borrar el volumen (`docker compose down -v`) para que se aplique. |
| `data/loan_data.csv` | El dataset original. **No se modifica nunca.** |
| `data/loan_data_clean.csv` | Generado por `limpieza.py`. Se sobreescribe en cada corrida. |
| `data/Metadata.txt` | Descripción humana de las columnas, alineada al dataset real. |
| `docs/diseño_tecnico.md` | El documento de arquitectura. Es la **fuente de verdad** sobre las reglas de negocio (cap. 9). Si cambia una regla, primero se cambia acá, después en el código. |
| `scripts/db.py` | Helper compartido para crear la conexión a Postgres. Existe para que los otros tres scripts no repitan las mismas 6 líneas de boilerplate. |
| `scripts/ingesta.py` | Módulo 1. CSV → Postgres. |
| `scripts/limpieza.py` | Módulo 2a. Postgres → Postgres + CSV depurado. |
| `scripts/transformacion.py` | Módulo 2b. Lee `loans_clean`, agrega features derivadas y produce `loans_transformed` + CSV. |
| `scripts/auditoria.py` | Quality check. Valida `loans_clean` y `loans_transformed` contra las reglas del diseño. |

---

## 4. Detalle técnico de cada script

### 4.1 `scripts/db.py`

**Qué hace:** crea un objeto `engine` de SQLAlchemy que sabe conectarse a Postgres.

**Cómo funciona:**

```python
def get_engine() -> Engine:
    user = os.getenv("DB_USER", "lacda")
    pwd = os.getenv("DB_PASSWORD", "lacda_pass")
    host = os.getenv("DB_HOST", "db")          # "db" cuando corre en docker
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "loans")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url, connect_args={"client_encoding": "utf8"})
```

**Detalles importantes:**

- Lee credenciales de variables de entorno con *defaults*. Esto permite:
  - Correr en Docker (donde `DB_HOST=db` por la red interna).
  - Correr en GitHub Actions (donde `DB_HOST=localhost` porque el servicio Postgres expone el puerto).
  - Correr local sin tocar nada (los defaults bastan si el contenedor `db` está expuesto en localhost:5432).
- El `client_encoding=utf8` evita errores cuando el sistema operativo tiene locale en español (sin esto, mensajes de error con ñ o tildes pueden romper la conexión).

**Si te preguntan en defensa:** "Es un patrón estándar para no duplicar código de conexión. Centralizar la configuración facilita cambiar la BD sin tocar tres archivos."

### 4.2 `scripts/ingesta.py`

**Qué hace:** lee el CSV crudo y lo carga en `loans_raw`.

**Pasos exactos:**

1. Verifica que `data/loan_data.csv` exista. Si no, falla.
2. Lee el CSV en un DataFrame de pandas.
3. Vacía `loans_raw` con `TRUNCATE TABLE loans_raw RESTART IDENTITY`. Esto borra todas las filas y reinicia el contador del `id`.
4. Inserta el DataFrame en `loans_raw` con `to_sql(..., if_exists="append", chunksize=1000)`. Inserta en lotes de 1000 filas para no saturar la conexión.
5. Imprime cuántas filas quedaron en la tabla para confirmar.

**Por qué TRUNCATE en vez de DROP/CREATE:** el TRUNCATE conserva el esquema (las columnas y tipos definidos en `init.sql`) y solo borra datos. Es más rápido y respeta el contrato del esquema.

**Por qué es idempotente:** correr el script dos veces produce exactamente el mismo resultado que correrlo una vez. Útil para CI y para volver a procesar el dataset sin acumular duplicados.

### 4.3 `scripts/limpieza.py`

**Qué hace:** aplica las reglas del cap. 9 del diseño técnico al contenido de `loans_raw` y produce `loans_clean` + el CSV depurado.

**Estructura del script:**

- `CAT_DOMAINS` — diccionario que define los valores válidos de cada columna categórica. Es la representación en código del diccionario de datos.
- `aplicar_reglas(df)` — función pura que recibe un DataFrame y devuelve el DataFrame depurado + estadísticas. **Pura** significa que no toca BD ni archivos; solo transforma datos. Esto la hace fácil de testear (de hecho, así la validamos durante el desarrollo).
- `main()` — orquesta: lee de Postgres, llama a `aplicar_reglas`, escribe a Postgres y exporta CSV.

**Orden de las reglas dentro de `aplicar_reglas`:**

1. **Duplicados** — `drop_duplicates`. Filas idénticas en todas las columnas se eliminan.
2. **Nulos** — `dropna`. Cualquier celda nula descarta la fila completa.
3. **Rangos numéricos** — 8 reglas (edad, experiencia, ingreso, score, monto, tasa, ratio, historial).
4. **Dominios categóricos** — 5 columnas categóricas + `loan_status`.

**Por qué este orden:** primero quitamos lo barato (duplicados, nulos), después validamos lo específico. Si una fila tiene un nulo en `person_age`, no tiene sentido evaluar la regla de rango sobre ese campo.

**Sobre `person_emp_exp`:** la regla "experiencia ≤ edad − 18" parece simple pero es la que captura el outlier famoso del dataset (filas con experiencia de 125 años). Es una regla de **consistencia entre columnas**, no de rango simple.

**Output:**

- Tabla `loans_clean` en Postgres (44.993 filas en condiciones normales).
- Archivo `data/loan_data_clean.csv` con el mismo contenido.
- Por consola: estadísticas de cuántas filas se removieron en cada etapa.

### 4.4 `scripts/transformacion.py`

**Qué hace:** lee `loans_clean`, agrega tres features derivadas y produce `loans_transformed` + el CSV correspondiente.

**Features agregadas:**

- `fico_band` — entero 1–5 según las bandas FICO oficiales (Poor/Fair/Good/Very Good/Exceptional). Útil porque el modelo puede aprender comportamientos no lineales por banda.
- `age_group` — entero 1–3 (Joven/Adulto/Senior). Lo mismo: discretiza una variable continua en grupos con sentido de negocio.
- `rate_x_pct_income` — float, producto de `loan_int_rate × loan_percent_income`. Es una **interacción**: captura el riesgo combinado de tener tasa alta sobre alta proporción del ingreso. Un modelo lineal no puede aprender interacciones por sí solo, por eso conviene crearlas a mano.

**Por qué son determinísticas:** las tres se calculan con fórmulas fijas, sin mirar la distribución del dataset. Esto significa que se pueden computar antes del *train/test split* sin riesgo de *data leakage*. Si en cambio escaláramos `person_income` (StandardScaler) antes del split, el scaler estaría usando estadísticas del test set → sesgo.

**Lo que NO hace este script:**

- **No codifica categóricas** (no convierte `male/female` a 0/1, ni hace one-hot de `loan_intent`). Eso se hace en el `Pipeline` de sklearn dentro del módulo de modelado, ajustándose solo con datos de entrenamiento.
- **No escala numéricas** (no aplica StandardScaler/MinMaxScaler). Misma razón.

**Cómo se defiende:** "La transformación que produce un CSV físico se limita a features determinísticas. El encoding y el scaling se manejan dentro del pipeline de entrenamiento para evitar contaminación entre train y test."

### 4.5 `scripts/auditoria.py`

**Qué hace:** verifica que `loans_clean` y `loans_transformed` cumplan TODAS las reglas del diseño. Si encuentra cualquier violación, falla con `sys.exit(1)`.

**Por qué existe si `limpieza.py` ya las aplica:** son cosas distintas.

- `limpieza.py` *transforma* (saca filas malas).
- `auditoria.py` *verifica* (confirma que las filas que quedaron son buenas).

Esta separación es importante porque:

- Si alguien modifica las reglas de `limpieza.py` y se le olvida una, la auditoría lo detecta.
- En CI, la auditoría es la red de seguridad: si pasa, sabemos que el CSV es válido.
- Permite hacer auditorías independientes (por ejemplo, sobre un dataset que NO pasó por nuestra limpieza).

**Importante:** `auditoria.py` reusa `CAT_DOMAINS` desde `limpieza.py` (`from limpieza import CAT_DOMAINS`). Así, si agregamos una categoría nueva, basta modificar un solo lugar.

---

## 5. Reglas de limpieza — explicación

Las reglas vienen del diccionario de datos (`docs/diseño_tecnico.md`, cap. 9). Cada una tiene una razón de negocio:

| Regla | Por qué |
|---|---|
| `person_age` ∈ [18, 100] | Menos de 18 no puede pedir crédito legalmente; más de 100 es claramente un error de captura. |
| `person_emp_exp` ≤ `person_age` − 18 | Una persona no puede tener más años de experiencia que años trabajables. Captura outliers de captura (125 años). |
| `person_income` ≥ 0 | Ingresos negativos no existen. |
| `credit_score` ∈ [300, 850] | Es el rango oficial del FICO score. |
| `loan_amnt` > 0 | Un préstamo de 0 no tiene sentido. |
| `loan_int_rate` ∈ [5, 30] | Rango razonable de tasas anuales. Tasas <5% en este contexto no son realistas; >30% serían usurarias. |
| `loan_percent_income` ∈ [0, 1] | Es una proporción. Si fuera >1, el préstamo sería mayor al ingreso anual completo. |
| `cb_person_cred_hist_length` ∈ [0, `person_age`] | No se puede tener más años de historial crediticio que años de vida. |
| Categóricos en su dominio | Los modelos de ML no toleran categorías nuevas en producción. Limitamos a las conocidas. |

**En la corrida real con el dataset actual,** solo se eliminan **7 filas** (todas con `person_age` > 100). El resto del dataset ya viene relativamente limpio.

---

## 6. Decisiones técnicas que tomamos (y por qué)

### 6.1 Modelo físico plano (no normalizado)

El diseño técnico (cap. 8) muestra un modelo lógico con 4 entidades (solicitantes, historial_crediticio, préstamos, resultado). En la práctica usamos **una sola tabla** plana (`loans_raw` y su versión depurada `loans_clean`).

**Por qué:**
- Para una pipeline de ML, una tabla denormalizada es más eficiente (un `SELECT *` y listo, sin joins).
- La columna `id` ya da trazabilidad por fila.
- Normalizar tiene valor en sistemas transaccionales con muchas escrituras concurrentes; no en datasets analíticos.

**Cómo se defiende:** "El modelo lógico de la documentación describe la estructura conceptual del dominio. El modelo físico se optimiza para el caso de uso concreto, que es analítica/ML."

### 6.2 `loans_raw` y `loans_clean` separados (no `UPDATE` sobre la misma)

Tener dos tablas en vez de una sola que se va modificando.

**Por qué:**
- **Trazabilidad:** siempre podemos comparar antes/después y saber qué eliminamos.
- **Reprocesamiento:** si descubrimos un error en las reglas, podemos volver a correr la limpieza sin tener que volver a leer el CSV.
- **Auditoría:** son las dos fuentes que la auditoría puede contrastar.

### 6.3 Idempotencia (TRUNCATE antes de cargar)

Cada script vacía la tabla destino antes de insertar.

**Por qué:** garantiza que correr la pipeline N veces produce el mismo resultado que correrla 1 vez. Sin esto, cada corrida acumularía filas y romperíamos el conteo.

### 6.4 La auditoría sale con código distinto de cero si falla

`sys.exit(1)` en vez de solo imprimir el error.

**Por qué:** en CI, GitHub Actions interpreta el código de salida. Si es 0, todo OK. Si es ≠0, falla el job y el commit queda en rojo. Es la forma estándar de comunicar éxito/fracaso a sistemas externos.

### 6.5 Helper `db.py` compartido

En vez de copiar el código de conexión en cada script.

**Por qué:** principio DRY (*Don't Repeat Yourself*). Si mañana cambia el motor (Postgres → MySQL) o agregamos un parámetro de conexión (timeout, ssl), se modifica un solo archivo.

### 6.6 Variables de entorno con *defaults*

`os.getenv("DB_HOST", "db")` en vez de hardcodear `"db"`.

**Por qué:** permite que el mismo código corra en distintos ambientes (Docker, local, CI) sin modificarlo. El Dockerfile/Compose pasa las variables, el código las lee.

---

## 7. Cómo correr (cheatsheet)

### Localmente con Docker (recomendado)

```bash
docker compose build      # construir la imagen (solo la primera vez o si cambian deps)
docker compose up         # correr la pipeline completa
docker compose down       # apagar todo
docker compose down -v    # apagar Y borrar el volumen (necesario si cambias init.sql)
```

### Una etapa puntual

```bash
docker compose run --rm app python scripts/limpieza.py
```

### Inspeccionar la base de datos

```bash
docker compose exec db psql -U lacda -d loans -c "SELECT COUNT(*) FROM loans_clean;"
docker compose exec db psql -U lacda -d loans -c "SELECT * FROM loans_clean LIMIT 5;"
```

### Sin Docker (si tienes Python y Postgres locales)

```bash
pip install -r requirements.txt
psql -U lacda -d loans -f db/init.sql
DB_HOST=localhost python scripts/ingesta.py
DB_HOST=localhost python scripts/limpieza.py
DB_HOST=localhost python scripts/auditoria.py
```

### CI (GitHub Actions)

Se dispara solo en cada push a `main` o pull request. Ver el badge en el README. Si está rojo, abre la pestaña "Actions" del repo y mira qué falló.

---

## 8. Preguntas frecuentes

**P: Si modifico `init.sql`, ¿basta con `docker compose up`?**
R: No. Postgres solo ejecuta `init.sql` la PRIMERA vez que arranca con volumen vacío. Tienes que `docker compose down -v` (con `-v` para borrar el volumen) y después `docker compose up`.

**P: ¿Qué pasa si el CSV tiene una nueva categoría que no está en `CAT_DOMAINS`?**
R: La limpieza la elimina (saca esas filas). Si es una categoría legítima, hay que agregarla a `CAT_DOMAINS` en `limpieza.py` *y* a la sección 9 del diseño técnico (regla: la documentación primero).

**P: ¿Por qué hay un PDF marcado como obsoleto?**
R: La versión 1 del diseño técnico tenía errores en el diccionario de datos (faltaba `HOMEIMPROVEMENT`, tipos incorrectos en algunas columnas). La versión vigente es `diseño_tecnico.md`. El PDF v1 se mantiene como historial.

**P: ¿Por qué no usamos sklearn / pipelines de transformación en esta entrega?**
R: Porque el alcance es solo ingesta + limpieza. Los pipelines de scikit-learn aplican en la fase de feature engineering / modelado, que es la próxima entrega.

**P: ¿Qué pasa si la auditoría falla?**
R: La pipeline se corta. En local lo ves en consola. En CI, el job aparece en rojo. Hay que investigar la causa: ¿cambió el CSV?, ¿hay un bug en `limpieza.py`?, ¿agregaron una regla nueva en el diseño que no se implementó?

**P: ¿Cómo agrego una regla nueva?**
R:
1. Agrégala al cap. 9 de `docs/diseño_tecnico.md`.
2. Implementala en `aplicar_reglas` (`limpieza.py`).
3. Agrégala como check en `auditar` (`auditoria.py`).
4. Corre la pipeline para validar.
5. Commit y push — el CI confirma.

---

## 9. Reparto de roles del equipo (sugerencia)

Aunque todos debemos dominar todo, para defender en presentación:

| Integrante | Foco | Archivos a dominar a fondo |
|---|---|---|
| Nicolás Fernández Vera | Procesamiento y limpieza | `limpieza.py`, `auditoria.py`, cap. 9 del diseño |
| Bastián Gutiérrez | Modelado y entrenamiento | `db.py`, `ingesta.py`, `init.sql`, decisiones de modelado |
| Víctor Gutiérrez | Documentación y arquitectura | `diseño_tecnico.md`, `Dockerfile`, `docker-compose.yml`, `ci.yml`, `README.md` |

**Pero todos deben poder responder cualquier pregunta sobre cualquier archivo.** Esta guía existe justamente para eso.

---

## 10. Checklist antes de entregar

- [ ] `docker compose up` corre la pipeline completa sin errores.
- [ ] La auditoría imprime "OK".
- [ ] El archivo `data/loan_data_clean.csv` existe y tiene ~44.993 filas.
- [ ] El badge de CI en el README está en verde.
- [ ] El `<usuario>/<repo>` del badge está reemplazado por el real.
- [ ] El PDF del diseño técnico v2 está regenerado (exportar `diseño_tecnico.md` desde un visor markdown).
- [ ] El repo no tiene archivos basura (`__pycache__`, `.DS_Store`, etc — el `.gitignore` ya los excluye, pero conviene revisar).
- [ ] Los tres integrantes leyeron esta guía y pueden defender el proyecto.

---

## 11. Próximos pasos (para futuras entregas)

1. **Encoding de categóricas y scaling** — dentro de un `sklearn.Pipeline` para evitar leakage.
2. **Entrenamiento del modelo** — validación cruzada, métricas (AUC, F1, recall en clase positiva).
3. **API REST con FastAPI** — endpoint `/predict` que recibe una solicitud y devuelve probabilidad de default.
4. **Containerizar la API** — agregar un servicio `api` al `docker-compose.yml`.
5. **Tests automatizados** — pytest con casos sintéticos para `aplicar_reglas` y `aplicar_transformaciones`.

Cuando llegue el momento, el `ci.yml` se extiende para correr los tests y validar el modelo.
