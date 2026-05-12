# Sistema de Clasificación de Aprobación de Préstamos
### Loan Approval Classification Dataset

**Institución:** DuocUC
**Carrera:** Ingeniería Informática — Especialidad en Inteligencia Artificial
**Asignatura:** Gestión de Datos para IA
**Docente:** Jazna Patricia Meza Hidalgo

**Integrantes — Grupo 4:**
- Nicolás Fernández Vera
- Bastián Gutiérrez
- Víctor Gutiérrez

**Versión:** 2 (corrige el diccionario de datos de la v1 contra el dataset real).
**Fecha:** 2026-05-05.

---

## Tabla de contenidos
1. Resumen técnico del sistema
2. Arquitectura seleccionada
3. Justificación de la arquitectura
4. Especificaciones tecnológicas
5. Estructura modular del sistema
6. Diagrama de arquitectura lógica
7. Diagrama de flujo de datos (DFD)
8. Diagrama lógico de base de datos
9. Diccionario de datos
10. Control de versiones
11. Documentación técnica
12. Ejemplo aplicado al caso
13. Interoperabilidad

---

## 1. Resumen técnico del sistema

El sistema corresponde a una solución de clasificación basada en inteligencia artificial cuyo objetivo es determinar si una solicitud de préstamo terminará en default o pago exitoso, usando el *Loan Approval Classification Dataset*.

Procesa variables como edad, ingresos, historial crediticio, monto del préstamo y tasa de interés para predecir `loan_status`.

La solución permitirá:
- Analizar datos de solicitantes.
- Procesar información financiera.
- Generar predicciones automatizadas.
- Apoyar la toma de decisiones en evaluación crediticia.

## 2. Arquitectura seleccionada

Se selecciona una **arquitectura en capas (Layered Architecture)** con enfoque modular.

Capas:
- **Presentación:** interfaz / consumo del sistema (API REST con FastAPI en una fase posterior).
- **Lógica de negocio:** procesamiento, validación y coordinación de módulos.
- **Datos:** almacenamiento y acceso a datos (PostgreSQL).

## 3. Justificación de la arquitectura

- **Separación de responsabilidades:** cada capa cumple una función específica.
- **Mantenibilidad:** permite modificar partes del sistema sin afectar otras.
- **Escalabilidad:** facilita agregar módulos (nuevos modelos, nuevos validadores).
- **Adaptación a IA:** permite integrar fácilmente modelos de clasificación.

## 4. Especificaciones tecnológicas

| Aspecto | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Librerías de datos | pandas, NumPy |
| Acceso a datos | SQLAlchemy, psycopg2 |
| Framework API (fase posterior) | FastAPI |
| Base de datos | PostgreSQL 16 |
| Contenedores | Docker, Docker Compose |
| Herramientas | GitHub, Visual Studio Code, Jupyter Notebook |

## 5. Estructura modular del sistema

| Módulo | Estado en esta entrega | Responsabilidad |
|---|---|---|
| Ingesta de datos | Implementado | Carga del CSV crudo a `loans_raw` |
| Procesamiento — limpieza | Implementado | Aplica reglas del cap. 9 y produce `loans_clean` + CSV limpio |
| Procesamiento — transformación (feature engineering) | Implementado | Genera features derivadas y produce `loans_transformed` + CSV |
| Auditoría de calidad | Implementado | Verifica `loans_clean` y `loans_transformed` contra las reglas |
| Entrenamiento | Pendiente | Generación del modelo de clasificación |
| Predicción | Pendiente | Evaluación de nuevas solicitudes |
| Almacenamiento | Implementado (Postgres) | Persistencia de datos crudos y procesados |

## 6. Diagrama de arquitectura lógica

```
                 ┌────────────────────────────┐
                 │  Usuario (Actor Externo)   │
                 └─────────────┬──────────────┘
                               │
   Capa de Presentación        ▼
                 ┌────────────────────────────┐
                 │  API REST (FastAPI)        │   (fase posterior)
                 └─────────────┬──────────────┘
                               │
   Capa de Lógica de Negocio   ▼
                 ┌────────────────────────────┐
                 │  Módulo de procesamiento   │
                 │  (data preprocessing)      │
                 └─────────────┬──────────────┘
                               │
                 ┌────────────────────────────┐
                 │  Módulo de modelo IA       │   (fase posterior)
                 └─────────────┬──────────────┘
                               │
   Capa de Datos               ▼
                 ┌────────────────────────────┐
                 │  PostgreSQL                │
                 │  loans_raw / loans_clean   │
                 └────────────────────────────┘
```

## 7. Diagrama de flujo de datos (DFD)

```
Usuario ──[datos del préstamo]──▶ Ingesta ──[datos crudos]──▶ Limpieza
                                                              │
                                       ┌──[datos preparados]──┘
                                       ▼
                                   Modelo IA ──[predicción]──▶ PostgreSQL ──▶ Resultado
```

El DFD permite visualizar entradas, procesos internos (limpieza, transformación, predicción) y salidas, asegurando trazabilidad del flujo.

## 8. Diagrama lógico de base de datos

Modelo lógico normalizado propuesto:

```
┌────────────────────────┐         ┌──────────────────────────┐
│    historial_crediticio│         │       solicitantes       │
├────────────────────────┤         ├──────────────────────────┤
│ id (PK)                │1───────1│ id (PK)                  │
│ cb_person_cred_hist_   │         │ person_age               │
│   length               │         │ person_gender            │
│ credit_score           │         │ person_education         │
│ previous_loan_defaults │         │ person_income            │
│   _on_file             │         │ person_emp_exp           │
└────────────────────────┘         │ person_home_ownership    │
                                   └────────────┬─────────────┘
                                                │ 1
                                                │
                                                │ N
                                   ┌────────────▼─────────────┐
                                   │        prestamos         │
                                   ├──────────────────────────┤
                                   │ id (PK)                  │
                                   │ loan_amnt                │       ┌──────────────┐
                                   │ loan_intent              │1─────1│   resultado  │
                                   │ loan_int_rate            │       ├──────────────┤
                                   │ loan_percent_income      │       │ id (PK)      │
                                   └──────────────────────────┘       │ loan_status  │
                                                                      └──────────────┘
```

**Nota de implementación física:** en esta entrega se materializa un modelo plano (`loans_raw`, `loans_clean`) que es espejo del CSV con dos columnas técnicas (`id`, `fecha_carga`). La normalización del modelo lógico queda disponible para una iteración posterior; un solo `SELECT *` simplifica la pipeline de ML aguas abajo.

## 9. Diccionario de datos

Esta sección es el **contrato de validación** que aplica el módulo de limpieza. Las reglas se derivan del dataset real (`data/loan_data.csv`).

### 9.1 Entidad: Solicitante (Person)

Atributos demográficos y de situación financiera/laboral de la persona que solicita el crédito.

| Campo | Descripción | Tipo de dato | Reglas de validación / negocio | Ejemplos |
|---|---|---|---|---|
| `person_age` | Edad biológica del solicitante. | Flotante (float) | Rango: 18 ≤ x ≤ 100. Valores fuera del rango se eliminan en limpieza. | 22.0, 25.0, 34.0 |
| `person_gender` | Identidad de género del solicitante. | Categórico | Dominio: {male, female}. | female, male |
| `person_education` | Máximo nivel educativo completado. | Categórico | Dominio: {High School, Bachelor, Master, Associate, Doctorate}. | Bachelor, Master |
| `person_income` | Ingreso anual bruto reportado. | Flotante (float) | x ≥ 0. | 71948.0, 12282.0 |
| `person_emp_exp` | Años de experiencia laboral activa. | Entero (int) | 0 ≤ x ≤ (person_age − 18). El dataset crudo contiene casos con valores >100 que se eliminan en limpieza. | 0, 3, 5 |
| `person_home_ownership` | Relación legal con la residencia. | Categórico | Dominio: {RENT, OWN, MORTGAGE, OTHER}. | RENT, MORTGAGE |
| `cb_person_cred_hist_length` | Años desde la apertura de la primera cuenta de crédito. | Flotante (float) | 0 ≤ x ≤ person_age. | 2.0, 3.0, 4.0 |
| `credit_score` | Puntaje crediticio (estándar FICO). | Entero (int) | 300 ≤ x ≤ 850. | 561, 635, 708 |

### 9.2 Entidad: Préstamo (Loan)

Características del préstamo solicitado y resultado de la clasificación de riesgo.

| Campo | Descripción | Tipo de dato | Reglas de validación / negocio | Ejemplos |
|---|---|---|---|---|
| `loan_amnt` | Monto total de capital solicitado. | Flotante (float) | x > 0. | 35000.0, 5500.0 |
| `loan_intent` | Motivo del préstamo. | Categórico | Dominio: {PERSONAL, EDUCATION, MEDICAL, VENTURE, DEBTCONSOLIDATION, **HOMEIMPROVEMENT**}. | PERSONAL, MEDICAL |
| `loan_int_rate` | Tasa de interés anual **expresada en porcentaje**. | Flotante (float) | 5 ≤ x ≤ 30. Ej: 16.02 representa 16,02%. | 16.02, 11.14, 7.9 |
| `loan_percent_income` | Razón préstamo / ingreso anual. | Flotante (float) | 0 ≤ x ≤ 1. Ej: 0.49 representa 49%. | 0.49, 0.08 |
| `previous_loan_defaults_on_file` | Defaults anteriores. | Categórico | Dominio: {Yes, No}. Factor crítico para el modelo. | No, Yes |
| `loan_status` | **Variable objetivo.** Resultado de la clasificación. | Binario (int) | Dominio: {0, 1}. 0 = Pagado/Aprobado, 1 = Default/No aprobado. | 0, 1 |

### 9.3 Reglas adicionales de la limpieza

Más allá de los rangos por columna, el módulo de limpieza:

- Elimina filas duplicadas.
- Elimina filas con valores nulos.
- Devuelve siempre la tabla `loans_clean` con la misma forma del CSV original (no cambia tipos ni nombres de columnas).

### 9.4 Features derivadas (módulo de transformación)

Después de la limpieza, el módulo de transformación agrega tres variables derivadas a `loans_transformed`. Son cálculos determinísticos (no dependen de la distribución del dataset), por lo que se pueden computar antes del *train/test split* sin riesgo de *data leakage*.

| Campo | Descripción | Tipo | Lógica |
|---|---|---|---|
| `fico_band` | Banda FICO del puntaje crediticio | Entero (1–5) | 1=Poor (300–579), 2=Fair (580–669), 3=Good (670–739), 4=Very Good (740–799), 5=Exceptional (800–850) |
| `age_group` | Grupo etario del solicitante | Entero (1–3) | 1=Joven (18–29), 2=Adulto (30–54), 3=Senior (55–100) |
| `rate_x_pct_income` | Interacción tasa × ratio préstamo/ingreso | Flotante | `loan_int_rate × loan_percent_income`. Captura riesgo combinado de tasa alta sobre alta proporción del ingreso. |

El **encoding de variables categóricas** y el **escalado de variables numéricas** *no* se aplican en esta etapa: se delegan al pipeline de modelado (`sklearn.Pipeline`) para que se ajusten únicamente con datos de entrenamiento.

### 9.5 Cambios respecto a la versión 1 del documento

- `loan_intent`: se incorpora la categoría **HOMEIMPROVEMENT** (presente en el dataset, ~10% de las filas; faltaba en la v1).
- `person_age`, `person_income`, `loan_amnt`, `cb_person_cred_hist_length`: se corrige el tipo a *flotante* (la v1 los listaba como enteros pese a venir con decimales en el CSV).
- `loan_int_rate`: se aclara que el valor está expresado en porcentaje (5–30), no en puntos básicos.
- `loan_percent_income`: se confirma como flotante (la v1 lo reportaba como entero en algunos lugares).

## 10. Control de versiones

Repositorio Git en GitHub para:
- Versiones del código.
- Control de cambios.
- Trabajo colaborativo entre los integrantes del equipo.

## 11. Documentación técnica

- `README.md` — documentación operativa: instalación, ejecución, estructura.
- `docs/diseño_tecnico.md` — documento de diseño (este archivo).
- Comentarios mínimos en el código, focalizados en el "porqué" de decisiones no obvias.

## 12. Ejemplo aplicado al caso

Si el sistema recibe miles de solicitudes simultáneas, la capa de aplicación puede escalar horizontalmente mediante contenedores Docker orquestados con Kubernetes, asegurando disponibilidad del servicio.

## 13. Interoperabilidad

La arquitectura permite integrar el sistema con plataformas externas (por ejemplo, APIs bancarias) mediante FastAPI, facilitando la comunicación entre sistemas.
