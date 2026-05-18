-- Esquema de la base de datos `loans`.
-- Modelo fisico alineado con el diseño tecnico (cap. 8-9):
--   * Entidad Solicitante (Person): datos demograficos y financieros del solicitante.
--   * Entidad Prestamo (Loan):      caracteristicas del credito y resultado.
--
-- Cada etapa de la pipeline produce un par de tablas (raw, clean, transformed)
-- siguiendo la misma separacion de entidades.

-- ============================================================
-- INGESTA (espejo del CSV crudo)
-- ============================================================

CREATE TABLE IF NOT EXISTS solicitantes_raw (
    id                          SERIAL PRIMARY KEY,
    person_age                  NUMERIC(5,1),
    person_gender               VARCHAR(20),
    person_education            VARCHAR(50),
    person_income               NUMERIC(12,2),
    person_emp_exp              INTEGER,
    person_home_ownership       VARCHAR(20),
    cb_person_cred_hist_length  NUMERIC(5,1),
    credit_score                INTEGER,
    fecha_carga                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prestamos_raw (
    id                              SERIAL PRIMARY KEY,
    solicitante_id                  INTEGER NOT NULL REFERENCES solicitantes_raw(id),
    loan_amnt                       NUMERIC(12,2),
    loan_intent                     VARCHAR(30),
    loan_int_rate                   NUMERIC(6,2),
    loan_percent_income             NUMERIC(6,4),
    previous_loan_defaults_on_file  VARCHAR(5),
    loan_status                     SMALLINT,
    fecha_carga                     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prestamos_raw_status ON prestamos_raw(loan_status);

-- ============================================================
-- LIMPIEZA (datos depurados segun reglas del cap. 9)
-- ============================================================

CREATE TABLE IF NOT EXISTS solicitantes_clean (
    id                          SERIAL PRIMARY KEY,
    person_age                  NUMERIC(5,1),
    person_gender               VARCHAR(20),
    person_education            VARCHAR(50),
    person_income               NUMERIC(12,2),
    person_emp_exp              INTEGER,
    person_home_ownership       VARCHAR(20),
    cb_person_cred_hist_length  NUMERIC(5,1),
    credit_score                INTEGER,
    fecha_carga                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prestamos_clean (
    id                              SERIAL PRIMARY KEY,
    solicitante_id                  INTEGER NOT NULL REFERENCES solicitantes_clean(id),
    loan_amnt                       NUMERIC(12,2),
    loan_intent                     VARCHAR(30),
    loan_int_rate                   NUMERIC(6,2),
    loan_percent_income             NUMERIC(6,4),
    previous_loan_defaults_on_file  VARCHAR(5),
    loan_status                     SMALLINT,
    fecha_carga                     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prestamos_clean_status ON prestamos_clean(loan_status);

-- ============================================================
-- TRANSFORMACION (features derivadas del cap. 9 + feature engineering)
-- ============================================================

CREATE TABLE IF NOT EXISTS solicitantes_transformed (
    id                          SERIAL PRIMARY KEY,
    person_age                  NUMERIC(5,1),
    person_gender               VARCHAR(20),
    person_education            VARCHAR(50),
    person_income               NUMERIC(12,2),
    person_emp_exp              INTEGER,
    person_home_ownership       VARCHAR(20),
    cb_person_cred_hist_length  NUMERIC(5,1),
    credit_score                INTEGER,
    -- Sin features derivadas a nivel solicitante: el EDA mostro que
    -- credit_score y person_age no correlacionan con loan_status en este
    -- dataset (|corr| < 0.03). Ver notebooks/features.ipynb.
    fecha_carga                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prestamos_transformed (
    id                              SERIAL PRIMARY KEY,
    solicitante_id                  INTEGER NOT NULL REFERENCES solicitantes_transformed(id),
    loan_amnt                       NUMERIC(12,2),
    loan_intent                     VARCHAR(30),
    loan_int_rate                   NUMERIC(6,2),
    loan_percent_income             NUMERIC(6,4),
    previous_loan_defaults_on_file  VARCHAR(5),
    loan_status                     SMALLINT,
    -- features derivadas (justificadas en notebooks/features.ipynb)
    rate_x_pct_income               NUMERIC(8,4),  -- interaccion tasa x % ingreso (|corr| 0.46)
    loan_burden                     NUMERIC(10,4), -- costo total / ingreso     (|corr| 0.40)
    has_prev_defaults               SMALLINT,      -- 0/1 encoding              (|corr| 0.54)
    fecha_carga                     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prestamos_transformed_status ON prestamos_transformed(loan_status);