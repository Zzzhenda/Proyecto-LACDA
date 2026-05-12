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
    -- features derivadas del solicitante
    fico_band                   SMALLINT,   -- 1..5 segun rango FICO
    age_group                   SMALLINT,   -- 1=joven 2=adulto 3=senior
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
    -- feature derivada del prestamo
    rate_x_pct_income               NUMERIC(8,4),
    fecha_carga                     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prestamos_transformed_status ON prestamos_transformed(loan_status);