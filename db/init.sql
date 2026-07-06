-- Esquema de la base de datos `loans`.
--
-- Una sola tabla: `loan_data`. La base de datos es el DESTINO del pipeline
-- (etapa de carga), no su area de trabajo. Las etapas intermedias
-- (ingesta -> limpieza -> transformacion -> validacion) intercambian datos
-- via CSV en data/, y solo el dataset final, ya validado, se carga aqui.
--
-- Ventajas de este diseño:
--   * Simplicidad: un unico contrato de esquema que mantener.
--   * Eficiencia: una sola escritura a la BD por corrida (al final),
--     en lugar de 3 pares de escrituras intermedias.
--   * Garantia de calidad: a la tabla solo llegan datos que pasaron el
--     gate de validacion (validacion.py sale con exit 1 si algo falla,
--     y la carga nunca se ejecuta).


CREATE TABLE IF NOT EXISTS loan_data (
    id                              SERIAL PRIMARY KEY,

    -- Solicitante
    person_age                      NUMERIC(5,1),
    person_gender                   VARCHAR(20),
    person_education                VARCHAR(50),
    person_income                   NUMERIC(12,2),
    person_emp_exp                  INTEGER,
    person_home_ownership           VARCHAR(20),
    cb_person_cred_hist_length      NUMERIC(5,1),
    credit_score                    INTEGER,

    -- Prestamo
    loan_amnt                       NUMERIC(12,2),
    loan_intent                     VARCHAR(30),
    loan_int_rate                   NUMERIC(6,2),
    loan_percent_income             NUMERIC(6,4),
    previous_loan_defaults_on_file  VARCHAR(5),
    loan_status                     SMALLINT,

    -- Features derivadas (creadas en transformacion.py)
    rate_x_pct_income               NUMERIC(8,4),   -- loan_int_rate * loan_percent_income (|corr| ~0.46)
    loan_burden                     NUMERIC(10,4),  -- costo total / ingreso anual          (|corr| ~0.37)
    has_prev_defaults               SMALLINT,       -- encoding 0/1 de previous defaults    (|corr| ~0.54)

    -- Trazabilidad
    fecha_carga                     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_loan_data_status ON loan_data(loan_status);
