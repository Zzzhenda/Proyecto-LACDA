Sistema de Clasificación de Aprobación de Préstamos
Este proyecto tiene como objetivo anticipar, mediante un modelo de IA, si una solicitud de préstamo bancario terminará en default (incumplimiento) o en pago exitoso, utilizando variables como historial crediticio, ingresos, experiencia laboral y monto del préstamo.


Componentes del sistema
Scripts de procesamiento: ingesta, limpieza, transformación y validación de datos.
Base de datos PostgreSQL: para la carga y consulta estructurada del dataset.
Modelo de IA (scikit-learn): clasificación binaria para predecir loan_status.
Documentación: diseño técnico completo + planificación.


Tecnologías utilizadas
Python 3
Pandas / NumPy / Scikit-learn
PostgreSQL
Docker
Git / GitHub / GitHub Actions
Render


Pipeline implementado
Etapa
Descripción
1. Diseño e instalación
Estructura de carpetas, setup del entorno, definición de herramientas
2. Ingesta
Lectura desde CSV, carga a memoria con pandas
3. Limpieza
Eliminación de duplicados, tratamiento de nulos, revisión de tipos
4. Transformación
Encoding de variables categóricas, normalización, validación de rangos
5. Carga en PostgreSQL
Subida del dataset limpio y validado a la base de datos local
6. Entrenamiento IA
Clasificación binaria con scikit-learn para la variable loan_status
7. Evaluación
Métricas: accuracy, F1, ROC-AUC; revisión de logs de ejecución



Estructura del repositorio
loan-approval-classification/

├── README.md

├── requirements.txt

├── docker-compose.yml

├── docs/

│   ├── diseño_tecnico.docx

│   └── planificacion.docx

├── scripts/

│   ├── ingesta.py

│   ├── limpieza.py

│   ├── transformacion.py

│   ├── carga_db.py

│   └── entrenamiento.py

├── data/

│   └── loan_data.csv

└── models/

    └── modelo_clasificacion.pkl


Cómo ejecutar el sistema (entorno ya instalado)
Clonar el repositorio
git clone https://github.com/grupo4/loan-approval-classification.git

Entrar a la carpeta del proyecto
cd loan-approval-classification

Ejecutar el pipeline por etapas

python scripts/ingesta.py

python scripts/limpieza.py

python scripts/transformacion.py

python scripts/carga_db.py

python scripts/entrenamiento.py

Revisar métricas y resultados desde consola


Documentación técnica
El documento de diseño técnico está disponible en:
docs/diseño_tecnico.docx


Equipo — Grupo 4
Nicolás Fernández Vera – Procesamiento y limpieza
Bastián Gutiérrez – Modelado y entrenamiento
Víctor Gutiérrez – Documentación y arquitectura

