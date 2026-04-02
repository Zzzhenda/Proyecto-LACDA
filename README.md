# Proyecto-LACDA
# Predicción de Agotamiento de Stock en Productos

Este proyecto tiene como objetivo anticipar, mediante un modelo de IA, si un producto en bodega se agotará en los próximos días, utilizando variables como historial de ventas, frecuencia de reposición y estacionalidad.

---

## Componentes del sistema

- **Scripts de procesamiento**: ingesta, limpieza, transformación y validación de datos.
- **Base de datos PostgreSQL**: para la carga y consulta estructurada de los datasets.
- **Modelo de IA (scikit-learn)**: clasificación binaria para predecir agotamiento.
- **Metabase (opcional)**: dashboard de visualización de resultados.
- **Documentación**: diseño técnico completo + planificación.

---

## Tecnologías utilizadas
- VS Code
- Python 3  
- Pandas / Scikit-learn  
- PostgreSQL  
- Docker
- Kubernetes
- Render
- Git / GitHub
- GitHub Actions
- Pandas

---

## Pipeline implementado

| Etapa | Descripción |
|-------|-------------|
| 1. Diseño e instalación | Estructura de carpetas, setup del entorno, definición de herramientas |
| 2. Ingesta | Lectura desde CSV (Kaggle o Mockaroo), carga a memoria |
| 3. Limpieza | Eliminación de duplicados, tratamiento de nulos, revisión de tipos |
| 4. Transformación | Creación de variables como días sin reposición, tasa de ventas, etc. |
| 5. Validación | Revisión de rangos, tipos, coherencia; validación básica |
| 6. Carga en PostgreSQL | Subida del dataset limpio y validado a la base de datos local |
| 7. Entrenamiento IA | Clasificación binaria con scikit-learn para variable `SeAgotara` |
| 8. Evaluación | Métricas como accuracy, recall; revisión de logs de ejecución |
| 9. Visualización | Panel con predicciones, stock proyectado y alertas (si aplica) |

---

## 📂 Estructura del repositorio

```
agotamiento-stock/
├── README.md
├── docs/
│   └── diseño_tecnico.pdf
├── scripts/
│   ├── ingesta.py
│   ├── limpieza.py
│   ├── transformacion.py
│   └── entrenamiento.py
├── data/
│   └── productos_ventas.csv
├── dashboards/
│   └── dashboard_metabase.png
├── docker-compose.yml
```

---

## Cómo ejecutar el sistema (entorno ya instalado)

1. Clonar el repositorio  
   `git clone https://github.com/usuario/agotamiento-stock.git`

2. Entrar a la carpeta del proyecto  
   `cd agotamiento-stock`

3. Ejecutar el pipeline manualmente por etapas  
   Ejemplo:  
   `python scripts/ingesta.py`  
   `python scripts/limpieza.py`  
   `python scripts/entrenamiento.py`

4. Visualizar los resultados y métricas desde consola o dashboard

---

## Documentación técnica

El documento de diseño técnico está disponible en:  
[`docs/diseño_tecnico.pdf`](docs/diseño_tecnico.pdf)

---

## Equipo

- Nicolas Fernandez – Procesamiento y limpieza  
- Bastian Gutierrez – Modelado y entrenamiento  
- Victor Gutierrez – Visualización y documentación
