FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Principio de menor privilegio: el pipeline no necesita root dentro del contenedor.
RUN useradd --create-home appuser

COPY . .
USER appuser

# Pipeline completo: ingesta -> qualitycheck -> limpieza -> transformacion -> validacion (+carga).
# Si una etapa falla (exit != 0), las siguientes no se ejecutan (gracias a `&&`).
# La carga a la BD ocurre dentro de validacion.py, solo si todas las reglas pasan.
# Para correr una etapa puntual: docker compose run --rm app python scripts/<etapa>.py
CMD ["sh", "-c", "python scripts/ingesta.py && python scripts/qualitycheck.py && python scripts/limpieza.py && python scripts/transformacion.py && python scripts/validacion.py && python scripts/train_model.py && python scripts/test_model.py"]
