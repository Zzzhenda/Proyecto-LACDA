FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Por defecto corre la pipeline completa: ingesta -> limpieza -> auditoria.
# Si una etapa falla, las siguientes no se ejecutan (gracias a `&&`).
# Para correr una etapa puntual: docker compose run --rm app python scripts/<etapa>.py
CMD ["sh", "-c", "python scripts/ingesta.py && python scripts/limpieza.py && python scripts/transformacion.py && python scripts/auditoria.py"]
