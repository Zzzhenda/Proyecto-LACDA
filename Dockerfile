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

# Pipeline completo: ingesta -> qualitycheck -> limpieza -> transformacion -> validacion (+carga) -> train -> test -> visuals -> DASHBOARD.
# Si una etapa falla (exit != 0), las siguientes no se ejecutan (gracias a `&&`).
# Al final de toda la cadena secuencial, se levanta el servidor persistente de Streamlit.
CMD ["sh", "-c", "python scripts/ingesta.py && python scripts/qualitycheck.py && python scripts/limpieza.py && python scripts/transformacion.py && python scripts/validacion.py && python scripts/train_model.py && python scripts/test_model.py && python scripts/visuals.py && streamlit run scripts/app.py --server.port=8501 --server.address=0.0.0.0"]