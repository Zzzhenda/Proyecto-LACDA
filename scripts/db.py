"""Helper compartido: construye el engine de SQLAlchemy a Postgres.

Lee credenciales de variables de entorno con defaults consistentes con
docker-compose.yml. Funciona tanto dentro del contenedor `app` (host=db)
como ejecutado localmente contra el puerto 5432 expuesto.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_engine() -> Engine:
    user = os.getenv("DB_USER", "lacda")
    pwd = os.getenv("DB_PASSWORD", "lacda_pass")
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "loans")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url, connect_args={"client_encoding": "utf8"})
