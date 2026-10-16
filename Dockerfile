# Imagen del servicio: mini-API de FastAPI + modelos ya entrenados.
# Los modelos van en el repositorio (carpeta models/), asi que basta con clonar y
# correr `docker build -t prediccion-acciones .`
FROM python:3.12-slim

# UV_NO_CACHE: uv no deja dentro de la imagen su cache de paquetes descargados.
# Sin esto la cache duplica el entorno virtual (~1.3 GB de mas en la imagen).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# 1) Dependencias primero: esta capa solo se reconstruye si cambia uv.lock.
#    --no-default-groups deja fuera pytest, ruff, mlflow y dvc (no hacen falta aqui).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

# 2) Codigo, configuracion y modelos.
COPY src ./src
COPY configs ./configs
COPY models ./models

# 3) Instala el paquete en modo editable: asi RAIZ_PROYECTO apunta a /app y el
#    codigo encuentra configs/ y models/ donde los espera.
RUN uv sync --frozen --no-default-groups

ENV PATH="/app/.venv/bin:$PATH"

# No correr como root dentro del contenedor. No se hace chown de /app: el usuario
# solo necesita LEER el codigo y el entorno, y cambiar el dueno de miles de archivos
# duplica la capa (mas tamano y minutos de espera en cada construccion).
RUN useradd --create-home app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 0.0.0.0 para que se pueda entrar desde fuera del contenedor.
CMD ["uvicorn", "prediccion_acciones.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
