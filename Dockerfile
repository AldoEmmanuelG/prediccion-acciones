# Imagen de la app: Streamlit + modelos ya entrenados.
# Antes de construirla, `models/` debe existir (corre `dvc pull` o el entrenamiento).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

# 1) Dependencias primero: esta capa solo se reconstruye si cambia uv.lock.
#    --no-default-groups deja fuera pytest, ruff, mlflow y dvc (no hacen falta aqui).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

# 2) Codigo, configuracion y modelos.
COPY src ./src
COPY configs ./configs
COPY .streamlit ./.streamlit
COPY models ./models

# 3) Instala el paquete en modo editable: asi RAIZ_PROYECTO apunta a /app y el
#    codigo encuentra configs/ y models/ donde los espera.
RUN uv sync --frozen --no-default-groups

ENV PATH="/app/.venv/bin:$PATH"

# No correr como root dentro del contenedor.
RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# 0.0.0.0 para que se pueda entrar desde fuera del contenedor.
CMD ["streamlit", "run", "src/prediccion_acciones/app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
