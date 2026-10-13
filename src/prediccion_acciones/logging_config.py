"""Configuracion central de logging: un solo lugar decide formato y nivel."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from prediccion_acciones.config import RAIZ_PROYECTO

FORMATO = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
NIVELES_VALIDOS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configurar_logging(
    nivel: str | None = None, ruta_env: Path | str = RAIZ_PROYECTO / ".env"
) -> None:
    """Configura el logging raiz.

    El nivel sale, en este orden, del argumento, de la variable de entorno
    `LOG_LEVEL` (que tambien se lee del archivo `.env` si existe) o de INFO. Una
    variable ya definida en el entorno gana sobre el archivo. Se puede llamar
    varias veces: reemplaza la configuracion anterior en vez de duplicar mensajes.
    """
    load_dotenv(ruta_env, override=False)
    elegido = (nivel or os.environ.get("LOG_LEVEL") or "INFO").upper()
    if elegido not in NIVELES_VALIDOS:
        elegido = "INFO"
    logging.basicConfig(level=elegido, format=FORMATO, force=True)
    logging.getLogger(__name__).debug("Logging configurado en nivel %s", elegido)
