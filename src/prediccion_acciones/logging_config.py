"""Configuracion central de logging: un solo lugar decide formato y nivel."""

from __future__ import annotations

import logging
import os

FORMATO = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
NIVELES_VALIDOS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configurar_logging(nivel: str | None = None) -> None:
    """Configura el logging raiz.

    El nivel sale, en este orden, del argumento, de la variable de entorno
    `LOG_LEVEL` o de INFO. Se puede llamar varias veces: reemplaza la
    configuracion anterior en vez de duplicar los mensajes.
    """
    elegido = (nivel or os.environ.get("LOG_LEVEL") or "INFO").upper()
    if elegido not in NIVELES_VALIDOS:
        elegido = "INFO"
    logging.basicConfig(level=elegido, format=FORMATO, force=True)
    logging.getLogger(__name__).debug("Logging configurado en nivel %s", elegido)
