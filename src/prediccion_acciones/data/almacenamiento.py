"""Lectura y escritura de tablas en Parquet (formato columnar, tipado y compacto)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def guardar_parquet(tabla: pd.DataFrame, ruta: Path | str) -> Path:
    """Guarda la tabla; crea la carpeta si no existe. Requiere `pyarrow`."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(ruta, index=False)
    logger.info("Guardadas %d filas en %s", len(tabla), ruta)
    return ruta


def cargar_parquet(ruta: Path | str) -> pd.DataFrame:
    """Lee una tabla guardada con `guardar_parquet`."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Corre primero la descarga.")
    tabla = pd.read_parquet(ruta)
    logger.info("Leidas %d filas de %s", len(tabla), ruta)
    return tabla
