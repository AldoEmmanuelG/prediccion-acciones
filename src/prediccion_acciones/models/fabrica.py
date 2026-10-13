"""Fabrica: crea modelos por nombre sin que quien los usa conozca la clase."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Importar los modulos con modelos los registra (efecto del decorador).
from prediccion_acciones.models import pytorch_modelo, sklearn_modelo  # noqa: F401
from prediccion_acciones.models.base import (
    BaseModel,
    ErrorModelo,
    constructores_registrados,
)


def modelos_disponibles() -> list[str]:
    return sorted(constructores_registrados())


def crear_modelo(
    nombre: str, columnas: Sequence[str], semilla: int = 42, **hiper: Any
) -> BaseModel:
    """Devuelve un modelo sin entrenar. `hiper` se pasa al clasificador."""
    registro = constructores_registrados()
    if nombre not in registro:
        raise ErrorModelo(
            f"Modelo '{nombre}' desconocido. Disponibles: {modelos_disponibles()}"
        )
    return registro[nombre](columnas, semilla=semilla, **hiper)
