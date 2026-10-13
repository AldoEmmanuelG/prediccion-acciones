"""Contrato comun de los modelos y registro para crearlos por nombre.

Patrones de diseno:
- Strategy: cada modelo (logistica, boosting, red neuronal) cumple el mismo
  contrato `BaseModel`; el resto del proyecto los usa sin saber cual es.
- Factory: los constructores se registran con `@registrar("nombre")` y se piden
  con `crear_modelo("nombre", ...)` (ver `fabrica.py`).
- Template Method: `fit`, `predict_proba` y `predict` hacen las validaciones
  comunes y llaman a los metodos abstractos que cada modelo implementa.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ErrorModelo(RuntimeError):
    """Uso incorrecto de un modelo: sin entrenar, columnas faltantes, nombre raro."""


class BaseModel(ABC):
    """Clasificador binario que devuelve la probabilidad de que el precio suba."""

    def __init__(self, columnas: Sequence[str], umbral: float = 0.5) -> None:
        if not columnas:
            raise ValueError("El modelo necesita al menos una columna de entrada")
        if not 0.0 < umbral < 1.0:
            raise ValueError("El umbral debe estar entre 0 y 1")
        self.columnas = tuple(columnas)
        self.umbral = umbral
        self._ajustado = False

    # --- Interfaz publica (plantilla): validaciones comunes ---

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[int] | np.ndarray,
        fechas: Sequence[Any] | np.ndarray | None = None,
    ) -> Self:
        """Entrena con las columnas del modelo, en el orden fijado al crearlo.

        `fechas` (una por fila) es opcional: la usan los modelos que separan una
        parte del entrenamiento por tiempo, como la red neuronal.
        """
        etiquetas = np.asarray(y)
        if len(etiquetas) != len(X):
            raise ValueError("X e y deben tener el mismo numero de filas")
        if fechas is not None and len(fechas) != len(X):
            raise ValueError("fechas y X deben tener el mismo numero de filas")
        self._ajustar(self._seleccionar(X), etiquetas, fechas)
        self._ajustado = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad (entre 0 y 1) de que el cierre de manana sea mayor."""
        self._exigir_entrenado()
        return np.asarray(self._probabilidad(self._seleccionar(X)), dtype=float)

    def predict(self, X: pd.DataFrame, umbral: float | None = None) -> np.ndarray:
        """Clase 1 (sube) si la probabilidad alcanza el umbral, 0 si no."""
        corte = self.umbral if umbral is None else umbral
        return (self.predict_proba(X) >= corte).astype(int)

    # --- Lo que cada modelo concreto implementa ---

    @abstractmethod
    def _ajustar(
        self, X: pd.DataFrame, y: np.ndarray, fechas: Sequence[Any] | None
    ) -> None:
        """Entrena con `X` que ya trae solo las columnas del modelo."""

    @abstractmethod
    def _probabilidad(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad de la clase 1 para cada fila de `X`."""

    @abstractmethod
    def guardar(self, ruta: Path | str) -> None:
        """Escribe el modelo entrenado en disco."""

    @classmethod
    @abstractmethod
    def cargar(cls, ruta: Path | str) -> Self:
        """Lee un modelo guardado con `guardar`."""

    # --- Ayudas internas ---

    def _seleccionar(self, X: pd.DataFrame) -> pd.DataFrame:
        faltan = [c for c in self.columnas if c not in X.columns]
        if faltan:
            raise ErrorModelo(f"A los datos les faltan columnas: {faltan}")
        return X[list(self.columnas)]

    def _exigir_entrenado(self) -> None:
        if not self._ajustado:
            raise ErrorModelo("El modelo no esta entrenado: llama primero a fit()")


# Un constructor recibe las columnas y la semilla y devuelve un modelo listo.
Constructor = Callable[..., BaseModel]
_REGISTRO: dict[str, Constructor] = {}


def registrar(nombre: str) -> Callable[[Constructor], Constructor]:
    """Decorador con parametros: anota un constructor en el registro."""

    def decorador(constructor: Constructor) -> Constructor:
        if nombre in _REGISTRO:
            raise ValueError(f"Ya existe un modelo registrado como '{nombre}'")
        _REGISTRO[nombre] = constructor
        return constructor

    return decorador


def constructores_registrados() -> dict[str, Constructor]:
    """Copia del registro (modificarla no altera el original)."""
    return dict(_REGISTRO)
