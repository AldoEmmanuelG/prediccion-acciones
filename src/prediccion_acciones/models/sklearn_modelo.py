"""Modelos de scikit-learn envueltos en el contrato `BaseModel`.

Cada uno es un `Pipeline` con un `ColumnTransformer` que estandariza las
columnas numericas y un clasificador al final. Estandarizar dentro del Pipeline
evita que la media y la desviacion de los datos de validacion se filtren al
entrenamiento durante la validacion cruzada.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prediccion_acciones.models.base import BaseModel, ErrorModelo, registrar

logger = logging.getLogger(__name__)

PASO_CLASIFICADOR = "clf"


def crear_pipeline(columnas: Sequence[str], clasificador: Any) -> Pipeline:
    """Estandariza las columnas indicadas y aplica el clasificador."""
    preprocesado = ColumnTransformer(
        [("num", StandardScaler(), list(columnas))], remainder="drop"
    )
    return Pipeline([("prep", preprocesado), (PASO_CLASIFICADOR, clasificador)])


class ModeloSklearn(BaseModel):
    """Envuelve un Pipeline de sklearn para que cumpla el contrato comun."""

    def __init__(
        self,
        columnas: Sequence[str],
        estimador: Pipeline,
        nombre: str,
        umbral: float = 0.5,
        ajustado: bool = False,
    ) -> None:
        super().__init__(columnas, umbral)
        self.estimador = estimador
        self.nombre = nombre
        self._ajustado = ajustado

    def _ajustar(
        self, X: pd.DataFrame, y: np.ndarray, fechas: Sequence[Any] | None
    ) -> None:
        self.estimador.fit(X, y)

    def _probabilidad(self, X: pd.DataFrame) -> np.ndarray:
        clases = list(self.estimador.classes_)
        if 1 not in clases:
            # Entrenado solo con dias a la baja: la probabilidad de subir es 0.
            return np.zeros(len(X))
        return self.estimador.predict_proba(X)[:, clases.index(1)]

    def guardar(self, ruta: Path | str) -> None:
        ruta = Path(ruta)
        self._exigir_entrenado()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        paquete = {
            "nombre": self.nombre,
            "columnas": self.columnas,
            "umbral": self.umbral,
            "estimador": self.estimador,
        }
        with ruta.open("wb") as archivo:
            joblib.dump(paquete, archivo)
        logger.info("Modelo '%s' guardado en %s", self.nombre, ruta)

    @classmethod
    def cargar(cls, ruta: Path | str) -> Self:
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el modelo guardado: {ruta}")
        with ruta.open("rb") as archivo:
            paquete = joblib.load(archivo)
        faltan = {"nombre", "columnas", "umbral", "estimador"} - set(paquete)
        if faltan:
            raise ErrorModelo(f"Archivo de modelo incompleto, faltan: {sorted(faltan)}")
        return cls(
            paquete["columnas"],
            paquete["estimador"],
            paquete["nombre"],
            paquete["umbral"],
            ajustado=True,
        )


@registrar("dummy")
def _dummy(columnas: Sequence[str], semilla: int = 42, **hiper: Any) -> ModeloSklearn:
    """Baseline: siempre responde la clase mas frecuente. Su AUC es 0.50."""
    clf = DummyClassifier(strategy="most_frequent", **hiper)
    return ModeloSklearn(columnas, crear_pipeline(columnas, clf), "dummy")


@registrar("logistica")
def _logistica(
    columnas: Sequence[str], semilla: int = 42, **hiper: Any
) -> ModeloSklearn:
    clf = LogisticRegression(
        solver="liblinear", max_iter=1000, random_state=semilla, **hiper
    )
    return ModeloSklearn(columnas, crear_pipeline(columnas, clf), "logistica")


@registrar("boosting")
def _boosting(
    columnas: Sequence[str], semilla: int = 42, **hiper: Any
) -> ModeloSklearn:
    parametros = {"n_estimators": 100, "random_state": semilla, **hiper}
    clf = GradientBoostingClassifier(**parametros)
    return ModeloSklearn(columnas, crear_pipeline(columnas, clf), "boosting")
