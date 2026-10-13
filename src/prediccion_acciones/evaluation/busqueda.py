"""Busqueda de hiperparametros con validacion cruzada temporal y dos metricas."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from prediccion_acciones.models.sklearn_modelo import (
    PASO_CLASIFICADOR,
    ModeloSklearn,
)

logger = logging.getLogger(__name__)

# AUC es la metrica principal (mide si el modelo ordena bien los dias); F1 se
# reporta aparte. Con ~53 % de dias al alza, F1 solo premia decir siempre "sube".
METRICAS = {"auc": "roc_auc", "f1": "f1"}
METRICA_PRINCIPAL = "auc"


@dataclass(frozen=True)
class ResultadoBusqueda:
    mejor_modelo: ModeloSklearn
    mejores_params: dict[str, Any]
    auc_cv: float
    auc_std: float
    f1_cv: float
    combinaciones: int
    tabla: pd.DataFrame


def buscar_hiperparametros(
    modelo: ModeloSklearn,
    rejilla: Mapping[str, Sequence[Any]],
    X: pd.DataFrame,
    y: Sequence[int] | np.ndarray,
    cv: Iterable[tuple[np.ndarray, np.ndarray]],
    n_iter: int | None = None,
    semilla: int = 42,
    n_jobs: int = -1,
) -> ResultadoBusqueda:
    """Rejilla (o busqueda aleatoria si se da `n_iter`) sobre el clasificador.

    Los nombres de `rejilla` son los del clasificador (por ejemplo `C`); aqui se
    les antepone el paso del Pipeline. Se reajusta con todo el entrenamiento
    usando el mejor AUC de validacion.
    """
    if not rejilla:
        raise ValueError("La rejilla de hiperparametros esta vacia")
    espacio = {f"{PASO_CLASIFICADOR}__{k}": list(v) for k, v in rejilla.items()}
    comunes: dict[str, Any] = {
        "cv": list(cv),
        "scoring": METRICAS,
        "refit": METRICA_PRINCIPAL,
        "n_jobs": n_jobs,
    }
    if n_iter is None:
        busqueda = GridSearchCV(modelo.estimador, espacio, **comunes)
    else:
        busqueda = RandomizedSearchCV(
            modelo.estimador, espacio, n_iter=n_iter, random_state=semilla, **comunes
        )
    busqueda.fit(X, np.asarray(y))

    i, r = busqueda.best_index_, busqueda.cv_results_
    mejores = {
        k.removeprefix(f"{PASO_CLASIFICADOR}__"): v
        for k, v in busqueda.best_params_.items()
    }
    mejor = ModeloSklearn(
        modelo.columnas,
        busqueda.best_estimator_,
        modelo.nombre,
        umbral=modelo.umbral,
        ajustado=True,
    )
    logger.info(
        "Busqueda '%s': AUC cv %.4f con %s",
        modelo.nombre,
        r["mean_test_auc"][i],
        mejores,
    )
    return ResultadoBusqueda(
        mejor_modelo=mejor,
        mejores_params=mejores,
        auc_cv=float(r["mean_test_auc"][i]),
        auc_std=float(r["std_test_auc"][i]),
        f1_cv=float(r["mean_test_f1"][i]),
        combinaciones=len(r["params"]),
        tabla=pd.DataFrame(r),
    )
