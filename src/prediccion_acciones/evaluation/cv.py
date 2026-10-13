"""Validacion cruzada temporal para cualquier `BaseModel`."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prediccion_acciones.evaluation.metricas import calcular_metricas
from prediccion_acciones.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultadoCV:
    """Metricas por pliegue y predicciones fuera de muestra de todos los pliegues."""

    auc_media: float
    auc_std: float
    f1_media: float
    por_pliegue: pd.DataFrame
    y_oof: np.ndarray
    prob_oof: np.ndarray


def validar_cruzada(
    crear: Callable[[], BaseModel],
    X: pd.DataFrame,
    y: np.ndarray,
    fechas: np.ndarray,
    cv: Sequence[tuple[np.ndarray, np.ndarray]],
) -> ResultadoCV:
    """Entrena un modelo nuevo en cada pliegue y mide sobre su validacion.

    `crear` devuelve un modelo sin entrenar. Se guardan las probabilidades de cada
    validacion ("fuera de muestra"): con ellas se elige el umbral sin tocar el
    holdout.
    """
    y = np.asarray(y)
    fechas = np.asarray(fechas)
    filas, ys, probs = [], [], []
    for i, (tr, va) in enumerate(cv, start=1):
        modelo = crear().fit(X.iloc[tr], y[tr], fechas=fechas[tr])
        prob = modelo.predict_proba(X.iloc[va])
        m = calcular_metricas(y[va], prob)
        filas.append({"pliegue": i, "auc": m.auc, "f1": m.f1, "n": m.n})
        ys.append(y[va])
        probs.append(prob)
        logger.debug("pliegue %d: AUC %.4f, F1 %.4f", i, m.auc, m.f1)
    tabla = pd.DataFrame(filas)
    return ResultadoCV(
        auc_media=float(tabla["auc"].mean()),
        auc_std=float(tabla["auc"].std(ddof=1)),
        f1_media=float(tabla["f1"].mean()),
        por_pliegue=tabla,
        y_oof=np.concatenate(ys),
        prob_oof=np.concatenate(probs),
    )
