"""Metricas de clasificacion y eleccion del umbral."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


@dataclass(frozen=True)
class Metricas:
    """Resultado de evaluar un modelo sobre un bloque de datos."""

    auc: float
    f1: float
    exactitud: float
    umbral: float
    n: int

    def a_dict(self) -> dict[str, float | int | None]:
        """Como diccionario; los NaN pasan a None para poder guardarlos en JSON."""
        return {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in asdict(self).items()
        }


def calcular_metricas(y: np.ndarray, prob: np.ndarray, umbral: float = 0.5) -> Metricas:
    """AUC (ordena bien?), F1 y exactitud con el umbral dado.

    Si `y` trae una sola clase el AUC no esta definido y se devuelve NaN.
    """
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    if len(y) != len(prob):
        raise ValueError("y y prob deben tener el mismo largo")
    prediccion = (prob >= umbral).astype(int)
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    return Metricas(
        auc=auc,
        f1=float(f1_score(y, prediccion, zero_division=0)),
        exactitud=float(accuracy_score(y, prediccion)),
        umbral=float(umbral),
        n=len(y),
    )


def umbral_youden(y: np.ndarray, prob: np.ndarray) -> float:
    """Umbral que maximiza TPR - FPR (punto de la curva ROC mas lejos de la diagonal).

    Se acota a [0.01, 0.99]. Si no hay dos clases devuelve 0.5.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, umbrales = roc_curve(y, prob)
    elegido = umbrales[int(np.argmax(tpr - fpr))]
    if not np.isfinite(elegido):
        return 0.5
    return float(np.clip(elegido, 0.01, 0.99))
