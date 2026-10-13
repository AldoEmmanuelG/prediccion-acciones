"""Validacion cruzada temporal por fechas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def splits_por_fecha(
    fechas_fila: pd.Series | np.ndarray, n_splits: int, gap: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pliegues de ventana creciente sobre las fechas unicas.

    Como el panel tiene varias acciones por dia, se parte por fecha (no por fila)
    para que todas las acciones de un mismo dia caigan del mismo lado. Devuelve
    pares (indices_train, indices_validacion) de filas, listos para `cv=` de
    `GridSearchCV`. `gap` separa entrenamiento y validacion en dias habiles.
    """
    fechas = np.asarray(fechas_fila)
    unicas = np.sort(pd.unique(fechas))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    salida = []
    for i_tr, i_va in tscv.split(unicas):
        tr = np.flatnonzero(np.isin(fechas, unicas[i_tr]))
        va = np.flatnonzero(np.isin(fechas, unicas[i_va]))
        salida.append((tr, va))
    return salida
