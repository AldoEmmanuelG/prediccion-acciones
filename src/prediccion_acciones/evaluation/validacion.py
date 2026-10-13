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


def dividir_por_fecha(
    fechas_fila: pd.Series | np.ndarray, fraccion_val: float, gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Separa las ultimas fechas como validacion, dejando `gap` dias de hueco.

    Sirve para el early stopping de la red: la validacion siempre es posterior al
    entrenamiento. Devuelve indices de filas (entrenamiento, validacion).
    """
    if not 0.0 < fraccion_val < 1.0:
        raise ValueError("fraccion_val debe estar entre 0 y 1")
    fechas = np.asarray(fechas_fila)
    unicas = np.sort(pd.unique(fechas))
    n_val = int(np.ceil(len(unicas) * fraccion_val))
    n_train = len(unicas) - n_val - gap
    if n_train < 1:
        raise ValueError("No hay fechas suficientes para entrenar y validar")
    idx_train = np.flatnonzero(np.isin(fechas, unicas[:n_train]))
    idx_val = np.flatnonzero(np.isin(fechas, unicas[-n_val:]))
    return idx_train, idx_val
