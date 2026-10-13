"""Particion temporal del panel: entrenamiento, holdout y prueba hacia adelante."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Particiones:
    """Tres bloques separados por fecha, nunca mezclados.

    - entrenamiento: sirve para elegir modelo (con validacion cruzada temporal).
    - holdout: ultimos meses antes del corte; se abre una sola vez para reportar.
    - adelante: fechas posteriores al corte; prueba hacia adelante de la app.
    """

    entrenamiento: pd.DataFrame
    holdout: pd.DataFrame
    adelante: pd.DataFrame


def dividir_panel(
    panel: pd.DataFrame, fecha_corte: date, meses_holdout: int, gap_dias: int
) -> Particiones:
    """Parte el panel por fecha.

    El holdout son los `meses_holdout` meses anteriores a `fecha_corte`. Entre el
    final del entrenamiento y el inicio del holdout se dejan `gap_dias` dias
    habiles sin usar, porque la etiqueta de un dia depende del siguiente.
    """
    corte = pd.Timestamp(fecha_corte)
    adelante = panel[panel["fecha"] > corte]
    hasta_corte = panel[panel["fecha"] <= corte]
    if hasta_corte.empty:
        raise ValueError("No hay datos hasta la fecha de corte")

    fechas = np.sort(hasta_corte["fecha"].unique())
    inicio_holdout = corte - pd.DateOffset(months=meses_holdout)
    i_hold = int(np.searchsorted(fechas, np.datetime64(inicio_holdout)))
    i_fin_train = i_hold - gap_dias - 1
    if i_hold >= len(fechas):
        raise ValueError("El holdout quedo vacio: revisa meses_holdout")
    if i_fin_train < 0:
        raise ValueError("No queda historia para entrenar antes del holdout")

    fin_train = fechas[i_fin_train]
    particiones = Particiones(
        entrenamiento=hasta_corte[hasta_corte["fecha"] <= fin_train].reset_index(
            drop=True
        ),
        holdout=hasta_corte[hasta_corte["fecha"] >= fechas[i_hold]].reset_index(
            drop=True
        ),
        adelante=adelante.reset_index(drop=True),
    )
    logger.info(
        "Particion: entrenamiento %d filas, holdout %d, adelante %d",
        len(particiones.entrenamiento),
        len(particiones.holdout),
        len(particiones.adelante),
    )
    return particiones
