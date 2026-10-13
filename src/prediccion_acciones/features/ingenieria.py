"""Features y etiqueta a partir de la tabla larga de precios.

Reglas que no se rompen:
- Las features de un dia usan solo datos hasta el cierre de ese dia.
- La etiqueta mira `horizonte` dias hacia adelante. Los ultimos dias de cada
  ticker no tienen futuro conocido y se eliminan (comparar contra NaN daria
  `False` y metaria una etiqueta falsa de "baja").
- Son retornos y razones, no precios: asi el mismo modelo sirve para todas las
  acciones, y el nombre del ticker no entra como feature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from prediccion_acciones.config import FeaturesConfig

logger = logging.getLogger(__name__)

COLUMNA_ETIQUETA = "y"
COLUMNAS_PRECIOS = ("fecha", "ticker", "cierre", "volumen")


def nombres_features(cfg: FeaturesConfig) -> list[str]:
    """Nombres de las features, en el orden en que salen en la tabla."""
    nombres = [f"ret_{k}d" for k in cfg.ventanas_retorno]
    for k in cfg.ventanas_rolling:
        nombres += [f"media_{k}d", f"desv_{k}d"]
    nombres += [
        f"rsi_{cfg.rsi_periodo}",
        f"dist_media_{max(cfg.ventanas_rolling)}d",
        "vol_rel",
    ]
    return nombres


def rsi(cierre: pd.Series, periodo: int) -> pd.Series:
    """RSI de Wilder: promedio exponencial de ganancias entre el de perdidas."""
    delta = cierre.diff()
    kw = {"alpha": 1 / periodo, "adjust": False, "min_periods": periodo}
    ganancia = delta.clip(lower=0).ewm(**kw).mean()
    perdida = (-delta.clip(upper=0)).ewm(**kw).mean()
    return 100 - 100 / (1 + ganancia / perdida)


def _features_un_ticker(
    cierre: pd.Series, volumen: pd.Series, cfg: FeaturesConfig
) -> pd.DataFrame:
    retorno_diario = cierre / cierre.shift(1) - 1
    f = pd.DataFrame(index=cierre.index)
    for k in cfg.ventanas_retorno:
        f[f"ret_{k}d"] = cierre / cierre.shift(k) - 1
    for k in cfg.ventanas_rolling:
        f[f"media_{k}d"] = retorno_diario.rolling(k).mean()
        f[f"desv_{k}d"] = retorno_diario.rolling(k).std()
    larga = max(cfg.ventanas_rolling)
    f[f"rsi_{cfg.rsi_periodo}"] = rsi(cierre, cfg.rsi_periodo)
    f[f"dist_media_{larga}d"] = cierre / cierre.rolling(larga).mean() - 1
    f["vol_rel"] = volumen / volumen.rolling(larga).mean()
    return f


def _validar_columnas(precios: pd.DataFrame) -> None:
    faltan = [c for c in COLUMNAS_PRECIOS if c not in precios.columns]
    if faltan:
        raise ValueError(f"A la tabla de precios le faltan columnas: {faltan}")


def calcular_features(precios: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Features por (fecha, ticker), sin etiqueta.

    Incluye el ultimo dia de cada ticker: es el que la app necesita para predecir.
    Se descartan las filas del arranque, donde las ventanas aun no se llenan.
    """
    _validar_columnas(precios)
    partes = []
    for ticker, grupo in precios.groupby("ticker", sort=True):
        g = grupo.dropna(subset=["cierre"]).sort_values("fecha")
        f = _features_un_ticker(
            g["cierre"].reset_index(drop=True),
            g["volumen"].reset_index(drop=True),
            cfg,
        )
        f.insert(0, "fecha", g["fecha"].to_numpy())
        f.insert(1, "ticker", ticker)
        partes.append(f)
    if not partes:
        raise ValueError("La tabla de precios esta vacia")
    tabla = pd.concat(partes, ignore_index=True)
    tabla = tabla.replace([np.inf, -np.inf], np.nan).dropna()
    return tabla.sort_values(["fecha", "ticker"]).reset_index(drop=True)


def calcular_etiqueta(precios: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Etiqueta `y` = 1 si el cierre de t+horizonte supera al de t.

    Las filas sin futuro conocido no aparecen en el resultado.
    """
    if horizonte < 1:
        raise ValueError("El horizonte debe ser al menos 1 dia")
    _validar_columnas(precios)
    partes = []
    for ticker, grupo in precios.groupby("ticker", sort=True):
        g = grupo.dropna(subset=["cierre"]).sort_values("fecha")
        cierre = g["cierre"].reset_index(drop=True)
        futuro = cierre.shift(-horizonte)
        etiqueta = pd.DataFrame(
            {
                "fecha": g["fecha"].to_numpy(),
                "ticker": ticker,
                COLUMNA_ETIQUETA: (futuro > cierre).astype(int),
            }
        )
        partes.append(etiqueta[futuro.notna().to_numpy()])
    return pd.concat(partes, ignore_index=True)


def construir_panel(
    precios: pd.DataFrame, cfg: FeaturesConfig, horizonte: int
) -> pd.DataFrame:
    """Tabla de entrenamiento: `fecha, ticker, <features>, y`, ordenada por fecha."""
    features = calcular_features(precios, cfg)
    etiqueta = calcular_etiqueta(precios, horizonte)
    panel = features.merge(etiqueta, on=["fecha", "ticker"], how="inner")
    panel = panel.sort_values(["fecha", "ticker"]).reset_index(drop=True)
    logger.info(
        "Panel construido: %d filas, %d tickers, %s a %s",
        len(panel),
        panel["ticker"].nunique(),
        panel["fecha"].min().date(),
        panel["fecha"].max().date(),
    )
    return panel
