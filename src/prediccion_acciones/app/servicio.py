"""Logica de la app, sin Streamlit: cargar artefactos y calcular lo que se muestra.

Separar esto de la interfaz permite probarlo con pytest y cambiar la interfaz sin
tocar los calculos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prediccion_acciones.config import RAIZ_PROYECTO, Config
from prediccion_acciones.features.ingenieria import calcular_features
from prediccion_acciones.models.base import BaseModel
from prediccion_acciones.models.sklearn_modelo import ModeloSklearn

logger = logging.getLogger(__name__)

NOMBRES_MODELO = {
    "dummy": "Siempre sube (baseline)",
    "logistica": "Regresión logística",
    "boosting": "Gradient boosting",
    "pytorch": "MLP (PyTorch)",
}
# Modelos que muestra la app. El boosting tambien se entrena (queda en MLflow y en
# metricas.json) pero la app se queda con el baseline lineal y la red neuronal.
MODELOS_APP = ("logistica", "pytorch")
SECTORES = {
    "AMD": "Semiconductores",
    "ORCL": "Software",
    "IT": "Servicios de información",
    "IBM": "TI empresarial",
    "OSIS": "Electrónica de seguridad",
    "LMT": "Defensa",
    "KOF": "Consumo básico (ADR)",
    "HUM": "Salud (aseguradora)",
    "ANF": "Retail de ropa",
    "LII": "Climatización",
}
EXTENSIONES = {"pytorch": ".pt"}


class ErrorArtefactos(RuntimeError):
    """Faltan los archivos que deja el entrenamiento."""


class ErrorEntrada(ValueError):
    """El ticker o el modelo pedidos no existen."""


@dataclass(frozen=True)
class Artefactos:
    """Todo lo que la app necesita, cargado una sola vez."""

    cfg: Config
    precios: pd.DataFrame
    features: pd.DataFrame
    metricas: dict
    adelante: pd.DataFrame
    modelos: dict[str, BaseModel]

    @property
    def tickers(self) -> list[str]:
        return list(self.cfg.datos.tickers_objetivo)

    @property
    def modelos_app(self) -> list[str]:
        """Modelos cargados que la app ofrece, en el orden de `MODELOS_APP`."""
        return [m for m in MODELOS_APP if m in self.modelos]

    @property
    def ultima_fecha(self) -> pd.Timestamp:
        return self.precios["fecha"].max()


def _cargar_modelo(nombre: str, ruta: Path) -> BaseModel:
    if nombre == "pytorch":
        # Se importa aqui para que la app funcione aunque solo haya modelos sklearn.
        from prediccion_acciones.models.pytorch_modelo import ModeloPytorch

        return ModeloPytorch.cargar(ruta)
    return ModeloSklearn.cargar(ruta)


def cargar_artefactos(
    directorio: Path | str | None = None, cfg: Config | None = None
) -> Artefactos:
    """Lee los modelos, metricas y tablas que dejo `entrenamiento.py`."""
    directorio = Path(directorio) if directorio else RAIZ_PROYECTO / "models"
    cfg = cfg or Config.desde_yaml()
    ruta_metricas = directorio / "metricas.json"
    ruta_precios = directorio / "precios_app.csv"
    for ruta in (ruta_metricas, ruta_precios):
        if not ruta.exists():
            raise ErrorArtefactos(
                f"Falta {ruta.name} en {directorio}. "
                "Corre primero: uv run python -m prediccion_acciones.entrenamiento"
            )
    with ruta_metricas.open(encoding="utf-8") as f:
        metricas = json.load(f)
    precios = pd.read_csv(ruta_precios, parse_dates=["fecha"])
    ruta_adelante = directorio / "prueba_adelante.csv"
    adelante = (
        pd.read_csv(ruta_adelante, parse_dates=["fecha"])
        if ruta_adelante.exists()
        else pd.DataFrame(columns=["fecha", "ticker", "y"])
    )

    modelos: dict[str, BaseModel] = {}
    for nombre in metricas["modelos"]:
        ruta = directorio / f"{nombre}{EXTENSIONES.get(nombre, '.joblib')}"
        if nombre != "dummy" and ruta.exists():
            modelos[nombre] = _cargar_modelo(nombre, ruta)
    if not modelos:
        raise ErrorArtefactos(f"No hay modelos entrenados en {directorio}")
    logger.info(
        "App lista: modelos %s, datos hasta %s", list(modelos), precios["fecha"].max()
    )
    return Artefactos(
        cfg=cfg,
        precios=precios,
        features=calcular_features(precios, cfg.features),
        metricas=metricas,
        adelante=adelante,
        modelos=modelos,
    )


@dataclass(frozen=True)
class Prediccion:
    """Resultado de predecir el siguiente cierre de una accion."""

    ticker: str
    modelo: str
    fecha_datos: pd.Timestamp
    cierre: float
    probabilidad_sube: float
    umbral: float

    @property
    def sube(self) -> bool:
        return self.probabilidad_sube >= self.umbral


def prediccion_ticker(art: Artefactos, ticker: str, modelo: str) -> Prediccion:
    """Probabilidad de que `ticker` suba en el siguiente cierre, con `modelo`."""
    if ticker not in art.tickers:
        raise ErrorEntrada(f"Ticker '{ticker}' no disponible. Opciones: {art.tickers}")
    if modelo not in art.modelos:
        raise ErrorEntrada(
            f"Modelo '{modelo}' no disponible. Opciones: {sorted(art.modelos)}"
        )
    fila = art.features[art.features["ticker"] == ticker].sort_values("fecha").tail(1)
    precio = art.precios[
        (art.precios["ticker"] == ticker)
        & (art.precios["fecha"] == fila["fecha"].iloc[0])
    ]
    m = art.modelos[modelo]
    return Prediccion(
        ticker=ticker,
        modelo=modelo,
        fecha_datos=fila["fecha"].iloc[0],
        cierre=float(precio["cierre"].iloc[0]),
        probabilidad_sube=float(m.predict_proba(fila)[0]),
        umbral=float(m.umbral),
    )


def resumen_proximo_cierre(art: Artefactos, modelo: str) -> pd.DataFrame:
    """Una fila por accion: probabilidad de que suba en el siguiente cierre."""
    ultimas = art.features.sort_values("fecha").groupby("ticker").tail(1)
    ultimas = ultimas.set_index("ticker").loc[art.tickers].reset_index()
    m = art.modelos[modelo]
    prob = m.predict_proba(ultimas)
    cierres = art.precios.set_index(["ticker", "fecha"])["cierre"]
    ultimas_cierres = [
        cierres.loc[(t, f)]
        for t, f in zip(ultimas["ticker"], ultimas["fecha"], strict=True)
    ]
    return pd.DataFrame(
        {
            "ticker": ultimas["ticker"],
            "sector": ultimas["ticker"].map(SECTORES).fillna(""),
            "fecha": ultimas["fecha"],
            "cierre": ultimas_cierres,
            "cambio_1d": ultimas["ret_1d"],
            "prob_sube": prob,
            "senal": ["Sube" if p >= m.umbral else "Baja" for p in prob],
        }
    ).sort_values("prob_sube", ascending=False, ignore_index=True)


def serie_ticker(art: Artefactos, ticker: str, dias: int) -> pd.DataFrame:
    """Cierre y medias moviles de 20 y 50 dias de los ultimos `dias` dias habiles."""
    s = art.precios[art.precios["ticker"] == ticker].sort_values("fecha").copy()
    s["sma20"] = s["cierre"].rolling(20).mean()
    s["sma50"] = s["cierre"].rolling(50).mean()
    return s.tail(dias).reset_index(drop=True)


def historial_adelante(art: Artefactos, ticker: str, modelo: str) -> pd.DataFrame:
    """Predicciones del modelo final sobre dias posteriores al corte (no vistos)."""
    vacio = pd.DataFrame(
        columns=["fecha", "prob_sube", "prediccion", "real", "acierto", "cierre"]
    )
    if art.adelante.empty or modelo not in art.adelante.columns:
        return vacio
    h = art.adelante[art.adelante["ticker"] == ticker].sort_values("fecha")
    if h.empty:
        return vacio
    umbral = art.modelos[modelo].umbral
    sube = h[modelo] >= umbral
    cierres = art.precios[art.precios["ticker"] == ticker].set_index("fecha")["cierre"]
    return pd.DataFrame(
        {
            "fecha": h["fecha"],
            "prob_sube": h[modelo],
            "prediccion": sube.map({True: "Sube", False: "Baja"}),
            "real": h["y"].map({1: "Subió", 0: "Bajó"}),
            "acierto": sube.astype(int) == h["y"],
            "cierre": h["fecha"].map(cierres),
        }
    ).reset_index(drop=True)


def tabla_comparacion(art: Artefactos) -> pd.DataFrame:
    """Metricas del baseline y de los modelos de la app, listas para mostrar."""
    filas = []
    for nombre, m in art.metricas["modelos"].items():
        if nombre != "dummy" and nombre not in MODELOS_APP:
            continue
        adelante = m.get("adelante") or {}
        filas.append(
            {
                "modelo": NOMBRES_MODELO.get(nombre, nombre),
                "clave": nombre,
                "auc_cv": m["cv"]["auc"],
                "auc_cv_std": m["cv"]["auc_std"],
                "f1_cv": m["cv"]["f1"],
                "auc_holdout": m["holdout"]["auc"],
                "f1_holdout": m["holdout"]["f1"],
                "auc_adelante": adelante.get("auc"),
                "n_adelante": adelante.get("n"),
            }
        )
    return pd.DataFrame(filas)
