"""Figuras del README: exploracion, curvas de aprendizaje y resultados.

Lee los archivos que deja el entrenamiento en `models/` y escribe PNG en
`reports/`. Se corre con `python -m prediccion_acciones.reportes`.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve

from prediccion_acciones.config import RAIZ_PROYECTO, Config
from prediccion_acciones.logging_config import configurar_logging

logger = logging.getLogger(__name__)

NOMBRES = {
    "logistica": "Regresión logística",
    "boosting": "Gradient boosting",
    "pytorch": "MLP (PyTorch)",
}
COLORES = {"logistica": "tab:blue", "boosting": "tab:green", "pytorch": "tab:red"}
TICKERS_MUESTRA = ("AMD", "KOF", "LMT", "IBM")
RANGO_RETORNO = 0.08


def figura_exploracion(precios: pd.DataFrame) -> Figure:
    """Histograma de los retornos diarios de unas acciones de muestra."""
    retornos = precios.sort_values(["ticker", "fecha"]).assign(
        retorno=lambda t: t.groupby("ticker")["cierre"].pct_change()
    )
    fig = Figure(figsize=(9, 4.5), layout="constrained")
    ax = fig.subplots()
    for ticker in TICKERS_MUESTRA:
        r = retornos.loc[retornos["ticker"] == ticker, "retorno"].dropna()
        if len(r):
            ax.hist(
                r,
                bins=80,
                alpha=0.5,
                label=ticker,
                range=(-RANGO_RETORNO, RANGO_RETORNO),
            )
    ax.set_title(f"Distribución de retornos diarios (recortada a ±{RANGO_RETORNO:.0%})")
    ax.set_xlabel("Retorno diario")
    ax.set_ylabel("Número de días")
    ax.legend(title="Acción")
    return fig


def figura_curvas(historia: pd.DataFrame) -> Figure:
    """Curvas de aprendizaje de la red: pérdida y AUC de validación por época."""
    mejor = historia.loc[historia["mejor"].astype(bool), "epoca"]
    epoca_mejor = int(mejor.iloc[0]) if len(mejor) else None
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)

    ax1.plot(historia["epoca"], historia["perdida_train"], label="Entrenamiento")
    ax1.plot(historia["epoca"], historia["perdida_val"], label="Validación")
    ax1.set_title("Pérdida por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida (entropía cruzada binaria)")

    ax2.plot(
        historia["epoca"], historia["auc_val"], color="tab:green", label="Validación"
    )
    ax2.axhline(0.5, color="k", linestyle=":", label="Azar (0.50)")
    ax2.set_title("AUC en validación por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("AUC-ROC")

    for ax in (ax1, ax2):
        if epoca_mejor is not None:
            ax.axvline(
                epoca_mejor,
                color="tab:red",
                linestyle="--",
                label=f"Mejor época ({epoca_mejor}): pesos restaurados",
            )
        ax.legend()
    return fig


def figura_resultados(holdout: pd.DataFrame, metricas: dict) -> Figure:
    """Curva ROC del holdout y exactitud de cada modelo contra el piso."""
    modelos = [m for m in NOMBRES if m in holdout.columns]
    y = holdout["y"].to_numpy()
    fig = Figure(figsize=(11, 4.8), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)

    for m in modelos:
        fpr, tpr, _ = roc_curve(y, holdout[m])
        auc = roc_auc_score(y, holdout[m])
        ax1.plot(fpr, tpr, color=COLORES[m], label=f"{NOMBRES[m]} (AUC {auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", label="Azar (AUC 0.500)")
    ax1.set_title("Curva ROC en el holdout")
    ax1.set_xlabel("Tasa de falsos positivos")
    ax1.set_ylabel("Tasa de verdaderos positivos")
    ax1.legend(loc="lower right")

    piso = metricas["modelos"]["dummy"]["holdout"]["exactitud"]
    valores = [metricas["modelos"][m]["holdout"]["exactitud"] for m in modelos]
    barras = ax2.bar(
        [NOMBRES[m] for m in modelos], valores, color=[COLORES[m] for m in modelos]
    )
    ax2.bar_label(barras, fmt="%.3f")
    ax2.axhline(
        piso, color="k", linestyle="--", label=f"Piso: siempre 'sube' ({piso:.3f})"
    )
    ax2.set_ylim(0, 0.7)
    ax2.set_title("Exactitud en el holdout frente al piso")
    ax2.set_xlabel("Modelo")
    ax2.set_ylabel("Exactitud (umbral de validación)")
    ax2.legend(loc="upper right")
    return fig


def generar_reportes(directorio_modelos: Path, directorio_reportes: Path) -> list[Path]:
    """Dibuja las tres figuras y devuelve las rutas escritas."""
    directorio_reportes.mkdir(parents=True, exist_ok=True)
    precios = pd.read_csv(directorio_modelos / "precios_app.csv", parse_dates=["fecha"])
    historia = pd.read_csv(directorio_modelos / "historia_pytorch.csv")
    holdout = pd.read_csv(directorio_modelos / "prediccion_holdout.csv")
    with (directorio_modelos / "metricas.json").open(encoding="utf-8") as archivo:
        metricas = json.load(archivo)

    figuras = {
        "01_exploracion_retornos.png": figura_exploracion(precios),
        "02_curvas_aprendizaje.png": figura_curvas(historia),
        "03_resultados_holdout.png": figura_resultados(holdout, metricas),
    }
    rutas = []
    for nombre, figura in figuras.items():
        ruta = directorio_reportes / nombre
        figura.savefig(ruta, dpi=130)
        logger.info("Figura escrita en %s", ruta)
        rutas.append(ruta)
    return rutas


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="Dibuja las figuras del README.").parse_args(
        argv
    )
    configurar_logging()
    cfg = Config.desde_yaml()
    generar_reportes(
        RAIZ_PROYECTO / cfg.rutas.modelos, RAIZ_PROYECTO / cfg.rutas.reportes
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
