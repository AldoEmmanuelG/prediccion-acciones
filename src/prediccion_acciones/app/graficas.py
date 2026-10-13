"""Graficas de la app (Matplotlib). Cada funcion devuelve una `Figure`.

Se dibujan con fondo transparente y texto claro para el tema oscuro fijo de la app
(`.streamlit/config.toml`). Todas llevan titulo, etiquetas de ejes y leyenda. Los
colores distinguen subida y bajada por tono y por forma (flechas), no solo por color.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

TEXTO = "#e5eef0"
APAGADO = "#8fa5ab"
REJILLA = "#2b3d42"
ACENTO = "#4fd1c5"
VERDE = "#4ade80"
ROJO = "#fb923c"
AMARILLO = "#facc15"
MORADO = "#c4b5fd"

# Margen de error aproximado del AUC con ~1,300 filas (ver "Acerca del modelo").
BANDA_RUIDO = 0.03


def _figura(ancho: float = 8, alto: float = 4) -> tuple[Figure, Axes]:
    """Figura transparente con texto claro, ejes sin marco superior ni derecho."""
    fig = Figure(figsize=(ancho, alto))
    fig.patch.set_facecolor("none")
    ax = fig.subplots()
    ax.set_facecolor("none")
    ax.tick_params(colors=APAGADO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.grid(color=REJILLA, alpha=0.8, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _leyenda(ax: Axes, **kwargs) -> None:
    leyenda = ax.legend(fontsize=8, frameon=False, **kwargs)
    for texto in leyenda.get_texts():
        texto.set_color(TEXTO)


def _titulos(ax: Axes, titulo: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(titulo, color=TEXTO, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=APAGADO)
    ax.set_ylabel(ylabel, color=APAGADO)


def grafica_precio(
    serie: pd.DataFrame,
    ticker: str,
    sma20: bool,
    sma50: bool,
    marcas: pd.DataFrame | None = None,
) -> Figure:
    """Cierre con medias moviles opcionales y flechas en las predicciones no vistas."""
    fig, ax = _figura(9, 4)
    ax.plot(serie["fecha"], serie["cierre"], color=ACENTO, linewidth=2, label="Cierre")
    if sma20:
        ax.plot(
            serie["fecha"],
            serie["sma20"],
            color=AMARILLO,
            linestyle="--",
            linewidth=1.4,
            label="Media 20 días",
        )
    if sma50:
        ax.plot(
            serie["fecha"],
            serie["sma50"],
            color=MORADO,
            linestyle=":",
            linewidth=1.6,
            label="Media 50 días",
        )
    if marcas is not None and not marcas.empty:
        sube = marcas[marcas["prediccion"] == "Sube"]
        baja = marcas[marcas["prediccion"] == "Baja"]
        ax.scatter(
            sube["fecha"],
            sube["cierre"],
            marker="^",
            s=70,
            color=VERDE,
            zorder=3,
            label="Predijo sube",
        )
        ax.scatter(
            baja["fecha"],
            baja["cierre"],
            marker="v",
            s=70,
            color=ROJO,
            zorder=3,
            label="Predijo baja",
        )
    _titulos(ax, f"{ticker}: cierre diario", "Fecha", "Precio de cierre (USD)")
    _leyenda(ax, loc="lower right", bbox_to_anchor=(1, 1), ncol=5)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def barras_modelos(tabla: pd.DataFrame) -> Figure:
    """AUC en holdout por modelo; la linea marca el azar (0.50) y la banda el ruido."""
    fig, ax = _figura(8, 3.6)
    ax.grid(axis="y", visible=False)
    y = range(len(tabla))
    colores = [APAGADO if c == "dummy" else ACENTO for c in tabla["clave"]]
    ax.barh(list(y), tabla["auc_holdout"], height=0.55, color=colores, alpha=0.9)
    for i, valor in zip(y, tabla["auc_holdout"], strict=True):
        ax.text(
            valor - 0.01,
            i,
            f"{valor:.4f}",
            ha="right",
            va="center",
            color="#0f1719",
            fontsize=9,
            fontweight="bold",
        )
    ax.axvspan(
        0.5 - BANDA_RUIDO,
        0.5 + BANDA_RUIDO,
        color=ACENTO,
        alpha=0.15,
        label=f"Zona de ruido (± {BANDA_RUIDO:.2f})",
    )
    ax.axvline(0.5, color=TEXTO, linestyle="--", linewidth=1.4, label="Azar = 0.50")
    ax.set_yticks(list(y), tabla["modelo"], color=TEXTO)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.65)
    _titulos(
        ax,
        "AUC en holdout: todos quedan al nivel del azar",
        "AUC-ROC en holdout",
        "Modelo",
    )
    _leyenda(ax, loc="upper right")
    fig.tight_layout()
    return fig
