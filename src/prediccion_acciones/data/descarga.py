"""Descarga de precios y volumenes. No sabe de modelos ni de la app."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from prediccion_acciones.data.utilidades import cronometro, medir_tiempo, reintentar

logger = logging.getLogger(__name__)

# Una "fuente" recibe tickers y fecha de inicio y devuelve la tabla de Yahoo
# (columnas de dos niveles: campo y ticker). Cambiarla permite probar sin red.
Fuente = Callable[[Sequence[str], date], pd.DataFrame]


class ErrorDescarga(RuntimeError):
    """La descarga no devolvio los datos esperados."""


@dataclass(frozen=True)
class Precios:
    """Cierres ajustados y volumenes, una columna por ticker y un dia por fila."""

    cierre: pd.DataFrame
    volumen: pd.DataFrame

    def a_largo(self) -> pd.DataFrame:
        """Tabla en formato largo: una fila por (fecha, ticker)."""
        cierre = self.cierre.rename_axis("fecha").stack(future_stack=True)
        volumen = self.volumen.rename_axis("fecha").stack(future_stack=True)
        largo = pd.concat({"cierre": cierre, "volumen": volumen}, axis=1)
        largo = largo.rename_axis(["fecha", "ticker"]).reset_index()
        return largo.sort_values(["ticker", "fecha"]).reset_index(drop=True)


def fuente_yfinance(tickers: Sequence[str], inicio: date) -> pd.DataFrame:
    """Fuente real. `yfinance` se importa aqui para que probar el resto no lo exija.

    `auto_adjust=True` corrige splits y dividendos: sin eso aparecerian saltos
    falsos en los retornos.
    """
    import yfinance as yf

    return yf.download(
        list(tickers), start=inicio.isoformat(), auto_adjust=True, progress=False
    )


class DescargadorPrecios:
    """Baja precios de una lista de tickers, con reintentos y validacion basica."""

    def __init__(
        self,
        tickers: Sequence[str],
        fecha_inicio: date,
        fuente: Fuente = fuente_yfinance,
        intentos: int = 3,
        espera: float = 2.0,
    ) -> None:
        self.tickers = tuple(tickers)
        self.fecha_inicio = fecha_inicio
        self._fuente = fuente
        # El decorador necesita los parametros desde el inicio; por eso se
        # aplica sobre el metodo ligado a esta instancia.
        self._pedir = reintentar(intentos=intentos, espera=espera)(self._pedir_una_vez)

    def _pedir_una_vez(self) -> pd.DataFrame:
        crudo = self._fuente(self.tickers, self.fecha_inicio)
        if crudo is None or crudo.empty:
            raise ErrorDescarga("La fuente devolvio una tabla vacia")
        return crudo

    @medir_tiempo
    def descargar(self) -> Precios:
        """Devuelve `Precios` o lanza `ErrorDescarga` si falta algun ticker."""
        with cronometro(f"descarga de {len(self.tickers)} tickers"):
            crudo = self._pedir()
        try:
            cierre, volumen = crudo["Close"], crudo["Volume"]
        except KeyError as exc:
            raise ErrorDescarga(f"La tabla no trae la columna {exc}") from exc

        faltantes = [t for t in self.tickers if t not in cierre.columns]
        if faltantes:
            raise ErrorDescarga(f"No llegaron datos para: {faltantes}")

        cierre = cierre[list(self.tickers)].sort_index()
        volumen = volumen[list(self.tickers)].sort_index()
        logger.info(
            "Descargados %d dias, de %s a %s",
            len(cierre),
            cierre.index.min().date(),
            cierre.index.max().date(),
        )
        return Precios(cierre=cierre, volumen=volumen)
