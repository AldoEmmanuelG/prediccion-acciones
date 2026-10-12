from datetime import date

import numpy as np
import pandas as pd
import pytest

from prediccion_acciones.data import utilidades
from prediccion_acciones.data.descarga import DescargadorPrecios, ErrorDescarga

TICKERS = ["AAA", "BBB"]
INICIO = date(2024, 1, 1)


@pytest.fixture(autouse=True)
def sin_esperas(monkeypatch):
    monkeypatch.setattr(utilidades.time, "sleep", lambda s: None)


def _tabla_yahoo(tickers=TICKERS, dias=5):
    """Imita la tabla de yfinance: columnas (campo, ticker)."""
    indice = pd.bdate_range("2024-01-02", periods=dias)
    columnas = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    valores = np.arange(dias * len(columnas), dtype=float).reshape(dias, -1) + 1
    return pd.DataFrame(valores, index=indice, columns=columnas)


def test_descarga_devuelve_cierre_y_volumen_por_ticker():
    desc = DescargadorPrecios(TICKERS, INICIO, fuente=lambda t, i: _tabla_yahoo())
    precios = desc.descargar()
    assert list(precios.cierre.columns) == TICKERS
    assert list(precios.volumen.columns) == TICKERS
    assert len(precios.cierre) == 5


def test_el_orden_de_columnas_sigue_al_pedido():
    desc = DescargadorPrecios(
        ["BBB", "AAA"], INICIO, fuente=lambda t, i: _tabla_yahoo()
    )
    assert list(desc.descargar().cierre.columns) == ["BBB", "AAA"]


def test_ticker_faltante_lanza_error():
    desc = DescargadorPrecios(
        ["AAA", "ZZZ"], INICIO, fuente=lambda t, i: _tabla_yahoo()
    )
    with pytest.raises(ErrorDescarga, match="ZZZ"):
        desc.descargar()


def test_tabla_vacia_lanza_error_tras_reintentar():
    llamadas = []

    def fuente(tickers, inicio):
        llamadas.append(1)
        return pd.DataFrame()

    desc = DescargadorPrecios(TICKERS, INICIO, fuente=fuente, intentos=3)
    with pytest.raises(ErrorDescarga, match="vacia"):
        desc.descargar()
    assert len(llamadas) == 3


def test_se_recupera_de_una_falla_de_red():
    llamadas = []

    def fuente(tickers, inicio):
        llamadas.append(1)
        if len(llamadas) == 1:
            raise ConnectionError("sin red")
        return _tabla_yahoo()

    desc = DescargadorPrecios(TICKERS, INICIO, fuente=fuente, intentos=3)
    assert len(desc.descargar().cierre) == 5
    assert len(llamadas) == 2


def test_a_largo_tiene_una_fila_por_fecha_y_ticker():
    desc = DescargadorPrecios(TICKERS, INICIO, fuente=lambda t, i: _tabla_yahoo())
    largo = desc.descargar().a_largo()
    assert list(largo.columns) == ["fecha", "ticker", "cierre", "volumen"]
    assert len(largo) == 5 * 2
    assert not largo.duplicated(["fecha", "ticker"]).any()
    fila = largo[(largo["ticker"] == "AAA") & (largo["fecha"] == "2024-01-02")]
    assert fila["cierre"].iloc[0] == 1.0
