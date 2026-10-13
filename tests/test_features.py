import numpy as np
import pandas as pd
import pytest

from prediccion_acciones.config import FeaturesConfig
from prediccion_acciones.features.ingenieria import (
    calcular_etiqueta,
    calcular_features,
    construir_panel,
    nombres_features,
    rsi,
)

CFG = FeaturesConfig(
    ventanas_retorno=(1, 2, 5, 10),
    ventanas_rolling=(5, 10, 20),
    rsi_periodo=14,
    ventanas_contexto=(1, 5),
)


def _precios(dias: int = 80, tickers=("AAA", "BBB"), semilla: int = 0):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2026-01-05", periods=dias)
    filas = []
    for t in tickers:
        cierre = 100 * np.cumprod(1 + rng.normal(0, 0.01, dias))
        volumen = rng.uniform(1_000, 2_000, dias)
        filas.append(
            pd.DataFrame(
                {"fecha": fechas, "ticker": t, "cierre": cierre, "volumen": volumen}
            )
        )
    return pd.concat(filas, ignore_index=True)


def test_nombres_son_13_y_unicos():
    nombres = nombres_features(CFG)
    assert len(nombres) == 13
    assert len(set(nombres)) == 13
    assert "rsi_14" in nombres and "dist_media_20d" in nombres


def test_features_tienen_las_columnas_y_ningun_nulo():
    f = calcular_features(_precios(), CFG)
    assert list(f.columns) == ["fecha", "ticker", *nombres_features(CFG)]
    assert not f.isna().any().any()
    assert np.isfinite(f[nombres_features(CFG)].to_numpy()).all()


def test_ret_1d_coincide_con_el_cambio_de_precio():
    precios = _precios()
    f = calcular_features(precios, CFG)
    fila = f[f["ticker"] == "AAA"].iloc[10]
    serie = precios[precios["ticker"] == "AAA"].set_index("fecha")["cierre"]
    i = serie.index.get_loc(fila["fecha"])
    esperado = serie.iloc[i] / serie.iloc[i - 1] - 1
    assert fila["ret_1d"] == pytest.approx(esperado)


def test_etiqueta_compara_con_el_dia_siguiente():
    precios = _precios()
    y = calcular_etiqueta(precios, horizonte=1)
    serie = precios[precios["ticker"] == "AAA"].set_index("fecha")["cierre"]
    fila = y[y["ticker"] == "AAA"].iloc[5]
    i = serie.index.get_loc(fila["fecha"])
    assert fila["y"] == int(serie.iloc[i + 1] > serie.iloc[i])


def test_ultimo_dia_de_cada_ticker_no_tiene_etiqueta():
    precios = _precios()
    y = calcular_etiqueta(precios, horizonte=1)
    ultimo = precios["fecha"].max()
    assert ultimo not in set(y["fecha"])


def test_horizonte_invalido_falla():
    with pytest.raises(ValueError):
        calcular_etiqueta(_precios(), horizonte=0)


def test_features_no_usan_el_futuro():
    precios = _precios()
    corte = precios["fecha"].iloc[40]
    completas = calcular_features(precios, CFG)
    truncadas = calcular_features(precios[precios["fecha"] <= corte], CFG)
    comunes = completas[completas["fecha"] <= corte].reset_index(drop=True)
    pd.testing.assert_frame_equal(comunes, truncadas)


def test_panel_tiene_etiqueta_y_pierde_el_ultimo_dia():
    precios = _precios()
    panel = construir_panel(precios, CFG, horizonte=1)
    assert set(panel["y"].unique()) <= {0, 1}
    assert panel["fecha"].max() < precios["fecha"].max()
    assert list(panel["fecha"]) == sorted(panel["fecha"])


def test_falta_una_columna_falla():
    with pytest.raises(ValueError, match="columnas"):
        calcular_features(_precios().drop(columns="volumen"), CFG)


def test_rsi_de_una_serie_que_solo_sube_es_100():
    sube = pd.Series(np.arange(1.0, 40.0))
    assert rsi(sube, 14).iloc[-1] == pytest.approx(100.0)
