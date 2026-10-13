import pandas as pd
import pytest

pytest.importorskip("pandera")

from prediccion_acciones.data.esquemas import (  # noqa: E402
    ErrorValidacion,
    validar_precios,
)


@pytest.fixture
def tabla_valida() -> pd.DataFrame:
    fechas = pd.date_range("2026-01-05", periods=4, freq="B")
    filas = [
        {"fecha": f, "ticker": t, "cierre": 100.0 + i, "volumen": 1000.0 + i}
        for t in ("AAA", "BBB")
        for i, f in enumerate(fechas)
    ]
    return pd.DataFrame(filas)


def test_tabla_valida_pasa(tabla_valida):
    resultado = validar_precios(tabla_valida, ["AAA", "BBB"])
    assert len(resultado) == len(tabla_valida)


def test_precio_no_positivo_falla(tabla_valida):
    tabla_valida.loc[0, "cierre"] = 0.0
    with pytest.raises(ErrorValidacion):
        validar_precios(tabla_valida)


def test_volumen_negativo_falla(tabla_valida):
    tabla_valida.loc[1, "volumen"] = -5.0
    with pytest.raises(ErrorValidacion):
        validar_precios(tabla_valida)


def test_filas_duplicadas_fallan(tabla_valida):
    duplicada = pd.concat([tabla_valida, tabla_valida.iloc[[0]]], ignore_index=True)
    with pytest.raises(ErrorValidacion):
        validar_precios(duplicada)


def test_nulos_fallan(tabla_valida):
    tabla_valida.loc[2, "cierre"] = None
    with pytest.raises(ErrorValidacion):
        validar_precios(tabla_valida)


def test_ticker_no_permitido_falla(tabla_valida):
    with pytest.raises(ErrorValidacion):
        validar_precios(tabla_valida, ["AAA"])


def test_columna_extra_falla(tabla_valida):
    tabla_valida["sobrante"] = 1
    with pytest.raises(ErrorValidacion):
        validar_precios(tabla_valida)


def test_salto_enorme_solo_avisa(tabla_valida):
    tabla_valida.loc[1, "cierre"] = 400.0
    with pytest.warns(UserWarning):
        resultado = validar_precios(tabla_valida)
    assert len(resultado) == len(tabla_valida)
