import logging

import pytest

from prediccion_acciones.data import utilidades
from prediccion_acciones.data.utilidades import cronometro, medir_tiempo, reintentar


@pytest.fixture(autouse=True)
def sin_esperas(monkeypatch):
    """Las pruebas no deben dormir de verdad."""
    esperas = []
    monkeypatch.setattr(utilidades.time, "sleep", esperas.append)
    return esperas


def test_reintentar_devuelve_el_resultado_si_no_falla():
    @reintentar(intentos=3)
    def siempre_bien():
        return 42

    assert siempre_bien() == 42


def test_reintentar_se_recupera_de_fallas_pasajeras(sin_esperas):
    llamadas = []

    @reintentar(intentos=3, espera=1.0, factor=2.0)
    def falla_dos_veces():
        llamadas.append(1)
        if len(llamadas) < 3:
            raise ConnectionError("sin red")
        return "listo"

    assert falla_dos_veces() == "listo"
    assert len(llamadas) == 3
    assert sin_esperas == [1.0, 2.0]  # espera exponencial


def test_reintentar_lanza_la_ultima_excepcion():
    @reintentar(intentos=2)
    def siempre_falla():
        raise ConnectionError("sin red")

    with pytest.raises(ConnectionError):
        siempre_falla()


def test_reintentar_no_atrapa_excepciones_no_listadas():
    llamadas = []

    @reintentar(intentos=3, excepciones=(ConnectionError,))
    def error_de_programa():
        llamadas.append(1)
        raise KeyError("x")

    with pytest.raises(KeyError):
        error_de_programa()
    assert len(llamadas) == 1


def test_reintentar_exige_al_menos_un_intento():
    with pytest.raises(ValueError):
        reintentar(intentos=0)


def test_reintentar_conserva_el_nombre_de_la_funcion():
    @reintentar()
    def mi_funcion():
        return 1

    assert mi_funcion.__name__ == "mi_funcion"


def test_medir_tiempo_devuelve_el_resultado_y_registra(caplog):
    @medir_tiempo
    def suma(a, b):
        return a + b

    with caplog.at_level(logging.DEBUG):
        assert suma(2, 3) == 5
    assert "suma tardo" in caplog.text


def test_cronometro_registra_inicio_y_fin(caplog):
    with caplog.at_level(logging.INFO), cronometro("prueba"):
        pass
    assert "Inicia: prueba" in caplog.text
    assert "Termina: prueba" in caplog.text


def test_cronometro_registra_el_fin_aunque_falle(caplog):
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError), cronometro("x"):
        raise RuntimeError("boom")
    assert "Termina: x" in caplog.text
