import logging

from prediccion_acciones.logging_config import configurar_logging


def test_respeta_el_nivel_pedido():
    configurar_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_lee_la_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    configurar_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_un_nivel_invalido_cae_a_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configurar_logging("RUIDOSO")
    assert logging.getLogger().level == logging.INFO
