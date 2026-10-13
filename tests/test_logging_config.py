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


def test_lee_el_nivel_desde_un_archivo_env(monkeypatch, tmp_path):
    # setenv y luego delenv: al terminar la prueba, pytest deja el entorno como estaba
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("LOG_LEVEL")
    env = tmp_path / ".env"
    env.write_text("LOG_LEVEL=ERROR\n", encoding="utf-8")
    configurar_logging(ruta_env=env)
    assert logging.getLogger().level == logging.ERROR


def test_una_variable_del_entorno_gana_sobre_el_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    env = tmp_path / ".env"
    env.write_text("LOG_LEVEL=ERROR\n", encoding="utf-8")
    configurar_logging(ruta_env=env)
    assert logging.getLogger().level == logging.WARNING
