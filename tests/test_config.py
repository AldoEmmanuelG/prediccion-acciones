from datetime import date

import pytest
import yaml

from prediccion_acciones.config import RUTA_CONFIG, Config, ErrorConfiguracion


@pytest.fixture
def crudo():
    with RUTA_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _escribir(tmp_path, datos):
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(datos), encoding="utf-8")
    return ruta


def test_la_configuracion_real_carga():
    config = Config.desde_yaml()
    assert len(config.datos.tickers_objetivo) == 10
    assert config.validacion.fecha_corte == date(2026, 9, 30)
    assert config.modelos.pytorch.capas_ocultas == (32, 16)


def test_todos_los_tickers_une_objetivo_y_contexto():
    datos = Config.desde_yaml().datos
    assert datos.todos_los_tickers == datos.tickers_objetivo + datos.tickers_contexto


def test_el_objeto_es_inmutable():
    config = Config.desde_yaml()
    with pytest.raises(AttributeError):
        config.datos.horizonte_dias = 5


def test_archivo_inexistente(tmp_path):
    with pytest.raises(ErrorConfiguracion, match="No existe"):
        Config.desde_yaml(tmp_path / "no_esta.yaml")


def test_falta_una_clave(tmp_path, crudo):
    del crudo["validacion"]["gap_dias"]
    with pytest.raises(ErrorConfiguracion, match="gap_dias"):
        Config.desde_yaml(_escribir(tmp_path, crudo))


def test_ticker_repetido(tmp_path, crudo):
    crudo["datos"]["tickers_contexto"].append(crudo["datos"]["tickers_objetivo"][0])
    with pytest.raises(ErrorConfiguracion, match="repetidos"):
        Config.desde_yaml(_escribir(tmp_path, crudo))


def test_fecha_invalida(tmp_path, crudo):
    crudo["validacion"]["fecha_corte"] = "30/09/2026"
    with pytest.raises(ErrorConfiguracion, match="fecha ISO"):
        Config.desde_yaml(_escribir(tmp_path, crudo))


def test_corte_anterior_al_inicio(tmp_path, crudo):
    crudo["validacion"]["fecha_corte"] = "2010-01-01"
    with pytest.raises(ErrorConfiguracion, match="posterior"):
        Config.desde_yaml(_escribir(tmp_path, crudo))


def test_fraccion_validacion_fuera_de_rango(tmp_path, crudo):
    crudo["modelos"]["pytorch"]["fraccion_validacion"] = 1.5
    with pytest.raises(ErrorConfiguracion, match="fraccion_validacion"):
        Config.desde_yaml(_escribir(tmp_path, crudo))
