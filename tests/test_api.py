import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandera")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from prediccion_acciones.api.esquemas import NombreModelo  # noqa: E402
from prediccion_acciones.api.main import crear_app  # noqa: E402
from prediccion_acciones.app import servicio  # noqa: E402
from prediccion_acciones.config import RUTA_CONFIG, Config  # noqa: E402
from prediccion_acciones.entrenamiento import ejecutar_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def directorio(tmp_path_factory):
    """Entrena con datos sinteticos (solo la logística) y deja los archivos."""
    cfg = Config.desde_yaml(RUTA_CONFIG)
    fechas = pd.bdate_range(cfg.datos.fecha_inicio, "2026-10-09")
    rng = np.random.default_rng(2)
    tickers = list(cfg.datos.tickers_objetivo)
    cierre = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (len(fechas), len(tickers))), axis=0),
        index=fechas,
        columns=tickers,
    )
    volumen = pd.DataFrame(
        rng.uniform(1e5, 2e5, cierre.shape), index=fechas, columns=tickers
    )
    tabla = pd.concat({"Close": cierre, "Volume": volumen}, axis=1)
    salida = tmp_path_factory.mktemp("api")
    ejecutar_pipeline(
        cfg,
        directorio_modelos=salida,
        modelos=("dummy", "logistica"),
        fuente=lambda pedidos, inicio: tabla,
        usar_cache=False,
        usar_mlflow=False,
        n_jobs=1,
    )
    return salida


@pytest.fixture(scope="module")
def cliente(directorio):
    # `with` dispara el arranque del servicio, que es donde se carga el modelo.
    with TestClient(crear_app(directorio)) as c:
        yield c


def test_health_dice_que_modelo_cargo_y_de_que_fecha(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["modelos_cargados"] == ["logistica"]
    assert cuerpo["modelo_por_defecto"] == "logistica"
    assert cuerpo["entrenado_hasta"] == "2026-09-30"
    assert cuerpo["datos_hasta"] == "2026-10-09"
    assert len(cuerpo["tickers"]) == 10


def test_predict_cumple_el_contrato(cliente):
    r = cliente.post("/predict", json={"ticker": "AMD", "modelo": "logistica"})
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {
        "ticker",
        "modelo",
        "fecha_datos",
        "cierre",
        "probabilidad_sube",
        "umbral",
        "prediccion",
    }
    assert cuerpo["ticker"] == "AMD"
    assert 0 <= cuerpo["probabilidad_sube"] <= 1
    esperado = "sube" if cuerpo["probabilidad_sube"] >= cuerpo["umbral"] else "baja"
    assert cuerpo["prediccion"] == esperado
    assert cuerpo["fecha_datos"] == "2026-10-09"


def test_predict_usa_la_logistica_si_no_se_pide_modelo(cliente):
    r = cliente.post("/predict", json={"ticker": "ibm"})
    assert r.status_code == 200
    assert r.json()["modelo"] == "logistica"
    assert r.json()["ticker"] == "IBM"  # el ticker se normaliza a mayúsculas


def test_un_ticker_desconocido_da_422_con_las_opciones(cliente):
    r = cliente.post("/predict", json={"ticker": "ZZZZ"})
    assert r.status_code == 422
    assert "AMD" in r.json()["detail"]


def test_un_modelo_fuera_de_catalogo_da_422(cliente):
    r = cliente.post("/predict", json={"ticker": "AMD", "modelo": "magia"})
    assert r.status_code == 422


def test_un_modelo_valido_pero_no_cargado_da_422(cliente):
    # En este entrenamiento de prueba solo se hizo la logística.
    r = cliente.post("/predict", json={"ticker": "AMD", "modelo": "pytorch"})
    assert r.status_code == 422
    assert "pytorch" in r.json()["detail"]


def test_sin_ticker_da_422(cliente):
    assert cliente.post("/predict", json={}).status_code == 422


def test_el_modelo_se_carga_una_sola_vez(cliente):
    antes = cliente.app.state.artefactos
    cliente.post("/predict", json={"ticker": "AMD"})
    cliente.post("/predict", json={"ticker": "LMT"})
    assert cliente.app.state.artefactos is antes


def test_la_documentacion_automatica_esta_disponible(cliente):
    assert cliente.get("/docs").status_code == 200
    assert "/predict" in cliente.get("/openapi.json").json()["paths"]


def test_la_raiz_sirve_la_pagina_web_que_usa_la_api(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    # La página llama a los mismos endpoints que el contrato de la API.
    assert "/health" in r.text
    assert "/predict" in r.text


def test_la_pagina_web_no_aparece_en_la_documentacion_de_la_api(cliente):
    assert "/" not in cliente.get("/openapi.json").json()["paths"]


def test_resumen_ordena_las_10_acciones_de_mayor_a_menor(cliente):
    r = cliente.get("/resumen", params={"modelo": "logistica"})
    assert r.status_code == 200
    cuerpo = r.json()
    acciones = cuerpo["acciones"]
    assert len(acciones) == 10
    probs = [a["probabilidad_sube"] for a in acciones]
    assert probs == sorted(probs, reverse=True)
    for a in acciones:
        esperado = "sube" if a["probabilidad_sube"] >= a["umbral"] else "baja"
        assert a["prediccion"] == esperado
    assert cuerpo["fecha_prediccion"] > cuerpo["fecha_datos"]


def test_resumen_con_un_modelo_no_cargado_da_422(cliente):
    r = cliente.get("/resumen", params={"modelo": "pytorch"})
    assert r.status_code == 422


def test_series_da_precios_con_medias_y_los_dias_no_vistos(cliente):
    r = cliente.get("/series/amd", params={"dias": 60, "modelo": "logistica"})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["ticker"] == "AMD"  # se normaliza a mayúsculas
    assert len(cuerpo["precios"]) == 60
    assert all(p["sma20"] is not None for p in cuerpo["precios"])
    assert cuerpo["fecha_corte"] == "2026-09-30"
    no_vistos = cuerpo["dias_no_vistos"]
    assert len(no_vistos) > 0
    assert all(d["fecha"] > cuerpo["fecha_corte"] for d in no_vistos)
    for d in no_vistos:
        assert d["acierto"] == (d["prediccion"] == d["real"])


def test_series_valida_el_ticker_y_el_rango_de_dias(cliente):
    assert cliente.get("/series/ZZZZ").status_code == 422
    assert cliente.get("/series/AMD", params={"dias": 5}).status_code == 422
    assert cliente.get("/series/AMD", params={"dias": 9999}).status_code == 422


def test_modelos_compara_contra_el_piso(cliente):
    r = cliente.get("/modelos")
    assert r.status_code == 200
    cuerpo = r.json()
    claves = [m["clave"] for m in cuerpo["modelos"]]
    assert claves == ["dummy", "logistica"]
    dummy = cuerpo["modelos"][0]
    assert cuerpo["piso_exactitud"] == dummy["exactitud_holdout"]
    assert 0 < cuerpo["piso_exactitud"] < 1
    assert set(cuerpo["periodos"]) == {"entrenamiento", "holdout", "adelante"}


def test_los_modelos_del_esquema_son_los_de_la_app():
    from typing import get_args

    assert set(get_args(NombreModelo)) == set(servicio.MODELOS_APP)
