import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandera")

from prediccion_acciones.config import RUTA_CONFIG, Config  # noqa: E402
from prediccion_acciones.entrenamiento import (  # noqa: E402
    ejecutar_pipeline,
    obtener_precios,
)


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.desde_yaml(RUTA_CONFIG)


@pytest.fixture(scope="module")
def fuente(cfg):
    """Imita a Yahoo: caminatas aleatorias hasta unos dias despues del corte."""
    fechas = pd.bdate_range(cfg.datos.fecha_inicio, "2026-10-09")
    rng = np.random.default_rng(0)
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
    return lambda pedidos, inicio: tabla


@pytest.fixture(scope="module")
def corrida(cfg, fuente, tmp_path_factory):
    salida = tmp_path_factory.mktemp("modelos")
    resultado = ejecutar_pipeline(
        cfg,
        directorio_modelos=salida,
        modelos=("dummy", "logistica"),
        fuente=fuente,
        usar_cache=False,
        usar_mlflow=False,
        n_jobs=1,
    )
    return resultado, salida


def test_las_tres_particiones_tienen_filas(corrida):
    resultado, _ = corrida
    assert all(n > 0 for n in resultado.filas.values())


def test_guarda_modelo_metricas_y_tablas_para_la_app(corrida):
    _, salida = corrida
    for archivo in (
        "logistica.joblib",
        "metricas.json",
        "prueba_adelante.csv",
        "prediccion_holdout.csv",
        "precios_app.csv",
    ):
        assert (salida / archivo).exists(), archivo
    assert not (salida / "dummy.joblib").exists()


def test_el_reporte_trae_cv_holdout_y_adelante(corrida, cfg):
    _, salida = corrida
    reporte = json.loads((salida / "metricas.json").read_text(encoding="utf-8"))
    assert reporte["fecha_corte"] == str(cfg.validacion.fecha_corte)
    modelo = reporte["modelos"]["logistica"]
    assert {"cv", "holdout", "adelante", "umbral", "hiperparametros"} <= set(modelo)
    assert modelo["holdout"]["n"] > 0


def test_el_dummy_queda_en_medio_y_la_logistica_cerca(corrida):
    resultado, _ = corrida
    por_nombre = {r.nombre: r for r in resultado.modelos}
    assert por_nombre["dummy"].cv_auc == 0.5
    assert 0.4 < por_nombre["logistica"].cv_auc < 0.6  # datos sin senal


def test_la_prueba_hacia_adelante_es_posterior_al_corte(corrida, cfg):
    _, salida = corrida
    adelante = pd.read_csv(salida / "prueba_adelante.csv", parse_dates=["fecha"])
    assert (adelante["fecha"] > pd.Timestamp(cfg.validacion.fecha_corte)).all()
    assert "logistica" in adelante.columns


def test_el_cache_guarda_y_reutiliza_los_precios(cfg, fuente, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr("prediccion_acciones.entrenamiento.RAIZ_PROYECTO", tmp_path)
    primera = obtener_precios(cfg, fuente, usar_cache=True)

    def no_debe_llamarse(*args, **kwargs):
        raise AssertionError("debio usar el cache")

    segunda = obtener_precios(cfg, no_debe_llamarse, usar_cache=True)
    assert len(segunda) == len(primera)


def test_con_mlflow_queda_una_corrida_por_modelo(cfg, fuente, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    ejecutar_pipeline(
        cfg,
        directorio_modelos=tmp_path / "modelos",
        modelos=("dummy", "logistica"),
        fuente=fuente,
        usar_cache=False,
        usar_mlflow=True,
        tracking_uri=uri,
        directorio_artefactos_mlflow=tmp_path / "artefactos",
        n_jobs=1,
    )
    mlflow.set_tracking_uri(uri)
    corridas = mlflow.search_runs(experiment_names=[cfg.rutas.mlflow_experimento])
    assert len(corridas) == 2
    assert "metrics.cv_auc" in corridas.columns
    assert "metrics.holdout_auc" in corridas.columns


def test_el_holdout_guarda_las_probabilidades_de_cada_modelo(corrida):
    _, salida = corrida
    holdout = pd.read_csv(salida / "prediccion_holdout.csv")
    assert {"fecha", "ticker", "y", "dummy", "logistica"} <= set(holdout.columns)
    assert holdout[["dummy", "logistica"]].notna().all().all()
