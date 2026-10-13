import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandera")

from prediccion_acciones.app import graficas, servicio  # noqa: E402
from prediccion_acciones.config import RUTA_CONFIG, Config  # noqa: E402
from prediccion_acciones.entrenamiento import ejecutar_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.desde_yaml(RUTA_CONFIG)


@pytest.fixture(scope="module")
def directorio(cfg, tmp_path_factory):
    """Corre el entrenamiento con datos sinteticos y deja sus archivos."""
    fechas = pd.bdate_range(cfg.datos.fecha_inicio, "2026-10-09")
    rng = np.random.default_rng(1)
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
    salida = tmp_path_factory.mktemp("app")
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
def art(directorio, cfg):
    return servicio.cargar_artefactos(directorio, cfg)


def test_carga_los_modelos_entrenados_y_omite_el_baseline(art):
    assert set(art.modelos) == {"logistica"}
    assert len(art.tickers) == 10


def test_sin_artefactos_el_error_dice_que_hacer(tmp_path, cfg):
    with pytest.raises(servicio.ErrorArtefactos, match="entrenamiento"):
        servicio.cargar_artefactos(tmp_path, cfg)


def test_resumen_trae_una_fila_por_accion_ordenada(art):
    r = servicio.resumen_proximo_cierre(art, "logistica")
    assert len(r) == 10
    assert r["prob_sube"].between(0, 1).all()
    assert set(r["senal"]) <= {"Sube", "Baja"}
    assert r["prob_sube"].is_monotonic_decreasing


def test_serie_respeta_los_dias_pedidos_y_trae_medias(art):
    s = servicio.serie_ticker(art, "AMD", 60)
    assert len(s) == 60
    assert not s[["sma20", "sma50"]].isna().any().any()


def test_historial_hacia_adelante_es_posterior_al_corte(art, cfg):
    h = servicio.historial_adelante(art, "AMD", "logistica")
    assert not h.empty
    assert (h["fecha"] > pd.Timestamp(cfg.validacion.fecha_corte)).all()
    assert set(h["prediccion"]) <= {"Sube", "Baja"}


def test_historial_de_un_modelo_sin_columna_esta_vacio(art):
    assert servicio.historial_adelante(art, "AMD", "boosting_inexistente").empty


def test_la_app_ofrece_solo_los_modelos_de_la_app(art):
    assert art.modelos_app == ["logistica"]


def test_tabla_de_comparacion_incluye_el_baseline(art):
    t = servicio.tabla_comparacion(art)
    assert set(t["clave"]) == {"dummy", "logistica"}
    assert t.loc[t["clave"] == "dummy", "auc_cv"].iloc[0] == 0.5


def test_las_graficas_llevan_titulo_ejes_y_leyenda(art):
    serie = servicio.serie_ticker(art, "AMD", 60)
    marcas = servicio.historial_adelante(art, "AMD", "logistica")
    figuras = [
        graficas.grafica_precio(serie, "AMD", True, True, marcas),
        graficas.barras_modelos(servicio.tabla_comparacion(art)),
    ]
    for fig in figuras:
        ax = fig.axes[0]
        assert ax.get_title("left")
        assert ax.get_xlabel()
        assert ax.get_ylabel()
        assert ax.get_legend() is not None


def test_las_graficas_son_transparentes_para_el_tema_oscuro(art):
    fig = graficas.barras_modelos(servicio.tabla_comparacion(art))
    assert fig.get_facecolor()[3] == 0
