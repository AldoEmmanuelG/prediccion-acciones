import numpy as np
import pandas as pd
import pytest

from prediccion_acciones.models.base import BaseModel, ErrorModelo, registrar
from prediccion_acciones.models.fabrica import crear_modelo, modelos_disponibles
from prediccion_acciones.models.sklearn_modelo import ModeloSklearn

COLUMNAS = ["a", "b"]


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 2)), columns=COLUMNAS)
    y = (X["a"] + rng.normal(0, 0.5, 400) > 0).astype(int)
    return X, y


def test_la_fabrica_conoce_los_modelos_esperados():
    assert {"dummy", "logistica", "boosting"} <= set(modelos_disponibles())


def test_nombre_desconocido_da_error_con_las_opciones():
    with pytest.raises(ErrorModelo, match="logistica"):
        crear_modelo("inexistente", COLUMNAS)


@pytest.mark.parametrize("nombre", ["dummy", "logistica", "boosting"])
def test_todos_cumplen_el_contrato(nombre, datos):
    X, y = datos
    modelo = crear_modelo(nombre, COLUMNAS)
    assert isinstance(modelo, BaseModel)
    prob = modelo.fit(X, y).predict_proba(X)
    assert prob.shape == (len(X),)
    assert ((prob >= 0) & (prob <= 1)).all()
    assert set(np.unique(modelo.predict(X))) <= {0, 1}


def test_la_logistica_aprende_la_senal(datos):
    X, y = datos
    prob = crear_modelo("logistica", COLUMNAS).fit(X, y).predict_proba(X)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, prob) > 0.85


def test_predecir_sin_entrenar_falla(datos):
    X, _ = datos
    with pytest.raises(ErrorModelo, match="no esta entrenado"):
        crear_modelo("logistica", COLUMNAS).predict_proba(X)


def test_columnas_faltantes_fallan(datos):
    X, y = datos
    modelo = crear_modelo("logistica", COLUMNAS).fit(X, y)
    with pytest.raises(ErrorModelo, match="faltan"):
        modelo.predict_proba(X[["a"]])


def test_el_orden_de_las_columnas_no_importa(datos):
    X, y = datos
    modelo = crear_modelo("logistica", COLUMNAS).fit(X, y)
    np.testing.assert_allclose(
        modelo.predict_proba(X), modelo.predict_proba(X[["b", "a"]])
    )


def test_filas_distintas_en_X_e_y_fallan(datos):
    X, y = datos
    with pytest.raises(ValueError, match="mismo numero"):
        crear_modelo("logistica", COLUMNAS).fit(X, y[:-1])


def test_umbral_cambia_las_clases_predichas(datos):
    X, y = datos
    modelo = crear_modelo("logistica", COLUMNAS).fit(X, y)
    assert modelo.predict(X, umbral=0.9).sum() < modelo.predict(X, umbral=0.1).sum()


def test_guardar_y_cargar_da_las_mismas_probabilidades(datos, tmp_path):
    X, y = datos
    modelo = crear_modelo("logistica", COLUMNAS, C=0.5).fit(X, y)
    modelo.umbral = 0.55
    ruta = tmp_path / "carpeta" / "m.joblib"
    modelo.guardar(ruta)
    cargado = ModeloSklearn.cargar(ruta)
    np.testing.assert_allclose(modelo.predict_proba(X), cargado.predict_proba(X))
    assert cargado.umbral == 0.55
    assert cargado.columnas == tuple(COLUMNAS)


def test_no_se_guarda_un_modelo_sin_entrenar(tmp_path):
    with pytest.raises(ErrorModelo):
        crear_modelo("logistica", COLUMNAS).guardar(tmp_path / "m.joblib")


def test_cargar_un_archivo_inexistente_falla(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModeloSklearn.cargar(tmp_path / "no_existe.joblib")


def test_registrar_dos_veces_el_mismo_nombre_falla():
    with pytest.raises(ValueError, match="Ya existe"):
        registrar("logistica")(lambda columnas, semilla=42: None)


def test_sin_columnas_o_con_umbral_invalido_falla():
    with pytest.raises(ValueError):
        crear_modelo("logistica", [])
    modelo = crear_modelo("logistica", COLUMNAS)
    with pytest.raises(ValueError):
        ModeloSklearn(COLUMNAS, modelo.estimador, "x", umbral=1.5)
