import numpy as np
import pandas as pd
import pytest

from prediccion_acciones.evaluation.busqueda import (
    METRICAS,
    buscar_hiperparametros,
)
from prediccion_acciones.evaluation.validacion import splits_por_fecha
from prediccion_acciones.models.fabrica import crear_modelo

COLUMNAS = ["a", "b"]


@pytest.fixture
def panel():
    rng = np.random.default_rng(1)
    fechas = pd.bdate_range("2024-01-01", periods=300)
    X = pd.DataFrame(rng.normal(size=(600, 2)), columns=COLUMNAS)
    X["fecha"] = np.repeat(fechas.to_numpy(), 2)
    y = (X["a"] + rng.normal(0, 0.7, 600) > 0).astype(int)
    return X, y


def test_rejilla_devuelve_dos_metricas_y_el_mejor_modelo(panel):
    X, y = panel
    cv = splits_por_fecha(X["fecha"].to_numpy(), n_splits=3, gap=2)
    res = buscar_hiperparametros(
        crear_modelo("logistica", COLUMNAS),
        {"C": [0.01, 1.0], "class_weight": [None, "balanced"]},
        X,
        y,
        cv,
        n_jobs=1,
    )
    assert res.combinaciones == 4
    assert set(METRICAS) == {"auc", "f1"}
    assert {"mean_test_auc", "mean_test_f1"} <= set(res.tabla.columns)
    assert res.auc_cv > 0.7
    assert set(res.mejores_params) == {"C", "class_weight"}
    assert res.mejor_modelo.predict_proba(X).shape == (len(X),)


def test_busqueda_aleatoria_respeta_n_iter(panel):
    X, y = panel
    cv = splits_por_fecha(X["fecha"].to_numpy(), n_splits=3, gap=2)
    res = buscar_hiperparametros(
        crear_modelo("logistica", COLUMNAS),
        {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        X,
        y,
        cv,
        n_iter=3,
        n_jobs=1,
    )
    assert res.combinaciones == 3


def test_rejilla_vacia_falla(panel):
    X, y = panel
    with pytest.raises(ValueError, match="vacia"):
        buscar_hiperparametros(
            crear_modelo("logistica", COLUMNAS), {}, X, y, [], n_jobs=1
        )
