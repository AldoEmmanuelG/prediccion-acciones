import numpy as np
import pandas as pd

from prediccion_acciones.evaluation.cv import validar_cruzada
from prediccion_acciones.evaluation.validacion import splits_por_fecha
from prediccion_acciones.models.fabrica import crear_modelo

COLUMNAS = ["a", "b"]


def _datos():
    rng = np.random.default_rng(3)
    fechas = np.repeat(pd.bdate_range("2023-01-02", periods=300).to_numpy(), 2)
    X = pd.DataFrame(rng.normal(size=(600, 2)), columns=COLUMNAS)
    y = (X["a"] + rng.normal(0, 0.5, 600) > 0).astype(int).to_numpy()
    return X, y, fechas


def test_devuelve_una_fila_por_pliegue_y_oof_sin_traslapes():
    X, y, fechas = _datos()
    cv = splits_por_fecha(fechas, n_splits=4, gap=2)
    res = validar_cruzada(lambda: crear_modelo("logistica", COLUMNAS), X, y, fechas, cv)
    assert len(res.por_pliegue) == 4
    assert len(res.y_oof) == len(res.prob_oof) == sum(len(va) for _, va in cv)
    assert res.auc_media > 0.7


def test_el_dummy_tiene_auc_de_medio():
    X, y, fechas = _datos()
    cv = splits_por_fecha(fechas, n_splits=3, gap=2)
    res = validar_cruzada(lambda: crear_modelo("dummy", COLUMNAS), X, y, fechas, cv)
    assert res.auc_media == 0.5
