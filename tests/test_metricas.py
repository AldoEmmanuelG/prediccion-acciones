import numpy as np
import pytest

from prediccion_acciones.evaluation.metricas import (
    calcular_metricas,
    umbral_youden,
)


def test_prediccion_perfecta():
    y = np.array([0, 0, 1, 1])
    m = calcular_metricas(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m.auc == 1.0
    assert m.f1 == 1.0
    assert m.exactitud == 1.0
    assert m.n == 4


def test_el_umbral_cambia_f1_y_exactitud():
    y = np.array([0, 1, 1, 1])
    prob = np.array([0.2, 0.4, 0.6, 0.8])
    assert calcular_metricas(y, prob, 0.5).exactitud == 0.75
    assert calcular_metricas(y, prob, 0.1).exactitud == 0.75
    assert calcular_metricas(y, prob, 0.7).exactitud == 0.5


def test_una_sola_clase_da_auc_nan_y_json_none():
    m = calcular_metricas(np.ones(5), np.linspace(0.1, 0.9, 5))
    assert np.isnan(m.auc)
    assert m.a_dict()["auc"] is None


def test_largos_distintos_fallan():
    with pytest.raises(ValueError):
        calcular_metricas(np.array([0, 1]), np.array([0.5]))


def test_youden_separa_dos_grupos():
    y = np.array([0] * 50 + [1] * 50)
    prob = np.concatenate([np.linspace(0.05, 0.4, 50), np.linspace(0.6, 0.95, 50)])
    assert 0.4 <= umbral_youden(y, prob) <= 0.6


def test_youden_sin_dos_clases_devuelve_medio():
    assert umbral_youden(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5
