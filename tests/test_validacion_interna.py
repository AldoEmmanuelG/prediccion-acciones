import numpy as np
import pandas as pd
import pytest

from prediccion_acciones.evaluation.validacion import dividir_por_fecha


@pytest.fixture
def fechas():
    return np.repeat(pd.bdate_range("2024-01-01", periods=100).to_numpy(), 3)


def test_la_validacion_es_posterior_al_entrenamiento_con_hueco(fechas):
    idx_tr, idx_va = dividir_por_fecha(fechas, fraccion_val=0.2, gap=5)
    unicas = np.sort(np.unique(fechas))
    entre = (unicas > fechas[idx_tr].max()) & (unicas < fechas[idx_va].min())
    assert fechas[idx_tr].max() < fechas[idx_va].min()
    assert entre.sum() == 5


def test_una_fecha_cae_completa_de_un_lado(fechas):
    idx_tr, idx_va = dividir_por_fecha(fechas, 0.2, 5)
    assert not set(fechas[idx_tr]) & set(fechas[idx_va])
    assert len(idx_va) == 20 * 3


def test_pocas_fechas_fallan():
    fechas = np.repeat(pd.bdate_range("2024-01-01", periods=6).to_numpy(), 2)
    with pytest.raises(ValueError, match="suficientes"):
        dividir_por_fecha(fechas, 0.5, 5)


def test_fraccion_invalida_falla(fechas):
    with pytest.raises(ValueError, match="fraccion_val"):
        dividir_por_fecha(fechas, 1.5, 5)
