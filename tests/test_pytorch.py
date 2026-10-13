import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from prediccion_acciones.config import PytorchConfig  # noqa: E402
from prediccion_acciones.models.base import BaseModel, ErrorModelo  # noqa: E402
from prediccion_acciones.models.fabrica import crear_modelo  # noqa: E402
from prediccion_acciones.models.pytorch_modelo import (  # noqa: E402
    MLP,
    ModeloPytorch,
)

COLUMNAS = ["a", "b"]
PARAMS = PytorchConfig(
    capas_ocultas=(8,),
    dropout=0.0,
    learning_rate=0.01,
    batch_size=64,
    epocas_max=40,
    paciencia=8,
    weight_decay=0.0,
    fraccion_validacion=0.2,
)


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    fechas = np.repeat(pd.bdate_range("2023-01-02", periods=400).to_numpy(), 2)
    X = pd.DataFrame(rng.normal(size=(800, 2)), columns=COLUMNAS)
    y = (X["a"] + rng.normal(0, 0.3, 800) > 0).astype(int).to_numpy()
    return X, y, fechas


def _modelo(semilla: int = 42) -> ModeloPytorch:
    return ModeloPytorch(COLUMNAS, PARAMS, semilla=semilla, gap_dias=2)


def test_la_red_devuelve_un_logit_por_fila_y_arranca_en_ln2():
    torch.manual_seed(0)
    red = MLP(13, (32, 16), 0.0)
    x = torch.randn(64, 13)
    y = torch.randint(0, 2, (64,)).float()
    assert tuple(red(x).shape) == (64,)
    perdida = torch.nn.BCEWithLogitsLoss()(red(x), y).item()
    assert perdida == pytest.approx(0.693, abs=0.15)


def test_la_red_puede_memorizar_un_lote_chico():
    torch.manual_seed(0)
    red = MLP(13, (64, 32), 0.0)
    x = torch.randn(64, 13)
    y = torch.randint(0, 2, (64,)).float()
    fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(red.parameters(), lr=0.01)
    for _ in range(800):
        perdida = fn(red(x), y)
        perdida.backward()
        opt.step()
        opt.zero_grad()
    assert perdida.item() < 0.35


def test_cumple_el_contrato_y_aprende_fuera_de_muestra(datos):
    from sklearn.metrics import roc_auc_score

    X, y, fechas = datos
    modelo = _modelo()
    assert isinstance(modelo, BaseModel)
    modelo.fit(X.iloc[:600], y[:600], fechas=fechas[:600])
    prob = modelo.predict_proba(X.iloc[600:])
    assert prob.shape == (200,)
    assert ((prob >= 0) & (prob <= 1)).all()
    assert roc_auc_score(y[600:], prob) > 0.8


def test_sin_fechas_falla(datos):
    X, y, _ = datos
    with pytest.raises(ErrorModelo, match="fechas"):
        _modelo().fit(X, y)


def test_registra_la_historia_y_la_mejor_epoca(datos):
    X, y, fechas = datos
    modelo = _modelo().fit(X, y, fechas=fechas)
    assert {"epoca", "perdida_train", "perdida_val", "auc_val"} <= set(
        modelo.historia.columns
    )
    assert 1 <= modelo.epoca_mejor <= len(modelo.historia) <= PARAMS.epocas_max
    assert modelo.auc_validacion == pytest.approx(
        modelo.historia["auc_val"].max(), abs=1e-4
    )


def test_misma_semilla_da_las_mismas_probabilidades(datos):
    X, y, fechas = datos
    p1 = _modelo(7).fit(X, y, fechas=fechas).predict_proba(X)
    p2 = _modelo(7).fit(X, y, fechas=fechas).predict_proba(X)
    np.testing.assert_allclose(p1, p2)


def test_guardar_y_cargar_da_las_mismas_probabilidades(datos, tmp_path):
    X, y, fechas = datos
    modelo = _modelo().fit(X, y, fechas=fechas)
    modelo.umbral = 0.52
    ruta = tmp_path / "modelos" / "mlp.pt"
    modelo.guardar(ruta)
    cargado = ModeloPytorch.cargar(ruta)
    np.testing.assert_allclose(
        modelo.predict_proba(X), cargado.predict_proba(X), atol=1e-6
    )
    assert cargado.umbral == 0.52
    assert cargado.columnas == tuple(COLUMNAS)
    assert cargado.params == PARAMS


def test_no_se_guarda_un_modelo_sin_entrenar(tmp_path):
    with pytest.raises(ErrorModelo):
        _modelo().guardar(tmp_path / "mlp.pt")


def test_la_fabrica_crea_el_modelo_pytorch():
    modelo = crear_modelo("pytorch", COLUMNAS, params=PARAMS, gap_dias=2)
    assert isinstance(modelo, ModeloPytorch)


def test_la_fabrica_exige_los_parametros():
    with pytest.raises(ErrorModelo, match="params"):
        crear_modelo("pytorch", COLUMNAS)
