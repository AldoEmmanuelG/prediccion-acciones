import json

import numpy as np
import pandas as pd
import pytest

from prediccion_acciones.reportes import (
    figura_curvas,
    figura_exploracion,
    figura_resultados,
    generar_reportes,
)


@pytest.fixture
def carpeta_modelos(tmp_path):
    rng = np.random.default_rng(0)
    fechas = pd.bdate_range("2025-01-01", periods=300)
    precios = pd.concat(
        pd.DataFrame(
            {
                "fecha": fechas,
                "ticker": t,
                "cierre": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(fechas))),
                "volumen": 1e5,
            }
        )
        for t in ("AMD", "KOF", "LMT", "IBM")
    )
    precios.to_csv(tmp_path / "precios_app.csv", index=False)

    epocas = np.arange(1, 13)
    pd.DataFrame(
        {
            "epoca": epocas,
            "perdida_train": 0.70 - 0.01 * epocas,
            "perdida_val": 0.69 + 0.002 * epocas,
            "auc_val": 0.50 + 0.003 * np.minimum(epocas, 5),
            "mejor": epocas == 5,
        }
    ).to_csv(tmp_path / "historia_pytorch.csv", index=False)

    y = rng.integers(0, 2, 400)
    holdout = pd.DataFrame({"fecha": 0, "ticker": "AMD", "y": y})
    for m in ("dummy", "logistica", "boosting", "pytorch"):
        holdout[m] = 0.5 if m == "dummy" else rng.uniform(0.3, 0.7, len(y))
    holdout.to_csv(tmp_path / "prediccion_holdout.csv", index=False)

    modelos = {
        m: {"holdout": {"exactitud": 0.5 + 0.01 * i}}
        for i, m in enumerate(("dummy", "logistica", "boosting", "pytorch"))
    }
    (tmp_path / "metricas.json").write_text(json.dumps({"modelos": modelos}))
    return tmp_path


def test_cada_figura_trae_titulo_ejes_y_leyenda(carpeta_modelos):
    precios = pd.read_csv(carpeta_modelos / "precios_app.csv", parse_dates=["fecha"])
    historia = pd.read_csv(carpeta_modelos / "historia_pytorch.csv")
    holdout = pd.read_csv(carpeta_modelos / "prediccion_holdout.csv")
    metricas = json.loads((carpeta_modelos / "metricas.json").read_text())
    for figura in (
        figura_exploracion(precios),
        figura_curvas(historia),
        figura_resultados(holdout, metricas),
    ):
        for ax in figura.axes:
            assert ax.get_title()
            assert ax.get_xlabel()
            assert ax.get_ylabel()
            assert ax.get_legend() is not None


def test_las_curvas_marcan_la_mejor_epoca(carpeta_modelos):
    historia = pd.read_csv(carpeta_modelos / "historia_pytorch.csv")
    etiquetas = [
        t.get_text() for t in figura_curvas(historia).axes[0].get_legend().texts
    ]
    assert any("(5)" in e for e in etiquetas)


def test_genera_los_tres_png(carpeta_modelos, tmp_path):
    rutas = generar_reportes(carpeta_modelos, tmp_path / "reports")
    assert len(rutas) == 3
    assert all(r.exists() and r.stat().st_size > 1000 for r in rutas)
