from datetime import date

import numpy as np
import pandas as pd
import pytest

from prediccion_acciones.evaluation.validacion import splits_por_fecha
from prediccion_acciones.features.particion import dividir_panel


@pytest.fixture
def panel() -> pd.DataFrame:
    fechas = pd.bdate_range("2025-01-02", "2026-12-31")
    return pd.DataFrame(
        {
            "fecha": np.repeat(fechas.to_numpy(), 2),
            "ticker": np.tile(["AAA", "BBB"], len(fechas)),
            "y": 1,
        }
    )


def test_bloques_no_se_traslapan_y_respetan_el_gap(panel):
    p = dividir_panel(panel, date(2026, 9, 30), meses_holdout=6, gap_dias=5)
    fin_train = p.entrenamiento["fecha"].max()
    inicio_hold = p.holdout["fecha"].min()
    fechas = np.sort(panel["fecha"].unique())
    huecos = ((fechas > fin_train) & (fechas < inicio_hold)).sum()
    assert huecos == 5
    assert p.holdout["fecha"].max() <= pd.Timestamp("2026-09-30")
    assert p.adelante["fecha"].min() > pd.Timestamp("2026-09-30")


def test_holdout_empieza_seis_meses_antes_del_corte(panel):
    p = dividir_panel(panel, date(2026, 9, 30), meses_holdout=6, gap_dias=5)
    assert p.holdout["fecha"].min() == pd.Timestamp("2026-03-30")


def test_una_fecha_cae_completa_de_un_solo_lado(panel):
    p = dividir_panel(panel, date(2026, 9, 30), meses_holdout=6, gap_dias=5)
    assert not set(p.entrenamiento["fecha"]) & set(p.holdout["fecha"])
    assert (p.holdout.groupby("fecha").size() == 2).all()


def test_sin_historia_suficiente_falla(panel):
    with pytest.raises(ValueError, match="entrenar"):
        dividir_panel(panel, date(2025, 3, 31), meses_holdout=6, gap_dias=5)


def test_splits_son_temporales_y_respetan_el_gap(panel):
    fechas = panel["fecha"].to_numpy()
    pliegues = splits_por_fecha(fechas, n_splits=5, gap=5)
    assert len(pliegues) == 5
    unicas = np.sort(pd.unique(fechas))
    for tr, va in pliegues:
        assert fechas[tr].max() < fechas[va].min()
        entre = ((unicas > fechas[tr].max()) & (unicas < fechas[va].min())).sum()
        assert entre == 5
        assert len(set(fechas[tr]) & set(fechas[va])) == 0


def test_cada_pliegue_entrena_con_mas_historia(panel):
    pliegues = splits_por_fecha(panel["fecha"].to_numpy(), 5, 5)
    tamanos = [len(tr) for tr, _ in pliegues]
    assert tamanos == sorted(tamanos)
