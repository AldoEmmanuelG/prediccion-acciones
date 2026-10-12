import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from prediccion_acciones.data.almacenamiento import (  # noqa: E402
    cargar_parquet,
    guardar_parquet,
)


def test_ida_y_vuelta_conserva_la_tabla(tmp_path):
    tabla = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "AAA"],
            "cierre": [10.5, 11.0],
        }
    )
    ruta = guardar_parquet(tabla, tmp_path / "sub" / "precios.parquet")
    pd.testing.assert_frame_equal(cargar_parquet(ruta), tabla)


def test_cargar_un_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="descarga"):
        cargar_parquet(tmp_path / "no_esta.parquet")
