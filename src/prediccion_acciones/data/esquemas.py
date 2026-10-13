"""Contrato de la tabla de precios: si algo no cumple, se detiene aqui.

La tabla en formato largo (una fila por fecha y ticker) es la entrada de todo lo
demas. Validarla con Pandera evita que un dato roto llegue a las features o al
modelo sin que nadie se entere.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

try:  # Pandera >= 0.24 separa la API de pandas; antes vivia en la raiz.
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

from pandera.errors import SchemaErrors

logger = logging.getLogger(__name__)

# Un cambio diario mayor a este umbral es sospechoso (split mal ajustado,
# dato corrupto). Solo avisa: hay dias reales con saltos fuertes.
UMBRAL_SALTO_DIARIO = 0.60


class ErrorValidacion(ValueError):
    """La tabla de precios no cumple el contrato."""


def _sin_saltos_raros(tabla: pd.DataFrame) -> bool:
    ordenada = tabla.sort_values(["ticker", "fecha"])
    # Cambio contra el dia anterior del mismo ticker, sin rellenar huecos.
    anterior = ordenada.groupby("ticker")["cierre"].shift(1)
    cambios = (ordenada["cierre"] / anterior - 1).abs().dropna()
    return bool((cambios <= UMBRAL_SALTO_DIARIO).all())


def crear_esquema_precios(
    tickers_permitidos: Sequence[str] | None = None,
) -> pa.DataFrameSchema:
    """Esquema de la tabla larga `fecha, ticker, cierre, volumen`."""
    checks_ticker = (
        [pa.Check.isin(list(tickers_permitidos))] if tickers_permitidos else []
    )
    return pa.DataFrameSchema(
        {
            "fecha": pa.Column("datetime64[ns]", nullable=False, coerce=True),
            "ticker": pa.Column(str, checks=checks_ticker, nullable=False),
            "cierre": pa.Column(
                float, checks=pa.Check.gt(0), nullable=False, coerce=True
            ),
            "volumen": pa.Column(
                float, checks=pa.Check.ge(0), nullable=False, coerce=True
            ),
        },
        checks=pa.Check(
            _sin_saltos_raros,
            name="sin_saltos_raros",
            error=f"Cambio diario mayor a {UMBRAL_SALTO_DIARIO:.0%}",
            raise_warning=True,
        ),
        unique=["fecha", "ticker"],
        strict=True,
    )


def validar_precios(
    tabla: pd.DataFrame, tickers_permitidos: Sequence[str] | None = None
) -> pd.DataFrame:
    """Devuelve la tabla validada (con tipos corregidos) o lanza `ErrorValidacion`."""
    esquema = crear_esquema_precios(tickers_permitidos)
    try:
        validada = esquema.validate(tabla, lazy=True)
    except SchemaErrors as exc:
        casos = exc.failure_cases
        logger.error("Validacion de precios fallo: %d casos", len(casos))
        raise ErrorValidacion(
            f"La tabla de precios no cumple el contrato "
            f"({len(casos)} casos). Primeros:\n{casos.head(10).to_string()}"
        ) from exc
    logger.info("Tabla de precios valida: %d filas", len(validada))
    return validada
