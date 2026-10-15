"""Contrato de la API: lo que entra y lo que sale, validado con Pydantic."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Los modelos que ofrece el servicio (los mismos que `servicio.MODELOS_APP`).
NombreModelo = Literal["logistica", "pytorch"]


class PeticionPrediccion(BaseModel):
    """Lo que el usuario manda a `POST /predict`."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"ticker": "AMD", "modelo": "logistica"}]}
    )

    ticker: str = Field(
        min_length=1,
        max_length=10,
        description="Símbolo de la acción, por ejemplo AMD (mayúsculas o minúsculas).",
    )
    modelo: NombreModelo = Field(
        default="logistica",
        description="Modelo que hace la predicción: 'logistica' o 'pytorch'.",
    )

    @field_validator("ticker")
    @classmethod
    def _normalizar_ticker(cls, valor: str) -> str:
        return valor.strip().upper()


class RespuestaPrediccion(BaseModel):
    """Predicción del siguiente cierre de una acción."""

    ticker: str
    modelo: NombreModelo
    fecha_datos: date = Field(description="Último día de mercado usado para predecir.")
    cierre: float = Field(description="Cierre de ese último día.")
    probabilidad_sube: float = Field(
        ge=0, le=1, description="Probabilidad de que el siguiente cierre sea mayor."
    )
    umbral: float = Field(
        ge=0,
        le=1,
        description="Umbral elegido en validación: desde aquí el modelo dice 'sube'.",
    )
    prediccion: Literal["sube", "baja"]


class RespuestaSalud(BaseModel):
    """Estado del servicio y del modelo que cargó."""

    estado: Literal["ok"]
    modelos_cargados: list[str]
    modelo_por_defecto: str
    entrenado_hasta: date = Field(description="Fecha de corte del entrenamiento.")
    datos_hasta: date = Field(description="Último día de mercado disponible.")
    tickers: list[str]
