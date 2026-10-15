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


class FilaResumen(BaseModel):
    """Una acción en el resumen del siguiente cierre."""

    ticker: str
    sector: str
    fecha_datos: date
    cierre: float
    cambio_1d: float = Field(description="Retorno del último día (0.01 = +1 %).")
    probabilidad_sube: float = Field(ge=0, le=1)
    umbral: float = Field(ge=0, le=1)
    prediccion: Literal["sube", "baja"]


class RespuestaResumen(BaseModel):
    """Las 10 acciones ordenadas de mayor a menor probabilidad de subir."""

    modelo: NombreModelo
    fecha_datos: date = Field(description="Último día de mercado usado.")
    fecha_prediccion: date = Field(
        description="Día hábil al que se refiere la predicción."
    )
    acciones: list[FilaResumen]


class PuntoPrecio(BaseModel):
    """Cierre de un día y sus medias móviles (vacías al inicio de la serie)."""

    fecha: date
    cierre: float
    sma20: float | None = None
    sma50: float | None = None


class DiaNoVisto(BaseModel):
    """Predicción del modelo sobre un día posterior a la fecha de corte."""

    fecha: date
    probabilidad_sube: float = Field(ge=0, le=1)
    prediccion: Literal["sube", "baja"]
    real: Literal["sube", "baja"] = Field(description="Lo que pasó ese día.")
    acierto: bool
    cierre: float


class RespuestaSerie(BaseModel):
    """Historial de precios de una acción y cómo le fue al modelo en días no vistos."""

    ticker: str
    modelo: NombreModelo
    dias: int
    fecha_corte: date = Field(description="Fecha de corte del entrenamiento.")
    precios: list[PuntoPrecio]
    dias_no_vistos: list[DiaNoVisto]


class FilaModelo(BaseModel):
    """Métricas de un modelo (el piso, `dummy`, también aparece aquí)."""

    clave: str
    nombre: str
    auc_validacion: float
    auc_validacion_std: float
    f1_validacion: float
    auc_holdout: float
    f1_holdout: float
    exactitud_holdout: float | None = None
    auc_adelante: float | None = None
    n_adelante: int | None = None


class RespuestaModelos(BaseModel):
    """Comparación de modelos contra el piso."""

    fecha_corte: date
    periodos: dict[str, list[str]]
    piso_exactitud: float | None = Field(
        default=None,
        description="Exactitud en el holdout de responder siempre la clase mayor.",
    )
    modelos: list[FilaModelo]
