"""Carga de la configuracion: el YAML se convierte en dataclasses inmutables.

El resto del codigo no lee el YAML directamente. Recibe un objeto `Config` con
campos con nombre y tipo, asi un error de escritura en la configuracion se
detecta aqui, al arrancar, y no a mitad de un entrenamiento.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ_PROYECTO / "configs" / "config.yaml"


class ErrorConfiguracion(ValueError):
    """La configuracion esta incompleta o tiene un valor invalido."""


@dataclass(frozen=True)
class RutasConfig:
    datos_crudos: Path
    datos_procesados: Path
    modelos: Path
    mlflow_experimento: str


@dataclass(frozen=True)
class DatosConfig:
    tickers_objetivo: tuple[str, ...]
    tickers_contexto: tuple[str, ...]
    fecha_inicio: date
    horizonte_dias: int

    @property
    def todos_los_tickers(self) -> tuple[str, ...]:
        return self.tickers_objetivo + self.tickers_contexto


@dataclass(frozen=True)
class FeaturesConfig:
    ventanas_retorno: tuple[int, ...]
    ventanas_rolling: tuple[int, ...]
    rsi_periodo: int
    ventanas_contexto: tuple[int, ...]


@dataclass(frozen=True)
class ValidacionConfig:
    n_splits: int
    gap_dias: int
    fecha_corte: date
    meses_holdout: int
    semilla: int


@dataclass(frozen=True)
class PytorchConfig:
    capas_ocultas: tuple[int, ...]
    dropout: float
    learning_rate: float
    batch_size: int
    epocas_max: int
    paciencia: int
    weight_decay: float
    fraccion_validacion: float


@dataclass(frozen=True)
class ModelosConfig:
    logistica: dict[str, list[Any]]
    boosting: dict[str, list[Any]]
    pytorch: PytorchConfig


@dataclass(frozen=True)
class Config:
    rutas: RutasConfig
    datos: DatosConfig
    features: FeaturesConfig
    validacion: ValidacionConfig
    modelos: ModelosConfig

    @classmethod
    def desde_yaml(cls, ruta: Path | str = RUTA_CONFIG) -> Config:
        """Lee el YAML, lo valida y devuelve la configuracion como objeto."""
        ruta = Path(ruta)
        logger.info("Cargando configuracion desde %s", ruta)
        try:
            with ruta.open(encoding="utf-8") as f:
                crudo = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise ErrorConfiguracion(f"No existe el archivo {ruta}") from exc
        if not isinstance(crudo, dict):
            raise ErrorConfiguracion(f"{ruta} no contiene un diccionario YAML")
        config = cls._construir(crudo)
        config._validar()
        return config

    @classmethod
    def _construir(cls, crudo: dict[str, Any]) -> Config:
        try:
            r, d, f = crudo["rutas"], crudo["datos"], crudo["features"]
            v, m = crudo["validacion"], crudo["modelos"]
            return cls(
                rutas=RutasConfig(
                    datos_crudos=Path(r["datos_crudos"]),
                    datos_procesados=Path(r["datos_procesados"]),
                    modelos=Path(r["modelos"]),
                    mlflow_experimento=str(r["mlflow_experimento"]),
                ),
                datos=DatosConfig(
                    tickers_objetivo=tuple(d["tickers_objetivo"]),
                    tickers_contexto=tuple(d["tickers_contexto"]),
                    fecha_inicio=_a_fecha(d["fecha_inicio"], "datos.fecha_inicio"),
                    horizonte_dias=int(d["horizonte_dias"]),
                ),
                features=FeaturesConfig(
                    ventanas_retorno=tuple(f["ventanas_retorno"]),
                    ventanas_rolling=tuple(f["ventanas_rolling"]),
                    rsi_periodo=int(f["rsi_periodo"]),
                    ventanas_contexto=tuple(f["ventanas_contexto"]),
                ),
                validacion=ValidacionConfig(
                    n_splits=int(v["n_splits"]),
                    gap_dias=int(v["gap_dias"]),
                    fecha_corte=_a_fecha(v["fecha_corte"], "validacion.fecha_corte"),
                    meses_holdout=int(v["meses_holdout"]),
                    semilla=int(v["semilla"]),
                ),
                modelos=ModelosConfig(
                    logistica=dict(m["logistica"]),
                    boosting=dict(m["boosting"]),
                    pytorch=_construir_pytorch(m["pytorch"]),
                ),
            )
        except KeyError as exc:
            raise ErrorConfiguracion(
                f"Falta la clave {exc} en la configuracion"
            ) from exc

    def _validar(self) -> None:
        d, v, p = self.datos, self.validacion, self.modelos.pytorch
        repetidos = {t for t in d.todos_los_tickers if d.todos_los_tickers.count(t) > 1}
        if repetidos:
            raise ErrorConfiguracion(f"Tickers repetidos: {sorted(repetidos)}")
        if not d.tickers_objetivo:
            raise ErrorConfiguracion("tickers_objetivo no puede estar vacio")
        if d.horizonte_dias < 1:
            raise ErrorConfiguracion("horizonte_dias debe ser >= 1")
        if v.fecha_corte <= d.fecha_inicio:
            raise ErrorConfiguracion("fecha_corte debe ser posterior a fecha_inicio")
        if v.n_splits < 2:
            raise ErrorConfiguracion("n_splits debe ser >= 2")
        if not 0 < p.fraccion_validacion < 1:
            raise ErrorConfiguracion("fraccion_validacion debe estar entre 0 y 1")
        if not 0 <= p.dropout < 1:
            raise ErrorConfiguracion("dropout debe estar entre 0 y 1")


def _construir_pytorch(crudo: dict[str, Any]) -> PytorchConfig:
    return PytorchConfig(
        capas_ocultas=tuple(crudo["capas_ocultas"]),
        dropout=float(crudo["dropout"]),
        learning_rate=float(crudo["learning_rate"]),
        batch_size=int(crudo["batch_size"]),
        epocas_max=int(crudo["epocas_max"]),
        paciencia=int(crudo["paciencia"]),
        weight_decay=float(crudo["weight_decay"]),
        fraccion_validacion=float(crudo["fraccion_validacion"]),
    )


def _a_fecha(valor: Any, nombre: str) -> date:
    """Acepta texto ISO ('2026-09-30') o una fecha que YAML ya haya convertido."""
    if isinstance(valor, date):
        return valor
    try:
        return date.fromisoformat(str(valor))
    except ValueError as exc:
        raise ErrorConfiguracion(f"{nombre}: '{valor}' no es una fecha ISO") from exc
