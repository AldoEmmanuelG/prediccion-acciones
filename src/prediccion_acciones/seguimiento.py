"""Registro de experimentos en MLflow, con una version nula para no usarlo.

`SeguimientoMLflow` guarda en cada corrida los hiperparametros, las metricas y el
modelo serializado. `SeguimientoNulo` tiene la misma interfaz y no hace nada
(patron Null Object): el flujo de entrenamiento no necesita preguntar si hay
MLflow antes de cada llamada.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SeguimientoNulo:
    """Misma interfaz que `SeguimientoMLflow`, sin efectos."""

    @contextmanager
    def corrida(
        self, nombre: str, etiquetas: Mapping[str, str] | None = None
    ) -> Iterator[None]:
        yield

    def parametros(self, params: Mapping[str, Any]) -> None:
        return None

    def metricas(self, valores: Mapping[str, float | None], prefijo: str = "") -> None:
        return None

    def artefacto(self, ruta: Path | str) -> None:
        return None


class SeguimientoMLflow(SeguimientoNulo):
    """Guarda cada corrida en MLflow (`mlflow ui` las muestra y compara)."""

    def __init__(
        self,
        experimento: str,
        tracking_uri: str | None = None,
        directorio_artefactos: Path | str | None = None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        # Los modelos y archivos de cada corrida van a esta carpeta; se fija al
        # crear el experimento (despues ya no se puede cambiar).
        if directorio_artefactos and mlflow.get_experiment_by_name(experimento) is None:
            carpeta = Path(directorio_artefactos)
            carpeta.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(experimento, artifact_location=carpeta.as_uri())
        mlflow.set_experiment(experimento)
        logger.info(
            "MLflow listo: experimento '%s' en %s",
            experimento,
            mlflow.get_tracking_uri(),
        )

    @contextmanager
    def corrida(
        self, nombre: str, etiquetas: Mapping[str, str] | None = None
    ) -> Iterator[None]:
        with self._mlflow.start_run(run_name=nombre):
            if etiquetas:
                self._mlflow.set_tags(dict(etiquetas))
            yield

    def parametros(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def metricas(self, valores: Mapping[str, float | None], prefijo: str = "") -> None:
        limpias = {
            f"{prefijo}{k}": float(v)
            for k, v in valores.items()
            if v is not None and not math.isnan(float(v))
        }
        self._mlflow.log_metrics(limpias)

    def artefacto(self, ruta: Path | str) -> None:
        self._mlflow.log_artifact(str(ruta))
