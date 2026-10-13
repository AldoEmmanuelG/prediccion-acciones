"""Flujo de entrenamiento de punta a punta.

precios -> validacion Pandera -> features y etiqueta -> particion por fechas ->
busqueda y validacion cruzada temporal -> holdout (una sola vez) -> reentrenar
hasta la fecha de corte -> prueba hacia adelante -> guardar modelos y metricas.

Se corre con `python -m prediccion_acciones.entrenamiento`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prediccion_acciones.config import RAIZ_PROYECTO, Config
from prediccion_acciones.data.almacenamiento import cargar_parquet, guardar_parquet
from prediccion_acciones.data.descarga import (
    DescargadorPrecios,
    Fuente,
    fuente_yfinance,
)
from prediccion_acciones.data.esquemas import validar_precios
from prediccion_acciones.data.utilidades import cronometro
from prediccion_acciones.evaluation.busqueda import buscar_hiperparametros
from prediccion_acciones.evaluation.cv import ResultadoCV, validar_cruzada
from prediccion_acciones.evaluation.metricas import (
    Metricas,
    calcular_metricas,
    umbral_youden,
)
from prediccion_acciones.evaluation.validacion import splits_por_fecha
from prediccion_acciones.features.ingenieria import construir_panel, nombres_features
from prediccion_acciones.features.particion import Particiones, dividir_panel
from prediccion_acciones.logging_config import configurar_logging
from prediccion_acciones.models.base import BaseModel
from prediccion_acciones.models.fabrica import crear_modelo
from prediccion_acciones.models.sklearn_modelo import ModeloSklearn
from prediccion_acciones.seguimiento import (
    SeguimientoMLflow,
    SeguimientoNulo,
)

logger = logging.getLogger(__name__)

MODELOS_POR_DEFECTO = ("dummy", "logistica", "boosting", "pytorch")
EXTENSIONES = {"pytorch": ".pt"}  # los demas se guardan como .joblib
NOMBRE_PRECIOS = "precios.parquet"


@dataclass(frozen=True)
class ResultadoModelo:
    """Todo lo que se midio de un modelo, listo para reportar."""

    nombre: str
    hiperparametros: dict[str, Any]
    cv_auc: float
    cv_auc_std: float
    cv_f1: float
    umbral: float
    holdout: Metricas
    adelante: Metricas | None
    ruta_modelo: Path | None


@dataclass(frozen=True)
class ResultadoEntrenamiento:
    modelos: tuple[ResultadoModelo, ...]
    directorio: Path
    filas: dict[str, int]


def obtener_precios(cfg: Config, fuente: Fuente, usar_cache: bool) -> pd.DataFrame:
    """Tabla larga de precios: del cache en Parquet o descargada de la fuente."""
    ruta = RAIZ_PROYECTO / cfg.rutas.datos_crudos / NOMBRE_PRECIOS
    if usar_cache and ruta.exists():
        return cargar_parquet(ruta)
    descargador = DescargadorPrecios(
        cfg.datos.tickers_objetivo, cfg.datos.fecha_inicio, fuente=fuente
    )
    largo = descargador.descargar().a_largo()
    # Estrategia de nulos: un dia sin cierre o sin volumen se descarta (no se
    # rellena), porque inventar un precio crearia retornos falsos.
    antes = len(largo)
    largo = largo.dropna(subset=["cierre", "volumen"]).reset_index(drop=True)
    if len(largo) < antes:
        logger.warning(
            "Se descartaron %d filas con cierre o volumen nulo", antes - len(largo)
        )
    if usar_cache:
        guardar_parquet(largo, ruta)
    return largo


def _a_json(valor: Any) -> Any:
    if is_dataclass(valor) and not isinstance(valor, type):
        return asdict(valor)
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, (Path, tuple, set)):
        return list(valor) if not isinstance(valor, Path) else str(valor)
    return str(valor)


def _hiperparametros(
    nombre: str,
    cfg: Config,
    columnas: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
    n_jobs: int,
) -> dict[str, Any]:
    """Argumentos para `crear_modelo`: los mejores de la rejilla, o los del YAML."""
    semilla = cfg.validacion.semilla
    if nombre == "pytorch":
        return {"params": cfg.modelos.pytorch, "gap_dias": cfg.validacion.gap_dias}
    rejilla = getattr(cfg.modelos, nombre, None)
    if not rejilla:
        return {}
    base = crear_modelo(nombre, columnas, semilla)
    assert isinstance(base, ModeloSklearn)
    with cronometro(f"busqueda de hiperparametros de {nombre}"):
        resultado = buscar_hiperparametros(
            base, rejilla, X, y, cv, semilla=semilla, n_jobs=n_jobs
        )
    return dict(resultado.mejores_params)


def _guardar_modelo(modelo: BaseModel, nombre: str, directorio: Path) -> Path:
    ruta = directorio / f"{nombre}{EXTENSIONES.get(nombre, '.joblib')}"
    modelo.guardar(ruta)
    return ruta


def entrenar_candidato(
    nombre: str,
    cfg: Config,
    columnas: list[str],
    particiones: Particiones,
    hasta_corte: pd.DataFrame,
    cv: list[tuple[np.ndarray, np.ndarray]],
    directorio: Path,
    seguimiento: SeguimientoNulo,
    n_jobs: int,
) -> tuple[ResultadoModelo, pd.Series | None]:
    """Evalua un modelo con CV, lo mide en el holdout y lo reentrena hasta el corte."""
    semilla = cfg.validacion.semilla
    tr, ho, ad = particiones.entrenamiento, particiones.holdout, particiones.adelante
    X_tr, y_tr, f_tr = tr[columnas], tr["y"].to_numpy(), tr["fecha"].to_numpy()

    with (
        seguimiento.corrida(nombre, {"modelo": nombre}),
        cronometro(f"modelo {nombre}"),
    ):
        hiper = _hiperparametros(nombre, cfg, columnas, X_tr, y_tr, cv, n_jobs)

        def crear() -> BaseModel:
            return crear_modelo(nombre, columnas, semilla, **hiper)

        # 1) Validacion cruzada temporal sobre el entrenamiento: aqui se decide todo.
        res_cv: ResultadoCV = validar_cruzada(crear, X_tr, y_tr, f_tr, cv)
        umbral = umbral_youden(res_cv.y_oof, res_cv.prob_oof)

        # 2) Holdout: se mide una sola vez, con el modelo entrenado solo con 'tr'.
        modelo_h = crear().fit(X_tr, y_tr, fechas=f_tr)
        m_holdout = calcular_metricas(
            ho["y"].to_numpy(), modelo_h.predict_proba(ho[columnas]), umbral
        )

        # 3) Modelo final: todo hasta la fecha de corte. Es el que usa la app.
        final = crear().fit(
            hasta_corte[columnas],
            hasta_corte["y"].to_numpy(),
            fechas=hasta_corte["fecha"].to_numpy(),
        )
        final.umbral = umbral

        # 4) Prueba hacia adelante: fechas posteriores al corte, nunca vistas.
        m_adelante, prob_adelante = None, None
        if len(ad):
            prob_adelante = pd.Series(
                final.predict_proba(ad[columnas]), index=ad.index, name=nombre
            )
            m_adelante = calcular_metricas(
                ad["y"].to_numpy(), prob_adelante.to_numpy(), umbral
            )

        ruta = None if nombre == "dummy" else _guardar_modelo(final, nombre, directorio)

        seguimiento.parametros(
            {
                **{k: v for k, v in hiper.items()},
                "umbral": round(umbral, 4),
                "fecha_corte": cfg.validacion.fecha_corte,
                "n_splits": cfg.validacion.n_splits,
                "gap_dias": cfg.validacion.gap_dias,
            }
        )
        seguimiento.metricas(
            {"auc": res_cv.auc_media, "auc_std": res_cv.auc_std, "f1": res_cv.f1_media},
            "cv_",
        )
        seguimiento.metricas(
            {k: v for k, v in m_holdout.a_dict().items() if k != "umbral"}, "holdout_"
        )
        if m_adelante is not None:
            seguimiento.metricas(
                {k: v for k, v in m_adelante.a_dict().items() if k != "umbral"},
                "adelante_",
            )
        if ruta is not None:
            seguimiento.artefacto(ruta)

    resultado = ResultadoModelo(
        nombre=nombre,
        hiperparametros=hiper,
        cv_auc=res_cv.auc_media,
        cv_auc_std=res_cv.auc_std,
        cv_f1=res_cv.f1_media,
        umbral=umbral,
        holdout=m_holdout,
        adelante=m_adelante,
        ruta_modelo=ruta,
    )
    logger.info(
        "%-10s CV AUC %.4f (+-%.4f) | holdout AUC %.4f | umbral %.2f",
        nombre,
        res_cv.auc_media,
        res_cv.auc_std,
        m_holdout.auc,
        umbral,
    )
    return resultado, prob_adelante


def _escribir_reporte(
    cfg: Config,
    resultados: list[ResultadoModelo],
    particiones: Particiones,
    probs: dict[str, pd.Series],
    precios: pd.DataFrame,
    directorio: Path,
) -> None:
    """Archivos que lee la app: metricas, prueba hacia adelante y precios."""

    def periodo(t: pd.DataFrame) -> list[str]:
        return [str(t["fecha"].min().date()), str(t["fecha"].max().date())]

    reporte = {
        "fecha_corte": str(cfg.validacion.fecha_corte),
        "periodos": {
            "entrenamiento": periodo(particiones.entrenamiento),
            "holdout": periodo(particiones.holdout),
            **(
                {"adelante": periodo(particiones.adelante)}
                if len(particiones.adelante)
                else {}
            ),
        },
        "modelos": {
            r.nombre: {
                "hiperparametros": r.hiperparametros,
                "cv": {"auc": r.cv_auc, "auc_std": r.cv_auc_std, "f1": r.cv_f1},
                "umbral": r.umbral,
                "holdout": r.holdout.a_dict(),
                "adelante": r.adelante.a_dict() if r.adelante else None,
            }
            for r in resultados
        },
    }
    with (directorio / "metricas.json").open("w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, default=_a_json, ensure_ascii=False)

    if probs:
        ad = particiones.adelante[["fecha", "ticker", "y"]].join(pd.DataFrame(probs))
        ad.to_csv(directorio / "prueba_adelante.csv", index=False)
    precios.to_csv(directorio / "precios_app.csv", index=False)


def _crear_seguimiento(
    cfg: Config, tracking_uri: str | None, artefactos: Path | None
) -> SeguimientoMLflow:
    """MLflow sobre SQLite (el almacenamiento en carpeta ya no es el predeterminado)."""
    uri = (
        tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(RAIZ_PROYECTO / 'mlflow.db').as_posix()}"
    )
    return SeguimientoMLflow(
        cfg.rutas.mlflow_experimento,
        uri,
        artefactos or RAIZ_PROYECTO / "mlartifacts",
    )


def ejecutar_pipeline(
    cfg: Config,
    directorio_modelos: Path | None = None,
    modelos: Sequence[str] = MODELOS_POR_DEFECTO,
    fuente: Fuente = fuente_yfinance,
    usar_cache: bool = True,
    usar_mlflow: bool = True,
    tracking_uri: str | None = None,
    directorio_artefactos_mlflow: Path | None = None,
    n_jobs: int = -1,
) -> ResultadoEntrenamiento:
    """Corre todo el flujo y devuelve los resultados de cada modelo."""
    directorio = directorio_modelos or RAIZ_PROYECTO / cfg.rutas.modelos
    directorio.mkdir(parents=True, exist_ok=True)

    with cronometro("flujo de entrenamiento"):
        precios = validar_precios(
            obtener_precios(cfg, fuente, usar_cache), cfg.datos.tickers_objetivo
        )
        columnas = nombres_features(cfg.features)
        panel = construir_panel(precios, cfg.features, cfg.datos.horizonte_dias)
        v = cfg.validacion
        particiones = dividir_panel(panel, v.fecha_corte, v.meses_holdout, v.gap_dias)
        hasta_corte = panel[panel["fecha"] <= pd.Timestamp(v.fecha_corte)]
        cv = splits_por_fecha(
            particiones.entrenamiento["fecha"].to_numpy(), v.n_splits, v.gap_dias
        )
        seguimiento = (
            _crear_seguimiento(cfg, tracking_uri, directorio_artefactos_mlflow)
            if usar_mlflow
            else SeguimientoNulo()
        )

        resultados, probs = [], {}
        for nombre in modelos:
            r, prob = entrenar_candidato(
                nombre,
                cfg,
                columnas,
                particiones,
                hasta_corte,
                cv,
                directorio,
                seguimiento,
                n_jobs,
            )
            resultados.append(r)
            if prob is not None:
                probs[nombre] = prob

        _escribir_reporte(cfg, resultados, particiones, probs, precios, directorio)

    filas = {
        "entrenamiento": len(particiones.entrenamiento),
        "holdout": len(particiones.holdout),
        "adelante": len(particiones.adelante),
    }
    return ResultadoEntrenamiento(tuple(resultados), directorio, filas)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena y evalua los modelos.")
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS_POR_DEFECTO))
    parser.add_argument("--sin-mlflow", action="store_true")
    parser.add_argument("--sin-cache", action="store_true", help="vuelve a descargar")
    args = parser.parse_args(argv)

    configurar_logging()
    cfg = Config.desde_yaml()
    tracking = None  # sale de MLFLOW_TRACKING_URI si esta definida
    resultado = ejecutar_pipeline(
        cfg,
        modelos=args.modelos,
        usar_cache=not args.sin_cache,
        usar_mlflow=not args.sin_mlflow,
        tracking_uri=tracking,
    )
    logger.info(
        "Listo. Artefactos en %s | filas %s", resultado.directorio, resultado.filas
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
