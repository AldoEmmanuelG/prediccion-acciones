"""Servicio FastAPI: predicciones, datos para la página web y la página en `/`.

El modelo se carga una sola vez, al arrancar el servicio (no en cada petición).
Se levanta con `uvicorn prediccion_acciones.api.main:app`. La documentación
interactiva queda en `/docs` y la página web, que consume estos mismos
endpoints, en `/`.

Endpoints principales: `GET /health` y `POST /predict`. Los de solo lectura
`GET /resumen`, `GET /series/{ticker}` y `GET /modelos` alimentan las tablas y
gráficas de la página y reutilizan los cálculos de `app/servicio.py`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from prediccion_acciones.api.esquemas import (
    DiaNoVisto,
    FilaModelo,
    FilaResumen,
    NombreModelo,
    PeticionPrediccion,
    PuntoPrecio,
    RespuestaModelos,
    RespuestaPrediccion,
    RespuestaResumen,
    RespuestaSalud,
    RespuestaSerie,
)
from prediccion_acciones.app import servicio

logger = logging.getLogger(__name__)

# Página web de una sola pieza (HTML, CSS y JS juntos, sin recursos externos).
RUTA_PAGINA = Path(__file__).parent / "static" / "index.html"

DESCRIPCION = (
    "Estima la probabilidad de que una acción cierre mañana por encima de hoy. "
    "Es un ejercicio académico: con datos públicos de precios el resultado es "
    "muy parecido al azar y no es asesoría de inversión."
)


def _numero(valor: object) -> float | None:
    """Convierte a float, dejando `None` donde pandas tiene un vacío (NaN)."""
    return None if pd.isna(valor) else float(valor)


def _validar_ticker(art: servicio.Artefactos, ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker not in art.tickers:
        detalle = f"Ticker '{ticker}' no disponible. Opciones: {art.tickers}"
        raise HTTPException(status_code=422, detail=detalle)
    return ticker


def _validar_modelo(art: servicio.Artefactos, modelo: str) -> None:
    if modelo not in art.modelos:
        detalle = f"Modelo '{modelo}' no disponible. Opciones: {sorted(art.modelos)}"
        raise HTTPException(status_code=422, detail=detalle)


def crear_app(directorio_modelos: Path | str | None = None) -> FastAPI:
    """Fábrica de la app; `directorio_modelos` deja probarla con otros archivos."""

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        # Aquí se carga el modelo: una vez por proceso, antes de atender peticiones.
        app.state.artefactos = servicio.cargar_artefactos(directorio_modelos)
        logger.info("Servicio listo con modelos %s", app.state.artefactos.modelos_app)
        yield

    app = FastAPI(
        title="Predicción de la dirección diaria de acciones",
        description=DESCRIPCION,
        version="1.0.0",
        lifespan=ciclo_de_vida,
    )

    pagina = RUTA_PAGINA.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def inicio() -> HTMLResponse:
        """Interfaz web: llama a los endpoints de la API desde el navegador."""
        return HTMLResponse(pagina)

    @app.get("/health", response_model=RespuestaSalud, tags=["servicio"])
    def salud(request: Request) -> RespuestaSalud:
        """Dice que el servicio está vivo, qué modelos cargó y de qué fecha son."""
        art: servicio.Artefactos = request.app.state.artefactos
        modelos = art.modelos_app
        return RespuestaSalud(
            estado="ok",
            modelos_cargados=modelos,
            modelo_por_defecto=modelos[0],
            entrenado_hasta=date.fromisoformat(str(art.metricas["fecha_corte"])),
            datos_hasta=art.ultima_fecha.date(),
            tickers=art.tickers,
        )

    @app.post("/predict", response_model=RespuestaPrediccion, tags=["predicción"])
    def predecir(peticion: PeticionPrediccion, request: Request) -> RespuestaPrediccion:
        """Probabilidad de que la acción suba en el siguiente cierre."""
        art: servicio.Artefactos = request.app.state.artefactos
        try:
            p = servicio.prediccion_ticker(art, peticion.ticker, peticion.modelo)
        except servicio.ErrorEntrada as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return RespuestaPrediccion(
            ticker=p.ticker,
            modelo=p.modelo,
            fecha_datos=p.fecha_datos.date(),
            cierre=p.cierre,
            probabilidad_sube=p.probabilidad_sube,
            umbral=p.umbral,
            prediccion="sube" if p.sube else "baja",
        )

    @app.get("/resumen", response_model=RespuestaResumen, tags=["datos"])
    def resumen(
        request: Request, modelo: NombreModelo = "logistica"
    ) -> RespuestaResumen:
        """Las 10 acciones con la probabilidad de subir, de mayor a menor."""
        art: servicio.Artefactos = request.app.state.artefactos
        _validar_modelo(art, modelo)
        tabla = servicio.resumen_proximo_cierre(art, modelo)
        umbral = float(art.modelos[modelo].umbral)
        ultima = tabla["fecha"].max()
        acciones = [
            FilaResumen(
                ticker=f.ticker,
                sector=f.sector,
                fecha_datos=f.fecha.date(),
                cierre=float(f.cierre),
                cambio_1d=float(f.cambio_1d),
                probabilidad_sube=float(f.prob_sube),
                umbral=umbral,
                prediccion="sube" if f.senal == "Sube" else "baja",
            )
            for f in tabla.itertuples()
        ]
        return RespuestaResumen(
            modelo=modelo,
            fecha_datos=ultima.date(),
            fecha_prediccion=(ultima + pd.offsets.BDay(1)).date(),
            acciones=acciones,
        )

    @app.get("/series/{ticker}", response_model=RespuestaSerie, tags=["datos"])
    def serie(
        ticker: str,
        request: Request,
        dias: Annotated[int, Query(ge=20, le=500)] = 60,
        modelo: NombreModelo = "logistica",
    ) -> RespuestaSerie:
        """Cierre y medias móviles de los últimos `dias`, y los días no vistos."""
        art: servicio.Artefactos = request.app.state.artefactos
        ticker = _validar_ticker(art, ticker)
        _validar_modelo(art, modelo)
        precios = servicio.serie_ticker(art, ticker, dias)
        historial = servicio.historial_adelante(art, ticker, modelo)
        return RespuestaSerie(
            ticker=ticker,
            modelo=modelo,
            dias=dias,
            fecha_corte=date.fromisoformat(str(art.metricas["fecha_corte"])),
            precios=[
                PuntoPrecio(
                    fecha=f.fecha.date(),
                    cierre=float(f.cierre),
                    sma20=_numero(f.sma20),
                    sma50=_numero(f.sma50),
                )
                for f in precios.itertuples()
            ],
            dias_no_vistos=[
                DiaNoVisto(
                    fecha=f.fecha.date(),
                    probabilidad_sube=float(f.prob_sube),
                    prediccion="sube" if f.prediccion == "Sube" else "baja",
                    real="sube" if f.real == "Subió" else "baja",
                    acierto=bool(f.acierto),
                    cierre=float(f.cierre),
                )
                for f in historial.itertuples()
            ],
        )

    @app.get("/modelos", response_model=RespuestaModelos, tags=["datos"])
    def modelos(request: Request) -> RespuestaModelos:
        """Métricas de cada modelo contra el piso (el modelo que siempre dice sube)."""
        art: servicio.Artefactos = request.app.state.artefactos
        metricas = art.metricas["modelos"]
        filas = []
        for f in servicio.tabla_comparacion(art).itertuples():
            n = _numero(f.n_adelante)
            filas.append(
                FilaModelo(
                    clave=f.clave,
                    nombre=f.modelo,
                    auc_validacion=float(f.auc_cv),
                    auc_validacion_std=float(f.auc_cv_std),
                    f1_validacion=float(f.f1_cv),
                    auc_holdout=float(f.auc_holdout),
                    f1_holdout=float(f.f1_holdout),
                    exactitud_holdout=metricas[f.clave]["holdout"].get("exactitud"),
                    auc_adelante=_numero(f.auc_adelante),
                    n_adelante=None if n is None else int(n),
                )
            )
        piso = metricas.get("dummy", {}).get("holdout", {}).get("exactitud")
        return RespuestaModelos(
            fecha_corte=date.fromisoformat(str(art.metricas["fecha_corte"])),
            periodos=art.metricas["periodos"],
            piso_exactitud=piso,
            modelos=filas,
        )

    return app


app = crear_app()
