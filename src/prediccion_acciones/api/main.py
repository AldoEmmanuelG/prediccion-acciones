"""Servicio FastAPI: `GET /health` y `POST /predict`.

El modelo se carga una sola vez, al arrancar el servicio (no en cada petición).
Se levanta con `uvicorn prediccion_acciones.api.main:app`. La documentación
interactiva queda en `/docs`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from prediccion_acciones.api.esquemas import (
    PeticionPrediccion,
    RespuestaPrediccion,
    RespuestaSalud,
)
from prediccion_acciones.app import servicio

logger = logging.getLogger(__name__)

DESCRIPCION = (
    "Estima la probabilidad de que una acción cierre mañana por encima de hoy. "
    "Es un ejercicio académico: con datos públicos de precios el resultado es "
    "muy parecido al azar y no es asesoría de inversión."
)


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

    return app


app = crear_app()
