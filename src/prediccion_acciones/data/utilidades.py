"""Decoradores y context manager reutilizables para la capa de datos."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

Funcion = Callable[..., Any]


def medir_tiempo(funcion: Funcion) -> Funcion:
    """Decorador: registra en DEBUG cuanto tardo la funcion en ejecutarse."""

    @functools.wraps(funcion)
    def envoltura(*args: Any, **kwargs: Any) -> Any:
        inicio = time.perf_counter()
        try:
            return funcion(*args, **kwargs)
        finally:
            logger.debug(
                "%s tardo %.2f s", funcion.__name__, time.perf_counter() - inicio
            )

    return envoltura


def reintentar(
    intentos: int = 3,
    espera: float = 1.0,
    factor: float = 2.0,
    excepciones: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Funcion], Funcion]:
    """Decorador con parametros: repite la funcion si falla con alguna excepcion.

    La espera crece por `factor` en cada intento (espera exponencial). Cada
    falla recuperable se registra en WARNING; si se agotan los intentos, se
    vuelve a lanzar la ultima excepcion.
    """
    if intentos < 1:
        raise ValueError("intentos debe ser >= 1")

    def decorador(funcion: Funcion) -> Funcion:
        @functools.wraps(funcion)
        def envoltura(*args: Any, **kwargs: Any) -> Any:
            pausa = espera
            for numero in range(1, intentos + 1):
                try:
                    return funcion(*args, **kwargs)
                except excepciones as exc:
                    if numero == intentos:
                        logger.error(
                            "%s fallo en el intento %d de %d: %s",
                            funcion.__name__,
                            numero,
                            intentos,
                            exc,
                        )
                        raise
                    logger.warning(
                        "%s fallo en el intento %d de %d (%s); reintento en %.1f s",
                        funcion.__name__,
                        numero,
                        intentos,
                        exc,
                        pausa,
                    )
                    time.sleep(pausa)
                    pausa *= factor

        return envoltura

    return decorador


@contextmanager
def cronometro(etiqueta: str) -> Iterator[None]:
    """Context manager: avisa cuando empieza y termina un bloque de trabajo."""
    logger.info("Inicia: %s", etiqueta)
    inicio = time.perf_counter()
    try:
        yield
    finally:
        logger.info("Termina: %s (%.2f s)", etiqueta, time.perf_counter() - inicio)
