"""Red neuronal (MLP) en PyTorch, con el mismo contrato que los demas modelos.

El ciclo de entrenamiento es el de siempre: forward, perdida, backward, step y
zero_grad en cada lote. Se detiene por early stopping sobre el AUC de validacion
y se restaura el estado de la mejor epoca.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prediccion_acciones.config import PytorchConfig
from prediccion_acciones.evaluation.validacion import dividir_por_fecha
from prediccion_acciones.models.base import BaseModel, ErrorModelo, registrar

logger = logging.getLogger(__name__)

# Epsilon: una mejora menor que esto es ruido y no reinicia la espera.
EPSILON_AUC = 1e-5


class DatasetAcciones(Dataset):
    """Pares (features, etiqueta) como tensores float32."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class MLP(nn.Module):
    """Capas lineales con ReLU y dropout; devuelve un logit por fila."""

    def __init__(
        self, n_entrada: int, capas_ocultas: Sequence[int], dropout: float
    ) -> None:
        super().__init__()
        capas: list[nn.Module] = []
        previo = n_entrada
        for n in capas_ocultas:
            capas += [nn.Linear(previo, n), nn.ReLU(), nn.Dropout(dropout)]
            previo = n
        capas.append(nn.Linear(previo, 1))
        self.red = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x).squeeze(-1)


class ModeloPytorch(BaseModel):
    """MLP que estandariza sus entradas con la media y escala del entrenamiento."""

    nombre = "pytorch"

    def __init__(
        self,
        columnas: Sequence[str],
        params: PytorchConfig,
        semilla: int = 42,
        gap_dias: int = 5,
        umbral: float = 0.5,
    ) -> None:
        super().__init__(columnas, umbral)
        self.params = params
        self.semilla = semilla
        self.gap_dias = gap_dias
        self.red: MLP | None = None
        self.historia: pd.DataFrame | None = None
        self.epoca_mejor = 0
        self.auc_validacion = float("nan")
        self.estado_optimizador: dict[str, Any] | None = None
        self._media: torch.Tensor | None = None
        self._escala: torch.Tensor | None = None

    def _nueva_red(self) -> MLP:
        return MLP(len(self.columnas), self.params.capas_ocultas, self.params.dropout)

    def _ajustar(
        self, X: pd.DataFrame, y: np.ndarray, fechas: Sequence[Any] | None
    ) -> None:
        if fechas is None:
            raise ErrorModelo(
                "ModeloPytorch necesita `fechas` en fit() para separar la "
                "validacion del early stopping por tiempo"
            )
        p = self.params
        idx_tr, idx_va = dividir_por_fecha(
            np.asarray(fechas), p.fraccion_validacion, self.gap_dias
        )
        X_np = X.to_numpy(dtype=float)
        # El escalador se ajusta solo con el entrenamiento interno: si viera la
        # validacion habria fuga de datos.
        escalador = StandardScaler().fit(X_np[idx_tr])
        X_tr = escalador.transform(X_np[idx_tr])
        X_va = escalador.transform(X_np[idx_va])
        y_tr, y_va = y[idx_tr], y[idx_va]

        torch.manual_seed(self.semilla)
        red = self._nueva_red()
        fn_perdida = nn.BCEWithLogitsLoss()
        optimizador = torch.optim.Adam(
            red.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
        )
        cargador = DataLoader(
            DatasetAcciones(X_tr, y_tr),
            batch_size=p.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.semilla),
        )

        historia = []
        mejor_auc, mejor_epoca, sin_mejora = -np.inf, 0, 0
        mejor_estado, mejor_optimizador = None, None
        for epoca in range(1, p.epocas_max + 1):
            red.train()
            suma, n = 0.0, 0
            for xb, yb in cargador:
                logits = red(xb)  # forward
                perdida = fn_perdida(logits, yb)  # perdida
                perdida.backward()  # backward
                optimizador.step()  # actualiza pesos
                optimizador.zero_grad()  # borra derivadas acumuladas
                suma += perdida.item() * len(yb)
                n += len(yb)

            perdida_va, auc_va = self._evaluar(red, X_va, y_va, fn_perdida)
            historia.append(
                {
                    "epoca": epoca,
                    "perdida_train": suma / n,
                    "perdida_val": perdida_va,
                    "auc_val": auc_va,
                }
            )
            logger.debug(
                "epoca %2d | train %.4f | val %.4f | auc val %.4f",
                epoca,
                suma / n,
                perdida_va,
                auc_va,
            )
            if auc_va > mejor_auc + EPSILON_AUC:
                mejor_auc, mejor_epoca, sin_mejora = auc_va, epoca, 0
                mejor_estado = copy.deepcopy(red.state_dict())
                mejor_optimizador = copy.deepcopy(optimizador.state_dict())
            else:
                sin_mejora += 1
            if sin_mejora >= p.paciencia:
                break

        red.load_state_dict(mejor_estado)  # restaura la mejor epoca
        red.eval()
        self.red = red
        self._media = torch.as_tensor(escalador.mean_, dtype=torch.float32)
        self._escala = torch.as_tensor(escalador.scale_, dtype=torch.float32)
        self.historia = pd.DataFrame(historia)
        self.epoca_mejor = mejor_epoca
        self.auc_validacion = float(mejor_auc)
        self.estado_optimizador = mejor_optimizador
        logger.info(
            "MLP entrenado: mejor epoca %d de %d, AUC validacion %.4f",
            mejor_epoca,
            len(historia),
            mejor_auc,
        )

    @staticmethod
    @torch.no_grad()
    def _evaluar(
        red: MLP, X: np.ndarray, y: np.ndarray, fn_perdida: nn.Module
    ) -> tuple[float, float]:
        red.eval()
        logits = red(torch.as_tensor(X, dtype=torch.float32))
        perdida = fn_perdida(logits, torch.as_tensor(y, dtype=torch.float32)).item()
        auc = roc_auc_score(y, torch.sigmoid(logits).numpy())
        return perdida, float(auc)

    @torch.no_grad()
    def _probabilidad(self, X: pd.DataFrame) -> np.ndarray:
        self.red.eval()
        x = torch.as_tensor(X.to_numpy(dtype=float), dtype=torch.float32)
        x = (x - self._media) / self._escala
        return torch.sigmoid(self.red(x)).numpy()

    def guardar(self, ruta: Path | str) -> None:
        """Un solo archivo: checkpoint completo mas lo que hace falta para predecir.

        Checkpoint: pesos, estado del optimizador, mejor epoca y mejor metrica.
        Para predecir: arquitectura, columnas, escalador y umbral. Se guarda
        tambien la historia por epoca, para poder dibujar las curvas despues.
        """
        ruta = Path(ruta)
        self._exigir_entrenado()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        hiper = asdict(self.params)
        hiper["capas_ocultas"] = list(hiper["capas_ocultas"])
        paquete = {
            "nombre": self.nombre,
            "state_dict": self.red.state_dict(),
            "columnas": list(self.columnas),
            "umbral": self.umbral,
            "hiper": hiper,
            "semilla": self.semilla,
            "gap_dias": self.gap_dias,
            "media": self._media,
            "escala": self._escala,
            "epoca_mejor": self.epoca_mejor,
            "auc_validacion": self.auc_validacion,
            "optimizador": self.estado_optimizador,
            "historia": self.historia.to_dict("list"),
        }
        with ruta.open("wb") as archivo:
            torch.save(paquete, archivo)
        logger.info("Modelo '%s' guardado en %s", self.nombre, ruta)

    @classmethod
    def cargar(cls, ruta: Path | str) -> Self:
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el modelo guardado: {ruta}")
        # weights_only=True: solo acepta tensores y tipos basicos, nunca codigo.
        with ruta.open("rb") as archivo:
            p = torch.load(archivo, weights_only=True, map_location="cpu")
        hiper = dict(p["hiper"])
        hiper["capas_ocultas"] = tuple(hiper["capas_ocultas"])
        modelo = cls(
            p["columnas"],
            PytorchConfig(**hiper),
            p["semilla"],
            p["gap_dias"],
            p["umbral"],
        )
        modelo.red = modelo._nueva_red()
        modelo.red.load_state_dict(p["state_dict"])
        modelo.red.eval()
        modelo._media, modelo._escala = p["media"], p["escala"]
        modelo.epoca_mejor = p["epoca_mejor"]
        modelo.auc_validacion = p["auc_validacion"]
        # Los checkpoints viejos no traen estos dos campos.
        modelo.estado_optimizador = p.get("optimizador")
        if "historia" in p:
            modelo.historia = pd.DataFrame(p["historia"])
        modelo._ajustado = True
        return modelo


@registrar("pytorch")
def _pytorch(
    columnas: Sequence[str],
    semilla: int = 42,
    params: PytorchConfig | None = None,
    gap_dias: int = 5,
    **_: Any,
) -> ModeloPytorch:
    if params is None:
        raise ErrorModelo("El modelo 'pytorch' necesita `params` (PytorchConfig)")
    return ModeloPytorch(columnas, params, semilla, gap_dias)
