# Predicción de la dirección diaria de acciones

Pipeline de Machine Learning de punta a punta (datos, validación, modelos, app y contenedor) que estima la probabilidad de que una acción suba o baje al día siguiente. Compara una regresión logística, un boosting y una red neuronal en PyTorch contra una línea base.

- **Estudiante:** Aldo Gutierrez
- **GitHub:** [AldoEmmanuelG](https://github.com/AldoEmmanuelG) · repositorio: [prediccion-acciones](https://github.com/AldoEmmanuelG/prediccion-acciones)
- **Contexto:** Proyecto final del Módulo 1, Diplomado en AI & LLM for Financial Markets (ITAM).

## Resultado en una línea

Los modelos **no superan al azar de forma demostrable**: el AUC de validación es ≈ 0.51 en todos y en el holdout oscila alrededor de 0.50 (ver [Resultados](#resultados)). Es un resultado esperado al predecir la dirección diaria con datos públicos de precios, y el proyecto lo reporta tal cual en lugar de ocultarlo.

## 1. Instalación

Requisitos: Python 3.12, [uv](https://docs.astral.sh/uv/) y Git. Docker es opcional (solo para el contenedor).

```bash
git clone https://github.com/AldoEmmanuelG/prediccion-acciones.git
cd prediccion-acciones
uv sync
```

Variables de entorno (no hay llaves de API; todo es opcional):

```bash
cp .env.example .env
```

En PowerShell: `Copy-Item .env.example .env`.

Datos y modelos versionados con DVC:

```bash
uv run dvc pull
```

> **Nota sobre el remoto de DVC.** El remoto configurado (`almacen_local`) es una carpeta local en el equipo del autor, como permitió el curso. En otra máquina `dvc pull` no encontrará esa carpeta. Alternativa sin remoto: `uv run dvc repro` vuelve a descargar los precios desde Yahoo Finance y reentrena (ver abajo); el resultado se reproduce con las mismas métricas porque la semilla y las fechas están fijas en `configs/config.yaml`.

## 2. Cómo ejecutar

### Entrenar

```bash
uv run dvc repro
```

Es equivalente a `uv run python -m prediccion_acciones.entrenamiento`, pero DVC solo vuelve a correr si cambiaron los datos, la configuración o el código. Opciones del script directo: `--modelos`, `--sin-mlflow`, `--sin-cache` (vuelve a descargar).

Ver los experimentos en MLflow:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Pruebas y estilo

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

### App (sin Docker, para desarrollo)

```bash
uv run streamlit run src/prediccion_acciones/app/main.py
```

Abre `http://localhost:8501`.

### App con Docker

El orden importa: la imagen incluye la carpeta `models/`, así que primero hay que tener los modelos (`dvc pull` o `dvc repro`) y después construir.

```bash
uv run dvc pull
docker build -t prediccion-acciones .
docker run --rm -p 8501:8501 prediccion-acciones
```

Abre `http://localhost:8501`. Para que otro dispositivo de la misma red entre, usa `http://<IP-de-la-máquina>:8501` y permite el puerto 8501 en el firewall (perfil de red Privado).

## 3. Problema y datos

**Problema.** Clasificación binaria: dado el cierre de hoy, ¿el cierre de mañana será mayor? La etiqueta es `y = 1` si `Close(t+1) > Close(t)`. Se entrena un solo modelo sobre el panel de 10 acciones.

**Acciones objetivo:** AMD, ORCL, IT, IBM, OSIS, LMT, KOF, HUM, ANF, LII. **Contexto** (no se predicen): SPY y NVDA, cuyo retorno entra como característica.

**Fuente:** Yahoo Finance vía `yfinance`, precios ajustados por splits y dividendos, desde 2015-01-01.

**13 características** (solo retornos y razones, nunca precios en bruto; usan datos hasta el cierre del día):

| Grupo | Características |
|---|---|
| Retornos rezagados | `ret_1d`, `ret_2d`, `ret_5d`, `ret_10d` |
| Ventanas móviles (5, 10, 20 días) | `media_*d` y `desv_*d` del retorno |
| Técnicas | `rsi_14`, `dist_media_20d`, `vol_rel` (volumen relativo) |

**Valores faltantes.** No se imputan: se eliminan. Los días sin cierre o volumen se descartan al cargar, las primeras filas de cada acción (donde las ventanas móviles aún no tienen datos) y los valores infinitos se eliminan al construir las características, y los últimos días de cada acción se quitan porque su cierre siguiente aún no se conoce. Imputar con la media del periodo completo metería información del futuro, y con tan pocos huecos descartar es más simple y seguro.

**Validación de datos (Pandera).** Un `DataFrameSchema` revisa tipos, rangos (cierre > 0, volumen ≥ 0), nulos, pares (fecha, ticker) únicos y columnas sin extras. Un cambio diario mayor al 60 % genera una advertencia. Si algo falla, el pipeline se detiene con un mensaje claro.

**Partición temporal (sin fuga de información):**

| Periodo | Fechas | Uso |
|---|---|---|
| Entrenamiento | 2015-02-02 → 2026-03-20 | Ajuste y validación cruzada |
| Holdout | 2026-03-30 → 2026-09-30 (1,280 filas) | Se abre **una sola vez** |
| Hacia adelante | 2026-10-01 → 2026-10-09 (70 filas) | Días posteriores al corte |

La validación cruzada es `TimeSeriesSplit` de 5 pliegues con un hueco (`gap`) de 5 días entre entrenamiento y validación. Los modelos finales se reentrenan con datos hasta la fecha de corte. El umbral de decisión de cada modelo se elige con el índice de Youden sobre las predicciones fuera de muestra de la validación cruzada.

## 4. Resultados

Métricas principales: **AUC-ROC** (¿el modelo ordena bien los días de subida frente a los de bajada?, 0.50 = azar) y **F1** (equilibrio entre precisión y exhaustividad de los días de subida).

| Modelo | AUC validación | F1 validación | AUC holdout | F1 holdout | AUC hacia adelante (n=70) |
|---|---|---|---|---|---|
| Dummy (línea base) | 0.500 | 0.692 | 0.500 | 0.672 | 0.500 |
| Regresión logística | 0.509 ± 0.017 | 0.676 | 0.470 | 0.541 | 0.595 |
| Gradient boosting | 0.508 ± 0.019 | 0.675 | 0.494 | 0.579 | 0.476 |
| MLP (PyTorch) | 0.509 ± 0.012 | 0.657 | 0.454 | 0.545 | 0.590 |

**Conclusión.**

- Los tres modelos tienen el mismo AUC de validación (≈ 0.51), prácticamente el azar. Con 1,280 filas el ruido del AUC en el holdout es de ±0.03, así que las diferencias entre 0.45 y 0.49 no son concluyentes.
- El **F1 del dummy es el más alto (≈ 0.69)** por una razón mecánica: cerca del 53 % de los días suben, y responder siempre "sube" da buen F1 sin ninguna capacidad predictiva. Por eso el AUC es la métrica principal y el F1 se reporta como complemento.
- La prueba hacia adelante tiene solo 70 filas: un AUC de 0.59 ahí es ruido, no evidencia.
- Lectura honesta: con estas características el mercado diario es casi impredecible; la logística y la red son los candidatos que se muestran en la app (el boosting queda en el pipeline y en el código, pero fuera de la app para no saturarla).

### Búsqueda de hiperparámetros

`GridSearchCV` dentro de la validación cruzada temporal, con AUC como métrica de selección (`refit`) y F1 reportado en paralelo (`evaluation/busqueda.py`).

| Modelo | Qué se probó | Qué se encontró |
|---|---|---|
| Logística | `C` ∈ {0.001, 0.01, 0.1, 1, 10} | Ganó `C = 0.001`, el extremo más regularizado de la rejilla |
| Boosting | `max_depth` ∈ {2, 3, 4, 6} × `learning_rate` ∈ {0.01, 0.03, 0.1} | Ganó `max_depth = 2`, `learning_rate = 0.01`, el modelo más simple |
| MLP (PyTorch) | Capas [32, 16], dropout 0.2, Adam (lr 0.001, L2 1e-4), early stopping por AUC con paciencia 10 | Se entrena con configuración fija de `config.yaml`; sin búsqueda |

Ambos ganadores están en el extremo "más simple" de su rejilla. Es la señal típica de que casi no hay patrón que aprender: la validación cruzada prefiere modelos que se parecen al azar.

## 5. Arquitectura

```
Yahoo Finance ──► descarga ──► Pandera ──► features + etiqueta ──► partición temporal
 (yfinance)      (reintentos)  (esquema)     (13 variables)        entrenamiento / holdout / adelante
                                                                          │
                                  ┌───────────────────────────────────────┘
                                  ▼
                 Fábrica de modelos ──► dummy · logística · boosting · MLP (PyTorch)
                                  │
                       validación cruzada + búsqueda ──► umbral (Youden)
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
              MLflow (SQLite)             models/ (.joblib, .pt, metricas.json)
         parámetros, métricas, artefactos         │  (versionado con DVC)
                                                  ▼
                                    Streamlit (app) ──► Docker
```

```
src/prediccion_acciones/
├── config.py            # lee configs/config.yaml (dataclasses inmutables)
├── entrenamiento.py     # orquesta todo el flujo
├── seguimiento.py       # MLflow detrás de una interfaz
├── data/                # descarga, esquemas Pandera, utilidades (decoradores)
├── features/            # ingeniería de variables y partición temporal
├── models/              # BaseModel, fábrica, sklearn, PyTorch
├── evaluation/          # métricas, validación cruzada, búsqueda
└── app/                 # servicio (lógica), gráficas y pantalla de Streamlit
```

## 6. Decisiones de diseño

| Patrón / técnica | Dónde | Por qué |
|---|---|---|
| **Template Method** (clase abstracta `BaseModel`) | `models/base.py` | Define el contrato común (`fit`, `predict_proba`, `predict`, `guardar`, `cargar`) y valida entradas una sola vez; cada modelo solo implementa `_ajustar` y `_probabilidad`. |
| **Factory** (decorador `@registrar`) | `models/base.py` (registro) y `models/fabrica.py` | Crear un modelo por nombre desde la configuración, sin `if/else` en el flujo. Agregar un modelo es registrarlo. |
| **Strategy** | `entrenamiento.py` | El flujo trata igual a sklearn y PyTorch porque comparten la interfaz; cambiar de modelo no cambia el flujo. |
| **Null Object** (`SeguimientoNulo` / `SeguimientoMLflow`) | `seguimiento.py` | Entrenar con o sin MLflow sin llenar el código de `if usar_mlflow`. Los tests corren sin tracking. |
| **Dataclasses inmutables** | `config.py`, resultados | Configuración y resultados con campos nombrados; `frozen=True` evita modificarlos por accidente. |
| **Decoradores** `medir_tiempo`, `reintentar` | `data/utilidades.py` | Tiempo de ejecución de la descarga y reintentos ante fallas de red. |
| **Context manager** `cronometro` | `data/utilidades.py` | Mide bloques de código; `joblib.dump` y `torch.save` también ocurren dentro de `with`. |
| **Separación app / servicio** | `app/servicio.py` vs `app/main.py` | La lógica (cargar artefactos, tablas) se prueba sin levantar Streamlit. |

**Trade-offs.**

- Un solo modelo para las 10 acciones (panel) en vez de uno por acción: más datos por modelo y menos riesgo de sobreajuste, a costa de ignorar diferencias propias de cada acción.
- Características como retornos y razones, no precios: el mismo modelo sirve a todas las acciones, a costa de perder el nivel de precio.
- Se priorizó evitar fugas de información (huecos en la validación, holdout abierto una vez, escalado dentro del `Pipeline`) sobre obtener métricas más vistosas.
- MLflow usa SQLite local y registra parámetros, métricas y el modelo como artefacto (`log_artifact`); no se usa el Model Registry.
- Docker usa `python:3.12-slim` con `uv`, capas ordenadas para aprovechar la caché (dependencias antes que el código), PyTorch solo CPU para reducir tamaño, usuario sin privilegios y `HEALTHCHECK`.

## Aviso

Proyecto académico. No constituye asesoría financiera ni de inversión.
