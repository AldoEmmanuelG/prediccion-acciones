"""Interfaz de Streamlit. Solo muestra; los calculos viven en `servicio.py`.

Se ejecuta con: streamlit run src/prediccion_acciones/app/main.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from prediccion_acciones.app import graficas, servicio
from prediccion_acciones.logging_config import configurar_logging

DIAS_DISPONIBLES = [30, 60, 120, 250]


@st.cache_resource(show_spinner="Cargando modelos y datos...")
def _cargar() -> servicio.Artefactos:
    configurar_logging()
    return servicio.cargar_artefactos()


def _fecha(valor: pd.Timestamp) -> str:
    return valor.strftime("%d/%m/%Y")


def _pestana_resumen(art: servicio.Artefactos, modelo: str) -> None:
    resumen = servicio.resumen_proximo_cierre(art, modelo)
    nombre = servicio.NOMBRES_MODELO[modelo]
    siguiente = resumen["fecha"].max() + pd.offsets.BDay(1)
    st.subheader("Probabilidad de que cada acción suba en el siguiente cierre")
    st.caption(
        f"Ordenadas de mayor a menor. Datos hasta el cierre del "
        f"{_fecha(resumen['fecha'].max())}; la predicción es para el cierre del "
        f"{_fecha(siguiente)}. Modelo: {nombre}."
    )
    alta, baja = resumen.iloc[0], resumen.iloc[-1]
    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Acciones con señal de subida", f"{(resumen['senal'] == 'Sube').sum()} de 10"
    )
    k2.metric("Probabilidad más alta", f"{alta['ticker']} {alta['prob_sube']:.1%}")
    k3.metric("Probabilidad más baja", f"{baja['ticker']} {baja['prob_sube']:.1%}")
    tabla = pd.DataFrame(
        {
            "Acción": resumen["ticker"],
            "Sector": resumen["sector"],
            "Último cierre (USD)": resumen["cierre"],
            "Prob. sube (%)": resumen["prob_sube"] * 100,
            "Probabilidad": resumen["prob_sube"] * 100,
            "Señal": resumen["senal"],
        }
    )
    st.dataframe(
        tabla,
        hide_index=True,
        column_config={
            "Último cierre (USD)": st.column_config.NumberColumn(format="%.2f"),
            "Prob. sube (%)": st.column_config.NumberColumn(format="%.1f"),
            "Probabilidad": st.column_config.ProgressColumn(
                format=" ", min_value=35, max_value=65
            ),
        },
    )


def _pestana_detalle(
    art: servicio.Artefactos,
    modelo: str,
    ticker: str,
    dias: int,
    sma20: bool,
    sma50: bool,
) -> None:
    resumen = servicio.resumen_proximo_cierre(art, modelo).set_index("ticker")
    fila = resumen.loc[ticker]
    historial = servicio.historial_adelante(art, ticker, modelo)

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de que suba", f"{fila['prob_sube'] * 100:.1f} %")
    c2.metric("Señal", fila["senal"])
    c3.metric(
        "Último cierre",
        f"{fila['cierre']:.2f} USD",
        f"{fila['cambio_1d'] * 100:+.2f} % en el día",
    )

    serie = servicio.serie_ticker(art, ticker, dias)
    marcas = historial[historial["fecha"].isin(serie["fecha"])]
    st.pyplot(graficas.grafica_precio(serie, ticker, sma20, sma50, marcas))

    st.subheader("Cómo le ha ido al modelo en días que no vio")
    if historial.empty:
        st.write("Todavía no hay días posteriores a la fecha de corte.")
        return
    aciertos = int(historial["acierto"].sum())
    st.write(
        f"Desde el {_fecha(art.cfg.validacion.fecha_corte)} (fecha de corte del "
        f"entrenamiento) acertó {aciertos} de {len(historial)} días en {ticker}. "
        "Son muy pocos días para sacar conclusiones; abajo, los 10 más recientes."
    )
    mostrar = historial.tail(10).iloc[::-1]
    mostrar = mostrar.assign(
        fecha=mostrar["fecha"].dt.strftime("%d/%m/%Y"),
        prob_sube=(mostrar["prob_sube"] * 100).round(1),
        acierto=mostrar["acierto"].map({True: "Sí", False: "No"}),
    )[["fecha", "prob_sube", "prediccion", "real", "acierto"]]
    mostrar.columns = ["Fecha", "Prob. de subida (%)", "Predicción", "Real", "Acertó"]
    st.dataframe(mostrar, hide_index=True)


def _pestana_modelos(art: servicio.Artefactos) -> None:
    tabla = servicio.tabla_comparacion(art)
    m = art.metricas
    p = m["periodos"]
    st.subheader("Comparación de modelos")
    st.caption(
        "Las barras parten de cero y llegan al AUC de cada modelo en el holdout; la "
        "línea punteada es el azar (0.50)."
    )
    visible = pd.DataFrame(
        {
            "Modelo": tabla["modelo"],
            "AUC validación": tabla["auc_cv"],
            "AUC holdout": tabla["auc_holdout"],
            "AUC hacia adelante": tabla["auc_adelante"],
            "F1 validación": tabla["f1_cv"],
            "F1 holdout": tabla["f1_holdout"],
            "Filas hacia adelante": tabla["n_adelante"],
        }
    )
    st.dataframe(
        visible,
        hide_index=True,
        column_config={
            "AUC validación": st.column_config.NumberColumn(format="%.3f"),
            "AUC holdout": st.column_config.ProgressColumn(
                format="%.4f", min_value=0, max_value=0.6
            ),
            "AUC hacia adelante": st.column_config.NumberColumn(format="%.4f"),
            "F1 validación": st.column_config.NumberColumn(format="%.3f"),
            "F1 holdout": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        f"Validación = 5 pliegues temporales (promedio). Holdout = "
        f"{p['holdout'][0]} a {p['holdout'][1]}, abierto una sola vez. Hacia "
        f"adelante = días posteriores al corte ({m['fecha_corte']}). Entrenamiento: "
        f"{p['entrenamiento'][0]} a {p['entrenamiento'][1]}."
    )

    izq, der = st.columns(2)
    with izq:
        st.subheader("Qué significan las métricas")
        st.markdown(
            "**AUC.** Si tomo un día que subió y uno que bajó, ¿con qué frecuencia "
            "el modelo le da mayor probabilidad al que subió? 0.50 es lanzar una "
            "moneda; 1.00 sería perfecto.\n\n"
            "**F1.** Equilibrio entre acertar los días de subida y no dar falsas "
            'alarmas. Engaña aquí: como ~53 % de los días suben, decir siempre "sube" '
            "ya da un F1 cercano a 0.69.\n\n"
            "**Holdout.** Seis meses que ningún modelo vio al entrenar ni al elegir "
            "hiperparámetros. Es el examen final y se abre una sola vez.\n\n"
            "**Hacia adelante.** Días posteriores al corte. Son pocos, así que su "
            "AUC oscila mucho y no demuestra nada por sí solo."
        )
    with der:
        st.pyplot(graficas.barras_modelos(tabla))

    st.subheader("Lectura honesta del resultado")
    st.markdown(
        "**Con estas 13 variables de precio y volumen, ningún modelo predice mejor "
        "que el azar.**"
    )
    st.write(
        "En validación los modelos empatan con el dummy en un AUC cercano a 0.51. "
        "En el holdout, con alrededor de 1,300 filas, el margen de error es de "
        "±0.03: los resultados quedan en o apenas debajo de esa zona de ruido, sin "
        "ser una señal real. Los valores más altos de la prueba hacia adelante "
        "salen de muy pocos días y no se sostienen en el holdout."
    )
    st.write(
        "Esto es lo esperado: la dirección de un día en acciones líquidas es casi "
        "impredecible con indicadores técnicos. Por eso las probabilidades de la app "
        "se quedan cerca de 50 % y por eso se reporta el piso (dummy) junto a cada "
        "modelo: el valor del proyecto está en un proceso sin fuga de información, "
        "no en una señal de trading."
    )


# Quita la barra flotante de las tablas (descargar CSV, buscar, pantalla completa).
OCULTAR_BARRA_TABLAS = (
    "<style>[data-testid='stElementToolbar'] {display: none;}</style>"
)


def main() -> None:
    st.set_page_config(page_title="Predicción diaria de acciones", layout="wide")
    st.markdown(OCULTAR_BARRA_TABLAS, unsafe_allow_html=True)
    try:
        art = _cargar()
    except servicio.ErrorArtefactos as error:
        st.error(str(error))
        st.stop()

    st.title("Predicción diaria")
    st.write(
        "¿Sube o baja la acción en el siguiente cierre? Elige una acción y un "
        "modelo en el panel de la izquierda: la pestaña *Resumen* compara las 10 "
        "acciones, *Detalle* muestra una con su historial y *Acerca del modelo* "
        "explica qué tan bien funcionan los modelos."
    )

    with st.sidebar:
        st.header("Controles")
        ticker = st.selectbox("Acción", art.tickers)
        modelo = st.radio(
            "Modelo",
            art.modelos_app,
            format_func=lambda k: servicio.NOMBRES_MODELO.get(k, k),
        )
        dias = st.select_slider("Historial (días hábiles)", DIAS_DISPONIBLES, value=60)
        sma20 = st.checkbox("Media móvil 20 días", value=True)
        sma50 = st.checkbox("Media móvil 50 días", value=False)
        st.caption(
            "Datos de Yahoo Finance con ajuste por splits y dividendos, hasta "
            f"el {_fecha(art.ultima_fecha)}."
        )

    resumen, detalle, modelos = st.tabs(
        ["Resumen de las 10", "Detalle por acción", "Acerca del modelo"]
    )
    with resumen:
        _pestana_resumen(art, modelo)
    with detalle:
        _pestana_detalle(art, modelo, ticker, dias, sma20, sma50)
    with modelos:
        _pestana_modelos(art)


main()
