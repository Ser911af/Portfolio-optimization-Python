import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Función para calcular el portafolio
def calcular_portafolio():
    tickers = ["BTC-USD", "ETH-USD", "BABA", "GC=F", "CL=F", "^IXIC", "MSFT", "AAPL", "SOL1-USD", "NIO", "UPWK", "SQM", "GOOGL", "META", "BIDU"]
    
    try:
        # Obtener los datos históricos de precios para cada ticker
        data = yf.download(tickers, start="2010-01-01", end="2024-04-10")

        # Seleccionar solo las columnas de precios de cierre ajustados
        cierres_ajustados = data['Adj Close']

        # Calcular los rendimientos diarios
        rendimientos_diarios = cierres_ajustados.pct_change(fill_method=None).dropna()

        # Calcular la matriz de covarianza
        covarianza = rendimientos_diarios.cov()

        # Calcular el rendimiento esperado como la media histórica de los rendimientos diarios
        rendimiento_esperado = rendimientos_diarios.mean() * 252  # Multiplicamos por 252 para convertir a rendimiento anual

        # Calcular la volatilidad como la desviación estándar histórica de los rendimientos diarios
        volatilidad = rendimientos_diarios.std() * np.sqrt(252)  # Multiplicamos por raíz cuadrada de 252 para convertir a volatilidad anual

        # Simular pesos aleatorios para los activos en el portafolio
        num_portfolios = 10000
        pesos_aleatorios = np.random.random((num_portfolios, len(tickers)))
        pesos_aleatorios /= pesos_aleatorios.sum(axis=1, keepdims=True)

        # Calcular rendimiento esperado y volatilidad para cada combinación de pesos
        rendimientos_portafolio = np.dot(pesos_aleatorios, rendimiento_esperado)
        volatilidades_portafolio = np.sqrt(np.diag(np.dot(np.dot(pesos_aleatorios, covarianza), pesos_aleatorios.T)))

        # Encontrar el índice del portafolio con el mayor ratio de Sharpe
        indice_max_sharpe = np.argmax((rendimientos_portafolio - 0.02) / volatilidades_portafolio)

        # Obtener los pesos óptimos del portafolio con el mayor ratio de Sharpe
        pesos_optimos = pesos_aleatorios[indice_max_sharpe]

        # Calcular el rendimiento y la volatilidad óptimos
        rendimiento_optimo = rendimientos_portafolio[indice_max_sharpe]
        volatilidad_optima = volatilidades_portafolio[indice_max_sharpe]

        return rendimiento_esperado, volatilidad, pesos_optimos, rendimiento_optimo, volatilidad_optima

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
        return None, None, None, None, None

# Función para generar y guardar los gráficos como archivos PNG
def generate_and_save_plots():
    # Calcular el portafolio óptimo y obtener los datos necesarios
    rendimiento_esperado, volatilidad, pesos_optimos, rendimiento_optimo, volatilidad_optima = calcular_portafolio()

    if rendimiento_esperado is None or volatilidad is None or pesos_optimos is None:
        st.error("No se pudo calcular el portafolio óptimo.")
        return

    # Verificación y depuración
    st.write("Tipo de rendimiento_esperado:", type(rendimiento_esperado))
    st.write("Contenido de rendimiento_esperado:", rendimiento_esperado)
    st.write("Tipo de volatilidad:", type(volatilidad))
    st.write("Contenido de volatilidad:", volatilidad)
    st.write("Pesos óptimos:", pesos_optimos)

    # Convertir pesos_optimos a un DataFrame de pandas para usar el índice
    df_pesos_optimos = pd.DataFrame(pesos_optimos, index=rendimiento_esperado.index, columns=['Ponderacion'])

    # Gráfico de rendimiento esperado anual de los activos
    fig_expected_returns = go.Figure()
    fig_expected_returns.add_trace(go.Bar(x=rendimiento_esperado.index, y=rendimiento_esperado.values, name='Rendimiento esperado anual'))
    fig_expected_returns.update_layout(title='Rendimiento esperado anual de los activos', xaxis_title='Activo', yaxis_title='Rendimiento')
    fig_expected_returns.write_image("fig_expected_returns.png")

    # Gráfico de volatilidad anual de los activos
    fig_volatility = go.Figure()
    fig_volatility.add_trace(go.Bar(x=volatilidad.index, y=volatilidad.values, name='Volatilidad anual'))
    fig_volatility.update_layout(title='Volatilidad anual de los activos', xaxis_title='Activo', yaxis_title='Volatilidad')
    fig_volatility.write_image("fig_volatility.png")

    # Gráfico de relación entre rendimiento y volatilidad anual
    fig_risk_return = go.Figure()
    fig_risk_return.add_trace(go.Scatter(x=volatilidad, y=rendimiento_esperado, mode='markers', text=rendimiento_esperado.index, marker=dict(size=10, color='blue'), name='Activos'))
    fig_risk_return.update_layout(title='Relación entre rendimiento y volatilidad anual', xaxis_title='Volatilidad anual', yaxis_title='Rendimiento esperado')
    fig_risk_return.write_image("fig_risk_return.png")

    # Gráfico de barras de ponderaciones óptimas de portafolio
    fig_weights_bar = go.Figure()
    fig_weights_bar.add_trace(go.Bar(x=df_pesos_optimos.index, y=df_pesos_optimos['Ponderacion'], name='Ponderaciones óptimas'))
    fig_weights_bar.update_layout(title='Ponderaciones óptimas de portafolio', xaxis_title='Activo', yaxis_title='Ponderación')
    fig_weights_bar.write_image("fig_weights_bar.png")

    # Gráfico de torta de ponderaciones óptimas de portafolio
    fig_weights_pie = go.Figure()
    fig_weights_pie.add_trace(go.Pie(labels=df_pesos_optimos.index, values=df_pesos_optimos['Ponderacion'], name='Ponderaciones óptimas'))
    fig_weights_pie.update_layout(title='Ponderaciones óptimas de portafolio')
    fig_weights_pie.write_image("fig_weights_pie.png")

    # Frontera eficiente de Markowitz con el portafolio óptimo
    fig_efficient_frontier = go.Figure()
    fig_efficient_frontier.add_trace(go.Scatter(x=volatilidad, y=rendimiento_esperado, mode='markers', text=rendimiento_esperado.index, marker=dict(size=10, color='blue'), name='Activos'))
    fig_efficient_frontier.add_trace(go.Scatter(x=[volatilidad_optima], y=[rendimiento_optimo], mode='markers', marker=dict(size=12, color='red'), name='Portafolio óptimo'))
    fig_efficient_frontier.update_layout(title='Frontera eficiente de Markowitz', xaxis_title='Volatilidad anual', yaxis_title='Rendimiento esperado')
    fig_efficient_frontier.write_image("fig_efficient_frontier.png")

# Generar y guardar los gráficos
generate_and_save_plots()

st.write("Gráficos guardados como archivos PNG.")
