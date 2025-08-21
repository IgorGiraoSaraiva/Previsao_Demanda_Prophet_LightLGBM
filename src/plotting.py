import plotly.express as px
import plotly.graph_objects as go
import config
import pandas as pd
import streamlit as st
from src.prediction import lgbm_otimo, modelo_prophet
from prophet import Prophet
import holidays

@st.cache_data
def performance_lojas_itens(df,tipificador,x,y,orientacao,titulo): 
    vendas_agg = df.groupby(tipificador)['sales'].sum().sort_values().reset_index()
    vendas_agg[tipificador] = vendas_agg[tipificador].astype(str)
    media_vendas = vendas_agg['sales'].mean()
    fig_performance = px.bar( 
        data_frame=vendas_agg,
        y= y,
        x= x,
        title= titulo,
        labels={
            tipificador: tipificador,
            'sales': 'Total de Vendas (em milhões)'
        },
        text='sales',
        orientation = orientacao
    )
    fig_performance.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    if orientacao == 'v':
        # Para barras verticais, a média é uma linha HORIZONTAL
        fig_performance.add_hline(
            y=media_vendas, 
            line_dash="dot",
            line_color="red")
    else: 
        # Para barras horizontais, a média é uma linha VERTICAL
        fig_performance.add_vline(
            x=media_vendas,
            line_dash="dot",
            line_color="red")
        # Força a exibição de todos os rótulos no eixo Y
        fig_performance.update_yaxes(tickmode='array', tickvals=vendas_agg[tipificador])
    return fig_performance


def fig_prophet(df_prophet, forecast,referencia):
    # Plotando o resultado final de forma interativa com Plotly
    fig_prophet = go.Figure()
    # Adicionar a linha de Vendas Reais (Histórico)
    fig_prophet.add_trace(go.Scatter(
        x=df_prophet['ds'],
        y=df_prophet['y'],
        mode='lines',
        name='Histórico de Vendas',
        line=dict(color='blue')
    ))
    # Adicionar a linha de Previsões do Prophet
    previsao_futura = forecast[forecast['ds'] > df_prophet['ds'].max()]
    fig_prophet.add_trace(go.Scatter(
        x=previsao_futura['ds'],
        y=previsao_futura['yhat'], 
        mode='lines',
        name='Previsão de  Vendas',
        line=dict(color='red', dash='dash')
    ))
    #Adicionando data de referência
    fig_prophet.add_vline(x = referencia, line_dash = 'dot')

    # Customizar o layout
    fig_prophet.update_layout(
        title= dict(text = 'Comportamento e Projeção de Vendas', 
                    font=dict(size = 25)),
        xaxis_title='Período',
        yaxis_title='Valor Total de Vendas',
    )
    return fig_prophet


def fig_lgbm(previsao_agg,previsao_past):
    fig_lgbm = go.Figure()

    # Adiciona o histórico
    fig_lgbm.add_trace(go.Scatter(
        x=previsao_past.index, y=previsao_past.values, 
        mode='lines', name='Histórico', line = dict(color='blue')
    ))

    # Adiciona a previsão
    fig_lgbm.add_trace(go.Scatter(
        x=previsao_agg.index, y=previsao_agg.values, 
        mode='lines', name='Previsão (próximos 30d)', line = dict(dash='dash', color='red')
    ))
    fig_lgbm.add_vline(x = '2018-01-01', line_dash = 'dot')
    fig_lgbm.update_layout(
        title= dict(text = 'Previsão de Demanda (30 dias)',font=dict(size = 25)),
        xaxis_title='Data',
        yaxis_title='Vendas'
    )
    
    return fig_lgbm