import streamlit as st
import pandas as pd
from config import opcoes_granularidade, periodos_granularidade
from src.data_processing import df_train, df_test, df_ML, df_ML_test, df_resultados
from src.prediction import previsao_prophet, previsao_lgbm
from src.plotting import performance_lojas_itens, fig_prophet, fig_lgbm

#Configuração de Iniciais

st.title("Dashboard de Previsão de Demanda")
# --- Bloco de CSS para um layout mais compacto ---

st.markdown("""
    <style>
        .block-container {padding-top: 2.0rem;}
        div[data-testid="stVerticalBlock"] {gap: 0.20rem;}
        h1 {margin-bottom: 0rem;}
    </style>""", 
    unsafe_allow_html=True)

st.set_page_config(layout='wide')
st.sidebar.title("Navegação")
pagina_selecionada = st.sidebar.radio("",["📈 Análise Exploratória dos Dados", 
                                       "📄 Previsão de Demanda"])

if 'pagina_selecionada' not in st.session_state:
    st.session_state['pagina_selecionada'] = "📈 Análise Exploratória dos Dados"

# VISÃO GERAL: EDA + PROPHET
if pagina_selecionada == "📈 Análise Exploratória dos Dados": 
    meta = 40000000
    vendas_totais = df_train['sales'].sum()
    diferenca = round(vendas_totais/meta,2)
    vendas_totais_formatado = f'R$ {vendas_totais:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    st.metric("Vendas Total",vendas_totais_formatado, delta=f"{diferenca:,}".replace(",", ".") + " vs. Meta")
    
    #Divisão das colunas
    eda_col_21, eda_col_22 = st.columns([1,3])
    with eda_col_21:
        # Criando Figura
        fig_performance_lojas = performance_lojas_itens(df = df_train,
                                tipificador = 'store',
                                x= 'sales',
                                y= 'store',
                                orientacao = 'h',
                                titulo = 'Total de Vendas por Lojas (2013-2017)')
        # Plotando Figura
        st.plotly_chart(fig_performance_lojas,use_container_width = True)
    with eda_col_22:
        # Criando Figura
        fig_performance_itens = performance_lojas_itens(df = df_train,
                                tipificador = 'item',
                                x= 'item',
                                y= 'sales',
                                orientacao = 'v',
                                titulo = 'Total de Vendas por Item (2013-2017)')
        # Plotando Figura
        st.plotly_chart(fig_performance_itens,use_container_width = True)
    st.markdown("---")
    # Trabalhando viz prophet
    lista_opcoes = list(opcoes_granularidade.keys())
    # Divisão em colunas de apoio
    eda_col_filtro, col_vazia_1 = st.columns([0.3,0.7])

    with eda_col_filtro:
        granularidade_label = st.selectbox("Selecione a Granularidade:",options = lista_opcoes)
    freq_selecionada = opcoes_granularidade[granularidade_label]
    periodos_para_prever = periodos_granularidade[freq_selecionada]
    
    df_prophet, forecast = previsao_prophet(df = df_train,
                                            periodo = periodos_para_prever,
                                            granularidade= freq_selecionada)
    st.plotly_chart(fig_prophet(df_prophet= df_prophet,
                                forecast = forecast,
                                referencia = '2018-01-01'),
                                use_container_width = True)
    st.caption("Previsão de Vendas (Prophet).")

# PREVISÃO DE DEMANADA: LGBM
if pagina_selecionada == "📄 Previsão de Demanda":
    pd_col_11, col_vazia2 = st.columns(2)
    pd_col_21, pd_col_22 = st.columns(2)
    with pd_col_21:
        lojas = [0] + sorted(df_train['store'].unique().tolist())
        loja_selecionada = st.selectbox("Selecione a Loja", options = lojas)
        itens = [0] + sorted(df_train['item'].unique().tolist())
        item_selecionado = st.selectbox("Selecione o Item", options = itens)
    pd_col_31, pd_col_32 = st.columns(2)
    with pd_col_31:
        previsao, previsao_agg, previsao_past,df_futuro = previsao_lgbm(df = df_ML_test,
                                                            loja = loja_selecionada,
                                                            item = item_selecionado,
                                                            agregacao= 'D',
                                                            df_past = df_train)
        
        fig_lgbm = fig_lgbm(previsao_agg = previsao_agg, previsao_past = previsao_past)
        st.plotly_chart(fig_lgbm, use_container_width = True)
        st.caption("Previsão de Vendas distribuida por dia (LGBM).")
    with pd_col_32:
        with st.expander("**Clique aqui para ver a Análise de Performance Detalhada do Modelo**", expanded= True):
            st.markdown("""
    O modelo consegue explicar 93%  toda a variabilidade presente nas vendas.
                        
    O Erro Médio Absoluto (MAE) nos diz que, em média, as previsões do modelo no conjunto de teste erraram por aproximadamente **6.1 unidades**.
                        
    O modelo não tem uma tendência a cometer erros muito grandes e esporádicos, apresentando uma distribuição de erros consistente.
    """)
            st.dataframe(df_resultados)
    with pd_col_11:
        previsao_prox_30 = previsao_agg.head(30).sum()
        delta_lgbm = (previsao_prox_30 - previsao_past.sum())
        valor_venda_previsao = f'R$ {previsao_agg.head(30).sum():,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
        st.metric( label = "Valor de Venda Previsto",
                value = valor_venda_previsao,
                delta = round(delta_lgbm,2))
    delta_formatado = f'R$ {abs(delta_lgbm):,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    if delta_lgbm < 0:
        st.warning(f"⚠️ **Alerta:** A previsão indica uma **queda** de **{delta_formatado}** em comparação com o período anterior.")
    elif delta_lgbm > 0:
        st.success(f"✅ **Positivo:** A previsão indica um **aumento** de **{delta_formatado}** em comparação com o período anterior.")
    else:
        st.info("ℹ️ A previsão indica **estabilidade**, sem mudanças significativas em comparação com o período anterior.")
    st.markdown("---")
    #Divisão das colunas
    pd_col_41, pd_col_42 = st.columns([1,3])
    with pd_col_41:
        # Criando Figura
        fig_performance_lojas = performance_lojas_itens(df = df_futuro,
                                tipificador = 'store',
                                x= 'sales',
                                y= 'store',
                                orientacao = 'h',
                                titulo = 'Total de Vendas por Lojas (previsto próx 30d)')
        # Plotando Figura
        st.plotly_chart(fig_performance_lojas,use_container_width = True)
    with pd_col_42:
        # Criando Figura
        fig_performance_itens = performance_lojas_itens(df = df_futuro,
                                tipificador = 'item',
                                x= 'item',
                                y= 'sales',
                                orientacao = 'v',
                                titulo = 'Total de Vendas por Item (previsto próx 30d)')
        # Plotando Figura
        st.plotly_chart(fig_performance_itens,use_container_width = True)
    