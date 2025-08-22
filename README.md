# Previsão de Demanda com Machine Learning e Dashboard Interativo
## Sobre o Projeto

Este projeto é uma solução de ponta a ponta para um desafio de previsão de demanda, utilizando um dataset do Kaggle com 5 anos de histórico de vendas de 50 produtos em 10 lojas diferentes. O objetivo foi transformar dados brutos em uma ferramenta de negócio acionável, capaz de prever vendas futuras e fornecer insights estratégicos e operacionais.
A análise completa, desde a exploração dos dados até a otimização de modelos, foi consolidada em um **dashboard interativo construído com Streamlit**, permitindo a análise granular das previsões.

---
## Principais Funcionalidades

- **Análise Exploratória Dinâmica:** Visualizações interativas da performance histórica de vendas por loja e por item.
- **Previsão de Demanda Granular:** Geração de previsões de curto prazo para qualquer combinação de loja e item, utilizando um modelo LightGBM otimizado.
- **Análise Estratégica Agregada:** Previsões de longo prazo e decomposição da série temporal (tendência e sazonalidade) com o modelo Prophet para uma visão macro do negócio.
- **Dashboard Interativo:** Uma interface web amigável que permite a exploração dos dados e previsões sem a necessidade de executar código.
---
## Tecnologias Utilizadas

Este projeto foi construído utilizando um ecossistema moderno de ferramentas de Ciência de Dados em Python:

- **Gerenciamento de Ambiente:** Poetry
- **Análise de Dados:** Pandas, NumPy
- **Modelagem Preditiva:**
  - **Machine Learning:** Scikit-learn, LightGBM
  - **Estatística:** Prophet, Statsmodels
- **Otimização de Hiperparâmetros:** Optuna
- **Interpretabilidade de Modelos (XAI):** SHAP
- **Visualização de Dados:** Plotly, Matplotlib, Seaborn
- **Dashboard Interativo:** Streamlit
- **Controle de Versão:** Git & GitHub
---
