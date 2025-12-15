# Painel de Inteligência Financeira - VPL Consultoria

> Um dashboard interativo para análise de indicadores econômicos, correção monetária e monitoramento de câmbio em tempo real.

![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

##  Sobre o Projeto

Este projeto é uma aplicação web desenvolvida em **Python** utilizando o framework **Streamlit**. O objetivo é centralizar dados econômicos vitais do cenário brasileiro para auxiliar na tomada de decisões financeiras.

O sistema coleta dados automaticamente de fontes oficiais (BCB, IBGE) e de mercado (Yahoo Finance), permitindo:
- Visualizar a evolução de índices (IPCA, SELIC, IGP-M, CDI, INPC).
- Calcular correções monetárias de valores passados.
- Acompanhar a cotação do Dólar e Euro (Intraday e Histórico desde 1994).
- Consultar expectativas de mercado (Boletim Focus).

##  Funcionalidades Principais

* **📈 Monitor de Índices:** Gráficos interativos (Plotly) com histórico de 12 meses e acumulados anuais.
* **🧮 Calculadora Financeira:** Ferramenta para corrigir ou descapitalizar valores com base no índice selecionado (ex: Quanto R$ 1.000 de 2015 valeriam hoje corrigidos pelo IPCA?).
* **💸 Câmbio Avançado:**
    * Cotação em tempo real.
    * Histórico completo desde o Plano Real (1994).
    * **Matriz de Retornos:** Mapa de calor (Heatmap) mostrando a rentabilidade mensal das moedas.
* **📥 Exportação de Dados:** Botões para download das tabelas em formato CSV.
* **🔭 Boletim Focus:** Integração com a API Olinda do BCB para exibir as previsões de mercado para o final do ano.

## 🛠️ Tecnologias Utilizadas

* **[Streamlit](https://streamlit.io/):** Interface web interativa.
* **[Pandas](https://pandas.pydata.org/):** Manipulação e tratamento de dados.
* **[Plotly](https://plotly.com/python/):** Visualização de dados e gráficos interativos.
* **[yFinance](https://pypi.org/project/yfinance/):** Dados de mercado (Câmbio).
* **[Sidrapy](https://pypi.org/project/sidrapy/):** API do IBGE (SIDRA).
* **Requests:** Integração com APIs do Banco Central do Brasil (SGS).

## 🔌 Fontes de Dados

A transparência dos dados é garantida através de conexões diretas com:
1.  **IBGE (SIDRA):** Para índices de inflação (IPCA, INPC).
2.  **Banco Central do Brasil (SGS):** Para taxas de juros e índices financeiros (SELIC, CDI, IGP-M).
3.  **Banco Central do Brasil (Olinda):** Para expectativas do Boletim Focus.
4.  **Yahoo Finance:** Para dados cambiais (USD/BRL, EUR/BRL).

## 💻 Como Rodar o Projeto Localmente

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/nome-do-repo.git](https://github.com/seu-usuario/nome-do-repo.git)
   cd nome-do-repo
2. **Crie um ambiente virtual (Opcional, mas recomendado):**   
   python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
3. **Instale as dependências:**
pip install -r requirements.txt
4. **Execute a aplicação:**
streamlit run app.py
