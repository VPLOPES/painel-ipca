import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
import requests
import yfinance as yf

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide"
)

# Estilo CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
    }
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        font-weight: 600;
        color: #003366;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE CARGA DE DADOS ---

# 1. IBGE (Sidra)
@st.cache_data
def get_sidra_data(table_code, variable_code):
    try:
        dados_raw = sidrapy.get_table(
            table_code=table_code, territorial_level="1", ibge_territorial_code="all", 
            variable=variable_code, period="last 360"
        )
        df = dados_raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'])
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m")
        df['ano'] = df['D2C'].str.slice(0, 4)
        return processar_dataframe_comum(df)
    except Exception as e:
        return pd.DataFrame()

# 2. Banco Central (SGS - Índices Mensais)
@st.cache_data
def get_bcb_data(codigo_serie):
    try:
        url = f"http://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
        df = pd.read_json(url)
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        return processar_dataframe_comum(df)
    except:
        return pd.DataFrame()

# 3. Boletim Focus
@st.cache_data(ttl=3600)
def get_focus_data():
    try:
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=1000&$orderby=Data%20desc&$format=json"
        response = requests.get(url)
        data_json = response.json()
        df = pd.DataFrame(data_json['value'])
        indicadores = ['IPCA', 'PIB Total', 'Selic', 'Câmbio']
        df = df[df['Indicador'].isin(indicadores)]
        df = df.rename(columns={'Data': 'data_relatorio', 'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'})
        df['ano_referencia'] = df['ano_referencia'].astype(int)
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        return df
    except:
        return pd.DataFrame()

# 4. Cotação de Moedas (Tempo Real)
@st.cache_data(ttl=300)
def get_currency_realtime():
    try:
        # Tickers do Yahoo: Dólar/Real e Euro/Real
        tickers = ["USDBRL=X", "EURBRL=X"]
        dados = {}
        
        for t in tickers:
            ticker_obj = yf.Ticker(t)
            # fast_info é mais rápido e raramente falha
            preco_atual = ticker_obj.fast_info['last_price']
            fechamento_anterior = ticker_obj.fast_info['previous_close']
            
            # Calculamos a variação percentual manualmente
            variacao = ((preco_atual - fechamento_anterior) / fechamento_anterior) * 100
            
            # Mapeia para o formato que seu painel já espera (USDBRL e EURBRL)
            key = t.replace("=X", "") 
            dados[key] = {
                'bid': preco_atual,
                'pctChange': variacao
            }
            
        df = pd.DataFrame.from_dict(dados, orient='index')
        return df
    except Exception as e:
        print(f"Erro Yahoo Finance Realtime: {e}")
        return pd.DataFrame()

# 5. NOVO: Histórico de Câmbio (BCB - Séries Diárias)
@st.cache_data(ttl=86400)
def get_cambio_historico():
    try:
        # Baixa dados desde o início do plano real (1994)
        # USDBRL=X é o ticker padrão. O Euro começou depois, mas o Yahoo gerencia os nulos.
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="1994-07-01", progress=False)
        
        # O yfinance retorna um MultiIndex nas colunas (Price, Ticker). 
        # Vamos pegar apenas o 'Close' (Fechamento)
        df = df['Close']
        
        # Renomeia para ficar bonito no gráfico
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        
        # Remove timezone se houver (para evitar erros de plotagem)
        df.index = df.index.tz_localize(None)
        
        # Preenche dias sem negociação (fins de semana) com o valor anterior
        df = df.ffill()
        
        return df
    except Exception as e:
        print(f"Erro Yahoo Finance Histórico: {e}")
        return pd.DataFrame()

# 6. Processamento Comum
def processar_dataframe_comum(df):
    df = df.sort_values('data_date', ascending=True)
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12).apply(np.prod, raw=True) - 1) * 100
    return df.sort_values('data_date', ascending=False)

# --- CÁLCULO ---
def calcular_correcao(df, valor, data_ini_code, data_fim_code):
    is_reverso = data_ini_code > data_fim_code
    if is_reverso:
        periodo_inicio, periodo_fim = data_fim_code, data_ini_code
    else:
        periodo_inicio, periodo_fim = data_ini_code, data_fim_code
    
    mask = (df['D2C'] >= periodo_inicio) & (df['D2C'] <= periodo_fim)
    df_periodo = df.loc[mask].copy()
    
    if df_periodo.empty:
        return None, "Período sem dados suficientes."
    
    fator_acumulado = df_periodo['fator'].prod()
    valor_final = valor / fator_acumulado if is_reverso else valor * fator_acumulado
    pct_total = (fator_acumulado - 1) * 100
    return {
        'valor_final': valor_final, 'percentual': pct_total, 'fator': fator_acumulado, 'is_reverso': is_reverso
    }, None

# ==============================================================================
# LAYOUT - SIDEBAR
# ==============================================================================
st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
st.sidebar.header("Configurações")

tipo_indice = st.sidebar.selectbox(
    "Selecione o Indicador",
    ["IPCA (Inflação Oficial)", "INPC (Salários)", "IGP-M (Aluguéis)", "SELIC (Taxa Básica)", "CDI (Investimentos)"]
)

with st.spinner(f"Carregando dados..."):
    if "IPCA" in tipo_indice:
        df = get_sidra_data("1737", "63")
        cor_tema = "#003366"
    elif "INPC" in tipo_indice:
        df = get_sidra_data("1736", "44")
        cor_tema = "#2E8B57"
    elif "IGP-M" in tipo_indice:
        df = get_bcb_data("189")
        cor_tema = "#8B0000"
    elif "SELIC" in tipo_indice:
        df = get_bcb_data("4390")
        cor_tema = "#DAA520"
    elif "CDI" in tipo_indice:
        df = get_bcb_data("4391")
        cor_tema = "#333333"

if df.empty:
    st.error("Erro ao carregar dados.")
    st.stop()

# --- CALCULADORA ---
st.sidebar.divider()
st.sidebar.subheader("Calculadora")
valor_input = st.sidebar.number_input("Valor (R$)", value=1000.00, step=100.00, format="%.2f")

lista_anos = sorted(df['ano'].unique(), reverse=True)
lista_meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
mapa_meses = {'Jan': '01', 'Fev': '02', 'Mar': '03', 'Abr': '04', 'Mai': '05', 'Jun': '06',
              'Jul': '07', 'Ago': '08', 'Set': '09', 'Out': '10', 'Nov': '11', 'Dez': '12'}

st.sidebar.markdown("**Data Referência**")
c1, c2 = st.sidebar.columns(2)
mes_ini = c1.selectbox("Mes Ini", lista_meses_nome, index=0, label_visibility="collapsed")
ano_ini = c2.selectbox("Ano Ini", lista_anos, index=1 if len(lista_anos)>1 else 0, label_visibility="collapsed")

st.sidebar.markdown("**Data Alvo**")
c3, c4 = st.sidebar.columns(2)
mes_fim = c3.selectbox("Mes Fim", lista_meses_nome, index=9, label_visibility="collapsed")
ano_fim = c4.selectbox("Ano Fim", lista_anos, index=0, label_visibility="collapsed")

if st.sidebar.button("Calcular", type="primary"):
    code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
    code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
    res, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
    
    if erro:
        st.error(erro)
    else:
        st.sidebar.divider()
        nome_indice = tipo_indice.split()[0]
        tipo_op = "Rendimento" if nome_indice in ["SELIC", "CDI"] else "Correção"
        texto_op = "Descapitalização" if res['is_reverso'] else f"{tipo_op} ({nome_indice})"
        st.sidebar.markdown(f"<small>{texto_op}</small>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<h2 style='color: {cor_tema}; margin:0;'>R$ {res['valor_final']:,.2f}</h2>", unsafe_allow_html=True)
        st.sidebar.markdown(f"Total Período: **{res['percentual']:.2f}%**")

# ==============================================================================
# ÁREA SUPERIOR: EXPANDERS
# ==============================================================================

# 1. BOLETIM FOCUS
with st.expander("🔭 Clique para ver: Expectativas de Mercado (Focus) & Câmbio Hoje", expanded=False):
    col_top1, col_top2 = st.columns([2, 1])
    
    # FOCUS
    df_focus = get_focus_data()
    ano_atual = date.today().year
    
    with col_top1:
        if not df_focus.empty:
            ultima_data = df_focus['data_relatorio'].max()
            df_last = df_focus[df_focus['data_relatorio'] == ultima_data]
            df_view = df_last[df_last['ano_referencia'] == ano_atual]
            df_pivot = df_view.pivot_table(index='Indicador', columns='ano_referencia', values='previsao', aggfunc='mean')
            
            data_str = pd.to_datetime(ultima_data).strftime('%d/%m/%Y')
            st.markdown(f"**Boletim Focus ({data_str}) - Fim de {ano_atual}**")
            
            fc1, fc2, fc3, fc4 = st.columns(4)
            ipca_f = df_pivot.get(ano_atual, {}).get('IPCA', 0)
            selic_f = df_pivot.get(ano_atual, {}).get('Selic', 0)
            pib_f = df_pivot.get(ano_atual, {}).get('PIB Total', 0)
            cambio_f = df_pivot.get(ano_atual, {}).get('Câmbio', 0)
            
            fc1.metric("IPCA", f"{ipca_f:.2f}%")
            fc2.metric("Selic", f"{selic_f:.2f}%")
            fc3.metric("PIB", f"{pib_f:.2f}%")
            fc4.metric("Dólar", f"R$ {cambio_f:.2f}")
        else:
            st.warning("Focus indisponível.")

    # MOEDAS
    df_moedas = get_currency_realtime()
    with col_top2:
        st.markdown("**Câmbio (Agora)**")
        mc1, mc2 = st.columns(2)
        if not df_moedas.empty:
            try:
                usd = df_moedas.loc['USDBRL']
                eur = df_moedas.loc['EURBRL']
                mc1.metric("Dólar", f"R$ {float(usd['bid']):.2f}", f"{float(usd['pctChange']):.2f}%")
                mc2.metric("Euro", f"R$ {float(eur['bid']):.2f}", f"{float(eur['pctChange']):.2f}%")
            except:
                st.info("Erro moedas")
        else:
            st.info("API indisponível")

# 2. HISTÓRICO DE CÂMBIO (NOVO!)
with st.expander("💸 Histórico de Câmbio (Dólar e Euro desde 1994)", expanded=False):
    st.markdown("Evolução das moedas frente ao Real (R$) desde o início do Plano Real.")
    
    # Carrega dados diários
    df_cambio = get_cambio_historico()
    
    if not df_cambio.empty:
        # --- 1. RESUMO DO TOPO (Último Fechamento) ---
        # Pega o último registro válido
        ultimo_dado = df_cambio.iloc[-1]
        penultimo_dado = df_cambio.iloc[-2]
        data_atual = df_cambio.index[-1].strftime('%d/%m/%Y')
        
        st.markdown(f"**Fechamento: {data_atual}**")
        col_res1, col_res2, col_res3 = st.columns([1,1,2])
        
        # Cálculo das variações diárias
        usd_val = ultimo_dado['Dólar']
        usd_var = ((usd_val - penultimo_dado['Dólar']) / penultimo_dado['Dólar']) * 100
        
        eur_val = ultimo_dado['Euro']
        eur_var = ((eur_val - penultimo_dado['Euro']) / penultimo_dado['Euro']) * 100
        
        col_res1.metric("Dólar", f"R$ {usd_val:.2f}", f"{usd_var:.2f}%")
        col_res2.metric("Euro", f"R$ {eur_val:.2f}", f"{eur_var:.2f}%")
        
        st.divider()

        # --- 2. CRIAÇÃO DAS ABAS ---
        tab_graf, tab_matriz, tab_tabela = st.tabs(["📈 Gráfico", "📅 Matriz de Retornos", "📋 Tabela Diária"])
        
        # === ABA 1: GRÁFICO (Com legenda corrigida) ===
        with tab_graf:
            cores_map = {"Dólar": "#00FF7F", "Euro": "#00BFFF"}
            
            fig_cambio = px.line(df_cambio, x=df_cambio.index, y=['Dólar', 'Euro'], 
                                 labels={'value': 'Cotação (R$)', 'variable': 'Moeda', 'data': ''},
                                 color_discrete_map=cores_map)
            
            fig_cambio.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E0E0E0"),
                hovermode="x unified",
                # CORREÇÃO DA LEGENDA: Centralizada e com fundo para leitura
                legend=dict(
                    orientation="h",
                    y=1.1, x=0.5,
                    xanchor="center",
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="#444",
                    borderwidth=1
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            fig_cambio.update_xaxes(
                showgrid=False, rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1A", step="year", stepmode="backward"),
                        dict(count=5, label="5A", step="year", stepmode="backward"),
                        dict(step="all", label="Tudo")
                    ]),
                    bgcolor="#262730", font=dict(color="white")
                )
            )
            fig_cambio.update_yaxes(showgrid=True, gridcolor='#333333', tickprefix="R$ ")
            
            st.plotly_chart(fig_cambio, use_container_width=True)

        # === ABA 2: MATRIZ DE CALOR (Variação Mensal %) ===
        with tab_matriz:
            st.caption("A matriz mostra a variação percentual (%) mês a mês. Verde = Valorização da moeda frente ao Real.")
            
            # Seletor de moeda para a matriz
            moeda_matriz = st.radio("Selecione a Moeda para a Matriz:", ["Dólar", "Euro"], horizontal=True)
            
            # Lógica: Transformar dados diários em Mensais (Último dia do mês)
            df_mensal = df_cambio[[moeda_matriz]].resample('ME').last() # 'ME' é Month End (pandas novo) ou use 'M'
            
            # Calcular retorno mensal
            df_retorno = df_mensal.pct_change() * 100
            df_retorno['ano'] = df_retorno.index.year
            df_retorno['mes'] = df_retorno.index.month_name().str.slice(0, 3) # Jan, Feb...
            
            # Traduzir meses (opcional, pois o pandas vem em inglês por padrão)
            mapa_meses_en_pt = {
                'Jan': 'Jan', 'Feb': 'Fev', 'Mar': 'Mar', 'Apr': 'Abr', 'May': 'Mai', 'Jun': 'Jun',
                'Jul': 'Jul', 'Aug': 'Ago', 'Sep': 'Set', 'Oct': 'Out', 'Nov': 'Nov', 'Dec': 'Dez'
            }
            df_retorno['mes'] = df_retorno['mes'].map(mapa_meses_en_pt)
            
            # Pivotar para formato Matriz (Ano x Mês)
            try:
                matrix_cambio = df_retorno.pivot(index='ano', columns='mes', values=moeda_matriz)
                colunas_ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                matrix_cambio = matrix_cambio[colunas_ordem].sort_index(ascending=False)
                
                # Exibir com cores (Vermelho cai, Verde sobe)
                st.dataframe(
                    matrix_cambio.style.background_gradient(cmap='RdYlGn', vmin=-5, vmax=5).format("{:.2f}%"), 
                    use_container_width=True, 
                    height=500
                )
            except Exception as e:
                st.info(f"Dados insuficientes para gerar a matriz completa: {e}")

        # === ABA 3: TABELA DETALHADA ===
        with tab_tabela:
            # Prepara tabela para download/visualização
            df_view = df_cambio.sort_index(ascending=False).copy()
            df_view.index.name = "Data"
            df_view = df_view.reset_index()
            # Formata data para brasileiro
            df_view['Data'] = df_view['Data'].dt.strftime('%d/%m/%Y')
            
            st.dataframe(
                df_view, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Dólar": st.column_config.NumberColumn(format="R$ %.4f"),
                    "Euro": st.column_config.NumberColumn(format="R$ %.4f")
                }
            )

    else:
        st.warning("Não foi possível carregar o histórico do Yahoo Finance.")

# ==============================================================================
# ÁREA PRINCIPAL: DETALHES DO ÍNDICE
# ==============================================================================

st.title(f"📊 Painel: {tipo_indice.split()[0]}")
st.markdown(f"**Dados históricos atualizados até:** {df.iloc[0]['data_fmt']}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Taxa do Mês", f"{df.iloc[0]['valor']:.2f}%")
kpi2.metric("Acumulado 12 Meses", f"{df.iloc[0]['acum_12m']:.2f}%")
kpi3.metric("Acumulado Ano (YTD)", f"{df.iloc[0]['acum_ano']:.2f}%")
kpi4.metric("Início da Série", df['ano'].min())

tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Matriz de Calor", "📋 Tabela Detalhada"])

with tab1:
    df_chart = df.dropna(subset=['acum_12m']).sort_values('data_date')
    indices_volateis = ["IGP-M", "SELIC", "CDI"]
    eh_volatil = any(idx in tipo_indice for idx in indices_volateis)
    if eh_volatil:
        df_chart = df_chart[df_chart['ano'].astype(int) >= 2000]
    
    fig = px.line(df_chart, x='data_date', y='acum_12m', title=f"Histórico 12 Meses")
    fig.update_traces(line_color=cor_tema, line_width=3)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    try:
        matrix = df.pivot(index='ano', columns='mes_nome', values='valor')
        ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        matrix = matrix[ordem].sort_index(ascending=False)
        st.dataframe(matrix.style.background_gradient(cmap='RdYlGn_r', axis=None, vmin=-0.1, vmax=1.5).format("{:.2f}"), use_container_width=True, height=500)
    except:
        st.warning("Matriz indisponível.")

with tab3:
    st.dataframe(df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']], use_container_width=True, hide_index=True)