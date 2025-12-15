import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
import requests

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
        url = "https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL"
        response = requests.get(url, timeout=5)
        data = response.json()
        df = pd.DataFrame.from_dict(data, orient='index')
        return df
    except:
        return pd.DataFrame()

# 5. NOVO: Histórico de Câmbio (BCB - Séries Diárias)
@st.cache_data(ttl=86400) # Cache de 24h
def get_cambio_historico():
    try:
        # O segredo: Fingir ser um navegador (Chrome) para o BCB não bloquear
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        # URLs (usando HTTPS)
        url_usd = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=json"
        url_eur = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.21619/dados?formato=json"
        
        # Baixando com requests e headers
        resp_usd = requests.get(url_usd, headers=headers, timeout=10)
        df_usd = pd.DataFrame(resp_usd.json())
        
        resp_eur = requests.get(url_eur, headers=headers, timeout=10)
        df_eur = pd.DataFrame(resp_eur.json())
        
        # Tratamento USD
        df_usd['data'] = pd.to_datetime(df_usd['data'], format='%d/%m/%Y')
        df_usd = df_usd.rename(columns={'valor': 'Dólar'})
        df_usd = df_usd.set_index('data')
        
        # Tratamento EUR
        df_eur['data'] = pd.to_datetime(df_eur['data'], format='%d/%m/%Y')
        df_eur = df_eur.rename(columns={'valor': 'Euro'})
        df_eur = df_eur.set_index('data')
        
        # Juntar tudo e filtrar a partir do Plano Real (01/07/1994)
        df_final = df_usd.join(df_eur, how='outer')
        df_final = df_final[df_final.index >= '1994-07-01']
        
        # Preencher buracos (fins de semana) com o valor anterior para o gráfico não quebrar
        df_final = df_final.ffill()
        
        return df_final
    except Exception as e:
        print(f"Erro ao baixar histórico de câmbio: {e}")
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
    
    # Carrega dados
    df_cambio = get_cambio_historico()
    
    if not df_cambio.empty:
        # Gráfico interativo
        fig_cambio = px.line(df_cambio, x=df_cambio.index, y=['Dólar', 'Euro'], 
                             labels={'value': 'Preço (R$)', 'variable': 'Moeda', 'data': 'Data'})
        
        # Personalização do gráfico
        fig_cambio.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
        fig_cambio.update_xaxes(
            rangeslider_visible=True, # Slider de zoom na parte inferior
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 Ano", step="year", stepmode="backward"),
                    dict(count=5, label="5 Anos", step="year", stepmode="backward"),
                    dict(count=10, label="10 Anos", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ])
            )
        )
        st.plotly_chart(fig_cambio, use_container_width=True)
    else:
        st.warning("Não foi possível carregar o histórico do Banco Central.")

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