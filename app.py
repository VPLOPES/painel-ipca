# CONSOLIDADO FINAL – PAINEL MACROECONÔMICO
# Base: codigo_funcionando_ipca.py
# Melhorias incorporadas da versao_1_ipca.py
# Objetivo: arquivo único, estável, com todas as APIs funcionando

import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
import requests
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
import urllib3

# ================================
# CONFIGURAÇÕES GERAIS
# ================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(
    page_title="VPL Consultoria – Inteligência Financeira",
    page_icon="📊",
    layout="wide"
)

# ================================
# ESTILO
# ================================
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

cores_matriz = ["#FFB3B3", "#FFFFFF", "#B3FFB3"]
cmap_matriz = LinearSegmentedColormap.from_list("pastel", cores_matriz)

CORES_INDICES = {
    'IPCA': '#00BFFF',
    'INPC': '#00FF7F',
    'IGP-M': '#FF6347',
    'SELIC': '#FFD700',
    'CDI': '#FFFFFF'
}

# ================================
# FUNÇÕES DE DADOS
# ================================

@st.cache_data(ttl=3600)
def get_sidra_data(table_code, variable_code):
    try:
        raw = sidrapy.get_table(
            table_code=table_code,
            territorial_level="1",
            ibge_territorial_code="all",
            variable=variable_code,
            period="last 360"
        )
        df = raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format='%Y%m')
        df['ano'] = df['data_date'].dt.year.astype(str)
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        return processar_dataframe(df)
    except:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_bcb_data(codigo):
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json"
        r = requests.get(url, verify=False, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['ano'] = df['data_date'].dt.year.astype(str)
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        return processar_dataframe(df)
    except:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_focus_data():
    try:
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=5000&$orderby=Data%20desc&$format=json"
        df = pd.DataFrame(requests.get(url, verify=False).json()['value'])
        df = df.rename(columns={'Data': 'data_relatorio', 'DataReferencia': 'ano', 'Mediana': 'valor'})
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df = df.sort_values('data_relatorio', ascending=False)
        df = df.drop_duplicates(subset=['Indicador', 'ano'])
        return df
    except:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_cambio_realtime():
    try:
        dados = {}
        for t in ['USDBRL=X', 'EURBRL=X']:
            info = yf.Ticker(t).fast_info
            dados[t] = {
                'preco': info['last_price'],
                'var': ((info['last_price'] - info['previous_close']) / info['previous_close']) * 100
            }
        df = pd.DataFrame(dados).T
        df.index = ['Dólar', 'Euro']
        return df
    except:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_cambio_historico():
    try:
        df = yf.download(['USDBRL=X', 'EURBRL=X'], start='1994-07-01', progress=False)['Close']
        df.columns = ['Dólar', 'Euro']
        return df.ffill()
    except:
        return pd.DataFrame()


# ================================
# PROCESSAMENTO
# ================================

def processar_dataframe(df):
    df = df.sort_values('data_date')
    df['mes'] = df['data_date'].dt.month
    meses = {1:'Jan',2:'Fev',3:'Mar',4:'Abr',5:'Mai',6:'Jun',7:'Jul',8:'Ago',9:'Set',10:'Out',11:'Nov',12:'Dez'}
    df['mes_nome'] = df['mes'].map(meses)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(12).apply(np.prod) - 1) * 100
    return df.sort_values('data_date', ascending=False)


def calcular_correcao(df, valor, ini, fim):
    reverso = ini > fim
    p_ini, p_fim = sorted([ini, fim])
    dfp = df[(df['D2C'] >= p_ini) & (df['D2C'] <= p_fim)]
    if dfp.empty:
        return None
    fator = dfp['fator'].prod()
    return valor / fator if reverso else valor * fator, fator

# ================================
# SIDEBAR
# ================================

st.sidebar.header("Configurações")

INDICES = {
    'IPCA (Inflação)': ('sidra', '1737', '63', 'IPCA'),
    'INPC (Salários)': ('sidra', '1736', '44', 'INPC'),
    'IGP-M (Aluguéis)': ('bcb', None, '189', 'IGP-M'),
    'SELIC': ('bcb', None, '4390', 'SELIC'),
    'CDI': ('bcb', None, '4391', 'CDI')
}

escolha = st.sidebar.selectbox("Indicador", list(INDICES.keys()))
origem, t, v, nome = INDICES[escolha]

with st.spinner("Carregando dados..."):
    df = get_sidra_data(t, v) if origem == 'sidra' else get_bcb_data(v)

if df.empty:
    st.error("Erro ao carregar dados")
    st.stop()

# ================================
# PAINEL PRINCIPAL
# ================================

st.title(f"📊 {nome}")
st.markdown(f"Última observação: **{df.iloc[0]['data_fmt']}**")

c1, c2, c3 = st.columns(3)
c1.metric("Mensal", f"{df.iloc[0]['valor']:.2f}%")
c2.metric("12 Meses", f"{df.iloc[0]['acum_12m']:.2f}%")
c3.metric("Ano", f"{df.iloc[0]['acum_ano']:.2f}%")

fig = px.line(df.sort_values('data_date'), x='data_date', y='acum_12m')
fig.update_layout(template='plotly_dark')
st.plotly_chart(fig, use_container_width=True)

# ================================
# FOCUS E CÂMBIO
# ================================

with st.expander("Focus e Câmbio"):
    df_focus = get_focus_data()
    if not df_focus.empty:
        st.dataframe(df_focus.head(20), use_container_width=True)

    df_fx = get_cambio_realtime()
    if not df_fx.empty:
        for m in df_fx.index:
            st.metric(m, f"R$ {df_fx.loc[m,'preco']:.2f}", f"{df_fx.loc[m,'var']:.2f}%")

# ================================
# HISTÓRICO DE CÂMBIO
# ================================

with st.expander("Histórico do Câmbio"):
    df_hist = get_cambio_historico()
    if not df_hist.empty:
        st.line_chart(df_hist)
