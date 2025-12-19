import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import requests
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
import urllib3
import time
from typing import Dict, Tuple, Optional
import warnings

# =============================================================================
# CONFIGURAÇÃO INICIAL
# =============================================================================

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS E ESTILIZAÇÃO
# =============================================================================

st.markdown("""
<style>
    /* Cards de Métricas */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f2f6 100%);
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #31333F;
    }
    
    /* Headers dos Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        font-weight: 600;
        color: white !important;
    }
    
    /* Métricas Nativas do Streamlit */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eee;
    }

    /* Box de Sucesso Sidebar */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 12px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cores para Matriz de Calor (Suave)
cores_matriz = ["#FFB3B3", "#FFFFFF", "#B3FFB3"]
cmap_custom = LinearSegmentedColormap.from_list("custom", cores_matriz)

# Configurações Globais
class Config:
    BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados"
    FOCUS_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
    HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    
    INDICES = {
        "IPCA (Inflação Oficial)": {"source": "sidra", "table": "1737", "variable": "63", "code": "IPCA", "color": "#00D9FF"},
        "INPC (Salários)": {"source": "sidra", "table": "1736", "variable": "44", "code": "INPC", "color": "#00FFA3"},
        "IGP-M (Aluguéis)": {"source": "bcb", "bcb_code": "189", "code": "IGP-M", "color": "#FF6B6B"},
        "SELIC (Taxa Básica)": {"source": "bcb", "bcb_code": "4390", "code": "SELIC", "color": "#FFD93D"},
        "CDI (Investimentos)": {"source": "bcb", "bcb_code": "4391", "code": "CDI", "color": "#A8E6CF"}
    }
    
    SERIES_MACRO = {
        'PIB (R$ Bi)': 4382, 'Dívida Líq. (% PIB)': 4513,
        'Res. Primário (% PIB)': 5793, 'Res. Nominal (% PIB)': 5811,
        'Balança Com. (US$ Mi)': 22707, 'Trans. Correntes (US$ Mi)': 22724,
        'IDP (US$ Mi)': 22885
    }

# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def hex_to_rgba(hex_color, opacity=0.2):
    """Converte HEX para RGBA para evitar erros no Plotly"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {opacity})"
    return hex_color

def retry_request(func, max_attempts=3, delay=1.0):
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Não vamos levantar erro para não quebrar a tela inteira, retornamos vazio
                    return None 
                time.sleep(delay)
        return None
    return wrapper

# =============================================================================
# FUNÇÕES DE CARGA DE DADOS
# =============================================================================

@st.cache_data(ttl=3600)
def get_sidra_data(table_code: str, variable_code: str) -> pd.DataFrame:
    try:
        dados_raw = sidrapy.get_table(
            table_code=table_code, territorial_level="1",
            ibge_territorial_code="all", variable=variable_code,
            period="last 360"
        )
        if dados_raw.empty: return pd.DataFrame()
        
        df = dados_raw.iloc[1:].copy()
        df = df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'})
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['D2C'].str.slice(0, 4)
        df['D2C'] = df['D2C'].astype(str)
        
        return df.dropna(subset=['valor', 'data_date'])
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_bcb_data(codigo_serie: str) -> pd.DataFrame:
    @retry_request
    def fetch():
        url = Config.BCB_BASE.format(codigo_serie) + "?formato=json"
        resp = requests.get(url, headers=Config.HEADERS, verify=False, timeout=10)
        resp.raise_for_status()
        return resp.json()
    
    data = fetch()
    if not data: return pd.DataFrame()
    
    try:
        df = pd.DataFrame(data)
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        return df.dropna(subset=['valor', 'data_date'])
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_focus_data() -> pd.DataFrame:
    try:
        url = f"{Config.FOCUS_URL}?$top=5000&$orderby=Data%20desc&$format=json"
        resp = requests.get(url, headers=Config.HEADERS, verify=False, timeout=15)
        resp.raise_for_status()
        
        df = pd.DataFrame(resp.json()['value'])
        indicadores = ['IPCA', 'PIB Total', 'Selic', 'Câmbio', 'IGP-M',
                       'IPCA Administrados', 'Conta corrente', 'Balança comercial',
                       'Investimento direto no país', 'Dívida líquida do setor público',
                       'Resultado primário', 'Resultado nominal']
        
        df = df[df['Indicador'].isin(indicadores)]
        df = df.rename(columns={'Data': 'data_relatorio', 'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'})
        df['ano_referencia'] = pd.to_numeric(df['ano_referencia'], errors='coerce')
        df['previsao'] = pd.to_numeric(df['previsao'], errors='coerce')
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        
        df = df.sort_values('data_relatorio', ascending=False)
        df = df.drop_duplicates(subset=['Indicador', 'ano_referencia'], keep='first')
        
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_currency_realtime() -> pd.DataFrame:
    try:
        tickers = {"USDBRL=X": "USDBRL", "EURBRL=X": "EURBRL"}
        dados = {}
        for ticker, nome in tickers.items():
            try:
                info = yf.Ticker(ticker).fast_info
                dados[nome] = {'bid': info['last_price'], 
                             'pctChange': ((info['last_price'] - info['previous_close']) / info['previous_close']) * 100}
            except:
                dados[nome] = {'bid': 0.0, 'pctChange': 0.0}
        return pd.DataFrame.from_dict(dados, orient='index')
    except: return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_cambio_historico() -> pd.DataFrame:
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="2000-01-01", progress=False)['Close']
        if df.empty: return pd.DataFrame()
        
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo').tz_localize(None)
        
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        return df.ffill()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_macro_real() -> Dict:
    resultados = {}
    
    for nome, codigo in Config.SERIES_MACRO.items():
        @retry_request
        def fetch():
            url = f"{Config.BCB_BASE.format(codigo)}/ultimos/24?formato=json"
            resp = requests.get(url, headers=Config.HEADERS, verify=False, timeout=10)
            return resp.json()
        
        dados = fetch()
        if not dados: continue
        
        try:
            df = pd.DataFrame(dados)
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            
            # Lógica específica para cada indicador
            if 'PIB' in nome and 'Dívida' not in nome and 'Res.' not in nome:
                val = df.iloc[-1]['valor'] / 1_000_000
            elif any(x in nome for x in ['Balança', 'Trans.', 'IDP']):
                val = df.iloc[-12:]['valor'].sum() / 1_000 # Soma 12m
            elif 'Primário' in nome or 'Nominal' in nome:
                val = df.iloc[-1]['valor'] * -1 # Inverte sinal
            else:
                val = df.iloc[-1]['valor']
            
            resultados[nome] = val
        except: continue
    
    return resultados

def processar_dataframe_comum(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    
    df = df.sort_values('data_date', ascending=True)
    df['mes_num'] = df['data_date'].dt.month
    meses = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
             7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12, min_periods=12).apply(np.prod, raw=True) - 1) * 100
    
    return df.sort_values('data_date', ascending=False)

def calcular_correcao(df: pd.DataFrame, valor: float, data_ini: str, data_fim: str) -> Tuple:
    is_reverso = int(data_ini) > int(data_fim)
    periodo_inicio = min(data_ini, data_fim)
    periodo_fim = max(data_ini, data_fim)
    
    mask = (df['D2C'] >= periodo_inicio) & (df['D2C'] <= periodo_fim)
    df_periodo = df.loc[mask]
    
    if df_periodo.empty: return None, "⚠️ Período sem dados"
    
    fator = df_periodo['fator'].prod()
    valor_final = valor / fator if is_reverso else valor * fator
    percentual = (fator - 1) * 100
    
    return {'valor_final': valor_final, 'percentual': percentual, 
            'fator': fator, 'is_reverso': is_reverso, 'meses': len(df_periodo)}, None

# =============================================================================
# SIDEBAR
# =============================================================================

try:
    st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
except:
    st.sidebar.markdown("## 📊 VPL CONSULTORIA")

st.sidebar.header("⚙️ Configurações")

tipo_indice = st.sidebar.selectbox("Selecione o Indicador", list(Config.INDICES.keys()))
config_indice = Config.INDICES[tipo_indice]

with st.spinner(f"Carregando {config_indice['code']}..."):
    if config_indice['source'] == 'sidra':
        df_raw = get_sidra_data(config_indice['table'], config_indice['variable'])
    else:
        df_raw = get_bcb_data(config_indice['bcb_code'])
    
    df = processar_dataframe_comum(df_raw)
    cor_tema = config_indice['color']

if not df.empty:
    st.sidebar.markdown(f"<div class='success-box'>✅ Atualizado: {df.iloc[0]['data_fmt']}</div>", unsafe_allow_html=True)
else:
    st.sidebar.error("Erro ao carregar dados")
    st.stop()

# Calculadora
st.sidebar.divider()
st.sidebar.subheader("🧮 Calculadora")

valor_input = st.sidebar.number_input("Valor (R$)", min_value=0.01, value=1000.00, step=100.00)

lista_anos = sorted(df['ano'].unique(), reverse=True)
meses_nome = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
mapa_meses = {m: f"{i:02d}" for i, m in enumerate(meses_nome, 1)}

st.sidebar.markdown("**📅 Data Inicial**")
c1, c2 = st.sidebar.columns(2)
mes_ini = c1.selectbox("Mês", meses_nome, index=0, key="mi", label_visibility="collapsed")
ano_ini = c2.selectbox("Ano", lista_anos, index=min(1, len(lista_anos)-1), key="ai", label_visibility="collapsed")

st.sidebar.markdown("**🎯 Data Final**")
c3, c4 = st.sidebar.columns(2)
mes_fim = c3.selectbox("Mês", meses_nome, index=11, key="mf", label_visibility="collapsed")
ano_fim = c4.selectbox("Ano", lista_anos, index=0, key="af", label_visibility="collapsed")

if st.sidebar.button("🚀 Calcular", type="primary", use_container_width=True):
    code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
    code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
    
    resultado, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
    
    if erro:
        st.sidebar.error(erro)
    else:
        st.sidebar.divider()
        label = "Descapitalização" if resultado['is_reverso'] else "Valor Corrigido"
        st.sidebar.markdown(f"**{label} ({config_indice['code']})**")
        st.sidebar.markdown(f"<h1 style='color: {cor_tema}; margin:0;'>R$ {resultado['valor_final']:,.2f}</h1>", unsafe_allow_html=True)
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Total", f"{resultado['percentual']:.2f}%")
        col2.metric("Fator", f"{resultado['fator']:.4f}")

# =============================================================================
# PAINEL PRINCIPAL
# =============================================================================

# Focus + Câmbio
with st.expander("🔭 Expectativas (Focus) & Câmbio", expanded=False):
    col1, col2 = st.columns([2, 1])
    
    df_focus = get_focus_data()
    ano_atual = date.today().year
    
    with col1:
        if not df_focus.empty:
            st.markdown(f"#### Boletim Focus")
            df_atual = df_focus[df_focus['ano_referencia'] == ano_atual]
            if not df_atual.empty:
                pivot = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='first')
                
                fc1, fc2, fc3, fc4 = st.columns(4)
                def get_val(idx): return pivot.loc[idx, 'previsao'] if idx in pivot.index else 0
                
                fc1.metric("IPCA", f"{get_val('IPCA'):.2f}%")
                fc2.metric("Selic", f"{get_val('Selic'):.2f}%")
                fc3.metric("PIB", f"{get_val('PIB Total'):.2f}%")
                fc4.metric("Dólar", f"R$ {get_val('Câmbio'):.2f}")
                
                st.divider()
                st.markdown("###### Projeções (Próximos 3 anos)")
                anos_proj = [ano_atual + i for i in range(3)]
                df_table = df_focus[df_focus['ano_referencia'].isin(anos_proj)]
                if not df_table.empty:
                    pivot_multi = df_table.pivot_table(index='Indicador', columns='ano_referencia', values='previsao')
                    st.dataframe(pivot_multi, use_container_width=True)
        else:
            st.warning("Focus indisponível no momento.")
    
    with col2:
        st.markdown("#### Câmbio Agora")
        df_moedas = get_currency_realtime()
        if not df_moedas.empty:
            mc1, mc2 = st.columns(2)
            usd = df_moedas.loc['USDBRL']
            eur = df_moedas.loc['EURBRL']
            
            def render_cambio(col, label, val):
                col.metric(label, f"R$ {val['bid']:.2f}", f"{val['pctChange']:.2f}%")
                
            render_cambio(mc1, "Dólar", usd)
            render_cambio(mc2, "Euro", eur)

# Macro
with st.expander("🧩 Conjuntura Macroeconômica", expanded=False):
    macro_dados = get_macro_real()
    
    if macro_dados:
        st.markdown("##### Atividade & Fiscal")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PIB", f"R$ {macro_dados.get('PIB (R$ Bi)', 0):.2f} Tri")
        c2.metric("Dív. Líquida", f"{macro_dados.get('Dívida Líq. (% PIB)', 0):.1f}% PIB")
        c3.metric("Res. Primário", f"{macro_dados.get('Res. Primário (% PIB)', 0):.2f}% PIB")
        c4.metric("Res. Nominal", f"{macro_dados.get('Res. Nominal (% PIB)', 0):.2f}% PIB")
        
        st.divider()
        st.markdown("##### Setor Externo")
        c5, c6, c7 = st.columns(3)
        c5.metric("Balança", f"US$ {macro_dados.get('Balança Com. (US$ Mi)', 0):.1f} Bi")
        c6.metric("Trans. Correntes", f"US$ {macro_dados.get('Trans. Correntes (US$ Mi)', 0):.1f} Bi")
        c7.metric("IDP", f"US$ {macro_dados.get('IDP (US$ Mi)', 0):.1f} Bi")
    else:
        st.warning("Dados macroeconômicos não carregados (Instabilidade BCB).")

# Câmbio Histórico
with st.expander("💸 Histórico de Câmbio (1994-2025)", expanded=False):
    df_cambio = get_cambio_historico()
    
    if not df_cambio.empty:
        tab1, tab2, tab3 = st.tabs(["Gráfico", "Matriz", "Tabela"])
        with tab1:
            fig = px.line(df_cambio, x=df_cambio.index, y=['Dólar', 'Euro'],
                          color_discrete_map={"Dólar": "#00FF7F", "Euro": "#00BFFF"})
            fig.update_layout(template="plotly_white", hovermode="x unified", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            moeda = st.radio("Moeda:", ["Dólar", "Euro"], horizontal=True)
            df_mensal = df_cambio[[moeda]].resample('ME').last()
            df_ret = df_mensal.pct_change() * 100
            df_ret['ano'] = df_ret.index.year
            df_ret['mes_num'] = df_ret.index.month
            
            meses_map = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                         7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
            df_ret['mes_nome'] = df_ret['mes_num'].map(meses_map)
            
            matriz = df_ret.pivot(index='ano', columns='mes_nome', values=moeda)
            ordem = list(meses_map.values())
            matriz = matriz[[c for c in ordem if c in matriz.columns]].sort_index(ascending=False)
            
            st.dataframe(matriz.style.background_gradient(cmap=cmap_custom, vmin=-5, vmax=5).format("{:.2f}%"), use_container_width=True, height=450)
        
        with tab3:
            df_view = df_cambio.reset_index().sort_values('Date', ascending=False)
            df_view['Date'] = pd.to_datetime(df_view['Date']).dt.strftime('%d/%m/%Y')
            st.dataframe(df_view, use_container_width=True)

# Painel Principal do Índice
st.title(f"📊 {config_indice['code']} - Análise Completa")
st.caption(f"Atualizado: {df.iloc[0]['data_fmt']}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Taxa Mês", f"{df.iloc[0]['valor']:.2f}%")
kpi2.metric("Acum. 12M", f"{df.iloc[0]['acum_12m']:.2f}%")
kpi3.metric("Acum. Ano", f"{df.iloc[0]['acum_ano']:.2f}%")
kpi4.metric("Desde", df['ano'].min())

tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "🗓️ Matriz", "📋 Dados"])

with tab1:
    # --- MELHORIA: SLIDER DE ANO ---
    st.markdown("##### Filtrar Histórico")
    anos_disponiveis = sorted(df['ano'].astype(int).unique())
    if anos_disponiveis:
        min_ano, max_ano = min(anos_disponiveis), max(anos_disponiveis)
        # Define 2018 como padrão se possível, senão usa o mínimo
        padrao = 2018 if 2018 >= min_ano else min_ano
        
        ano_selecionado = st.slider(
            "Selecione o ano inicial:",
            min_value=min_ano,
            max_value=max_ano,
            value=padrao
        )
        
        # Filtra o dataframe
        df_chart = df[df['ano'].astype(int) >= ano_selecionado].sort_values('data_date')
    else:
        df_chart = df.sort_values('data_date')

    # Gráfico de Área com correção de cor
    fig = px.area(df_chart, x='data_date', y='acum_12m', title=f"Acumulado 12 Meses - {config_indice['code']}")
    
    # Converte cor hexadecimal para RGBA para evitar erros no Plotly (Imagem 1)
    fill_color = hex_to_rgba(cor_tema, 0.2)
    
    fig.update_traces(line_color=cor_tema, fillcolor=fill_color)
    fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    try:
        matriz = df.pivot(index='ano', columns='mes_nome', values='valor')
        ordem = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
        matriz = matriz[[c for c in ordem if c in matriz.columns]].sort_index(ascending=False)
        st.dataframe(matriz.style.background_gradient(cmap=cmap_custom, vmin=-1.5, vmax=1.5).format("{:.2f}%"), use_container_width=True, height=500)
    except: st.warning("Matriz indisponível")

with tab3:
    df_export = df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].copy()
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, f"{config_indice['code']}.csv", "text/csv")
    st.dataframe(df_export, use_container_width=True)

# Rodapé
st.divider()
st.markdown("<div style='text-align: center; color: #666;'>VPL Consultoria • Dados: IBGE, BCB, Yahoo Finance</div>", unsafe_allow_html=True)
