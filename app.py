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

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        font-weight: 600;
        color: white !important;
        padding: 15px;
    }
    .stMetric {
        background-color: rgba(102, 126, 234, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 12px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Configurações
class Config:
    CORES = {
        'IPCA': '#00D9FF', 'INPC': '#00FFA3', 'IGP-M': '#FF6B6B',
        'SELIC': '#FFD93D', 'CDI': '#A8E6CF'
    }
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

cores_matriz = ["#FF6B6B", "#FFFFFF", "#4ECDC4"]
cmap_custom = LinearSegmentedColormap.from_list("custom", cores_matriz)

# =============================================================================
# FUNÇÕES
# =============================================================================

def retry_request(func, max_attempts=3, delay=1.0):
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(delay)
    return wrapper

@st.cache_data(ttl=3600)
def get_sidra_data(table_code: str, variable_code: str) -> pd.DataFrame:
    try:
        dados_raw = sidrapy.get_table(
            table_code=table_code, territorial_level="1",
            ibge_territorial_code="all", variable=variable_code,
            period="last 360"
        )
        if dados_raw.empty:
            return pd.DataFrame()
        
        df = dados_raw.iloc[1:].copy()
        df = df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'})
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['D2C'].str.slice(0, 4)
        df['D2C'] = df['D2C'].astype(str)
        
        return df.dropna(subset=['valor', 'data_date'])
    except Exception as e:
        st.error(f"Erro Sidra: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_bcb_data(codigo_serie: str) -> pd.DataFrame:
    @retry_request
    def fetch():
        url = Config.BCB_BASE.format(codigo_serie) + "?formato=json"
        resp = requests.get(url, headers=Config.HEADERS, verify=False, timeout=15)
        resp.raise_for_status()
        return resp.json()
    
    try:
        data = fetch()
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        
        return df.dropna(subset=['valor', 'data_date'])
    except Exception as e:
        st.warning(f"Erro BCB: {e}")
        return pd.DataFrame()

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
    except Exception as e:
        st.warning(f"Focus indisponível: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_currency_realtime() -> pd.DataFrame:
    try:
        tickers = {"USDBRL=X": "USDBRL", "EURBRL=X": "EURBRL"}
        dados = {}
        
        for ticker, nome in tickers.items():
            try:
                info = yf.Ticker(ticker).fast_info
                preco = info['last_price']
                anterior = info['previous_close']
                variacao = ((preco - anterior) / anterior) * 100
                dados[nome] = {'bid': preco, 'pctChange': variacao}
            except:
                dados[nome] = {'bid': 5.50 if 'USD' in ticker else 6.00, 'pctChange': 0.0}
        
        return pd.DataFrame.from_dict(dados, orient='index')
    except:
        return pd.DataFrame.from_dict({
            'USDBRL': {'bid': 5.50, 'pctChange': 0.0},
            'EURBRL': {'bid': 6.00, 'pctChange': 0.0}
        }, orient='index')

@st.cache_data(ttl=86400)
def get_cambio_historico() -> pd.DataFrame:
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="1994-07-01", progress=False)['Close']
        if df.empty:
            return pd.DataFrame()
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo').tz_localize(None)
        
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        return df.ffill()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_macro_real() -> Dict:
    resultados = {}
    historico = {}
    
    for nome, codigo in Config.SERIES_MACRO.items():
        @retry_request
        def fetch():
            url = f"{Config.BCB_BASE.format(codigo)}/ultimos/60?formato=json"
            resp = requests.get(url, headers=Config.HEADERS, verify=False, timeout=10)
            resp.raise_for_status()
            return resp.json()
        
        try:
            dados = fetch()
            if not dados:
                continue
            
            df = pd.DataFrame(dados)
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            df['data_dt'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
            
            df_chart = df.copy()
            if 'PIB' in nome:
                df_chart['valor'] /= 1_000_000
            elif nome in ['Balança Com. (US$ Mi)', 'Trans. Correntes (US$ Mi)', 'IDP (US$ Mi)']:
                df_chart['valor'] /= 1_000
            elif 'Primário' in nome or 'Nominal' in nome:
                df_chart['valor'] *= -1
            
            historico[nome] = df_chart
            
            if 'PIB' in nome:
                val_kpi = df.iloc[-1]['valor'] / 1_000_000
            elif nome in ['Balança Com. (US$ Mi)', 'Trans. Correntes (US$ Mi)', 'IDP (US$ Mi)']:
                val_kpi = df.iloc[-12:]['valor'].sum() / 1_000
            elif 'Primário' in nome or 'Nominal' in nome:
                val_kpi = df.iloc[-1]['valor'] * -1
            else:
                val_kpi = df.iloc[-1]['valor']
            
            resultados[nome] = val_kpi
        except:
            continue
    
    return {'dados': resultados, 'historico': historico}

def processar_dataframe_comum(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
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
    df_periodo = df.loc[mask].sort_values('data_date', ascending=True)
    
    if df_periodo.empty:
        return None, "⚠️ Período sem dados"
    
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
    st.sidebar.markdown(
        f"<div class='success-box'>✅ Atualizado: {df.iloc[0]['data_fmt']}</div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.error("Erro ao carregar dados")
    st.stop()

# Calculadora
st.sidebar.divider()
st.sidebar.subheader("🧮 Calculadora")

valor_input = st.sidebar.number_input("Valor (R$)", min_value=0.01, value=1000.00, step=100.00, format="%.2f")

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
        nome_idx = config_indice['code']
        tipo_op = "Rendimento" if nome_idx in ["SELIC", "CDI"] else "Correção"
        label = "Descapitalização" if resultado['is_reverso'] else f"{tipo_op}"
        
        st.sidebar.markdown(f"**{label} ({nome_idx})**")
        st.sidebar.markdown(
            f"<h1 style='color: {cor_tema}; margin:0;'>R$ {resultado['valor_final']:,.2f}</h1>",
            unsafe_allow_html=True
        )
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Total", f"{resultado['percentual']:.2f}%")
        col2.metric("Fator", f"{resultado['fator']:.6f}")
        st.sidebar.caption(f"{resultado['meses']} meses")

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
                fc1.metric("IPCA", f"{pivot.loc['IPCA', 'previsao'] if 'IPCA' in pivot.index else 0:.2f}%")
                fc2.metric("Selic", f"{pivot.loc['Selic', 'previsao'] if 'Selic' in pivot.index else 0:.2f}%")
                fc3.metric("PIB", f"{pivot.loc['PIB Total', 'previsao'] if 'PIB Total' in pivot.index else 0:.2f}%")
                fc4.metric("Dólar", f"R$ {pivot.loc['Câmbio', 'previsao'] if 'Câmbio' in pivot.index else 0:.2f}")
                
                st.divider()
                
                anos_proj = [ano_atual + i for i in range(3)]
                df_table = df_focus[df_focus['ano_referencia'].isin(anos_proj)]
                
                if not df_table.empty:
                    pivot_multi = df_table.pivot_table(index='Indicador', columns='ano_referencia', values='previsao')
                    st.dataframe(pivot_multi, use_container_width=True)
        else:
            st.warning("Focus indisponível")
    
    with col2:
        st.markdown("#### Câmbio Agora")
        df_moedas = get_currency_realtime()
        
        if not df_moedas.empty:
            mc1, mc2 = st.columns(2)
            usd = df_moedas.loc['USDBRL']
            eur = df_moedas.loc['EURBRL']
            mc1.metric("Dólar", f"R$ {float(usd['bid']):.2f}", f"{float(usd['pctChange']):.2f}%")
            mc2.metric("Euro", f"R$ {float(eur['bid']):.2f}", f"{float(eur['pctChange']):.2f}%")

# Macro
with st.expander("🧩 Conjuntura Macroeconômica", expanded=False):
    macro_result = get_macro_real()
    macro_dados = macro_result.get('dados', {})
    
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

# Câmbio Histórico
with st.expander("💸 Histórico de Câmbio (1994-2025)", expanded=False):
    df_cambio = get_cambio_historico()
    
    if not df_cambio.empty:
        tab1, tab2, tab3 = st.tabs(["Gráfico", "Matriz", "Tabela"])
        
        with tab1:
            fig = px.line(df_cambio, x=df_cambio.index, y=['Dólar', 'Euro'],
                         color_discrete_map={"Dólar": "#00FF7F", "Euro": "#00BFFF"})
            fig.update_layout(template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            moeda = st.radio("Moeda:", ["Dólar", "Euro"], horizontal=True)
            df_mensal = df_cambio[[moeda]].resample('ME').last()
            df_ret = df_mensal.pct_change() * 100
            df_ret['ano'] = df_ret.index.year
            df_ret['mes'] = df_ret.index.month
            
            meses_map = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                        7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
            df_ret['mes_nome'] = df_ret['mes'].map(meses_map)
            
            matriz = df_ret.pivot(index='ano', columns='mes_nome', values=moeda)
            ordem = list(meses_map.values())
            matriz = matriz[[c for c in ordem if c in matriz.columns]].sort_index(ascending=False)
            
            st.dataframe(
                matriz.style.background_gradient(cmap=cmap_custom, vmin=-8, vmax=8).format("{:.2f}%"),
                use_container_width=True, height=450
            )
        
        with tab3:
            df_view = df_cambio.reset_index()
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
    df_chart = df.dropna(subset=['acum_12m']).sort_values('data_date')
    fig = px.line(df_chart, x='data_date', y='acum_12m')
    fig.update_traces(line_color=cor_tema, line_width=3)
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    try:
        matriz = df.pivot(index='ano', columns='mes_nome', values='valor')
        ordem = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
        matriz = matriz[[c for c in ordem if c in matriz.columns]].sort_index(ascending=False)
        
        st.dataframe(
            matriz.style.background_gradient(cmap=cmap_custom, vmin=-2, vmax=2).format("{:.2f}%"),
            use_container_width=True, height=500
        )
    except:
        st.warning("Matriz indisponível")

with tab3:
    df_export = df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].copy()
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, f"{config_indice['code']}.csv", "text/csv")
    st.dataframe(df_export, use_container_width=True)

# Rodapé
st.divider()
col1, col2, col3 = st.columns(3)
col1.markdown("**VPL Consultoria**\nInteligência Financeira")
col2.markdown("**Fontes**\nIBGE, BCB, Yahoo Finance")
col3.markdown(f"**Atualizado**\n{datetime.now().strftime('%d/%m/%Y %H:%M')}")
