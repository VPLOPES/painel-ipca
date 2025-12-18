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
import time
from typing import Dict, Tuple, Optional, List
import urllib3

# Desabilita avisos de SSL (apenas para APIs do governo)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Otimizado
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 8px;
        font-weight: 600;
        color: white;
        border: none;
        padding: 12px;
    }
    .stMetric {
        background-color: rgba(30, 60, 114, 0.3);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    div[data-testid="stExpander"] {
        background-color: rgba(0,0,0,0.2);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Paleta de cores melhorada
CORES = {
    'IPCA': '#00D9FF',
    'INPC': '#00FFA3',
    'IGP-M': '#FF6B6B',
    'SELIC': '#FFD93D',
    'CDI': '#A8E6CF'
}

cores_matriz = ["#FF6B6B", "#FFFFFF", "#4ECDC4"]
cmap_custom = LinearSegmentedColormap.from_list("custom_rdylgn", cores_matriz)

# --- CLASSES DE CONFIGURAÇÃO ---
class APIConfig:
    """Configurações centralizadas de APIs"""
    BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados"
    FOCUS_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    SERIES_MACRO = {
        'PIB': 4382,
        'Dívida Líq.': 4513,
        'Res. Primário': 5793,
        'Res. Nominal': 5811,
        'Balança Com.': 22707,
        'Trans. Correntes': 22724,
        'IDP': 22885
    }
    
    INDICADORES_FOCUS = [
        'IPCA', 'PIB Total', 'Selic', 'Câmbio', 'IGP-M',
        'IPCA Administrados', 'Conta corrente', 'Balança comercial',
        'Investimento direto no país', 'Dívida líquida do setor público',
        'Resultado primário', 'Resultado nominal'
    ]

# --- UTILITÁRIOS ---
def retry_request(func, max_attempts: int = 3, delay: float = 1.0):
    """Decorator para retry de requisições"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(delay)
        return None
    return wrapper

def formatar_numero(valor: float, tipo: str = 'moeda') -> str:
    """Formatação padronizada de números"""
    if pd.isna(valor):
        return "-"
    
    formatadores = {
        'moeda': f"R$ {valor:,.2f}",
        'percentual': f"{valor:.2f}%",
        'bilhao': f"R$ {valor:.1f} Bi",
        'trilhao': f"R$ {valor:.2f} Tri",
        'dolar_bi': f"US$ {valor:.1f} Bi"
    }
    
    return formatadores.get(tipo, f"{valor:.2f}")

# --- FUNÇÕES DE CARGA DE DADOS (OTIMIZADAS) ---

@st.cache_data(ttl=3600)
def get_sidra_data(table_code: str, variable_code: str) -> pd.DataFrame:
    """Carrega dados do IBGE/Sidra com tratamento robusto"""
    try:
        dados_raw = sidrapy.get_table(
            table_code=table_code,
            territorial_level="1",
            ibge_territorial_code="all",
            variable=variable_code,
            period="last 360"
        )
        
        df = dados_raw.iloc[1:].copy()
        df = df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'})
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['D2C'].str.slice(0, 4)
        
        return processar_dataframe_comum(df)
    except Exception as e:
        st.error(f"Erro ao carregar Sidra: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_bcb_data(codigo_serie: str) -> pd.DataFrame:
    """Carrega dados do Banco Central com retry automático"""
    @retry_request
    def fetch_bcb():
        url = APIConfig.BCB_BASE.format(codigo_serie) + "?formato=json"
        response = requests.get(
            url,
            headers=APIConfig.HEADERS,
            verify=False,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    try:
        data = fetch_bcb()
        df = pd.DataFrame(data)
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        
        return processar_dataframe_comum(df)
    except Exception as e:
        st.warning(f"Erro BCB {codigo_serie}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_focus_data() -> pd.DataFrame:
    """Carrega expectativas do Boletim Focus"""
    try:
        url = f"{APIConfig.FOCUS_URL}?$top=2000&$orderby=Data%20desc&$format=json"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json()['value'])
        df = df[df['Indicador'].isin(APIConfig.INDICADORES_FOCUS)]
        
        df = df.rename(columns={
            'Data': 'data_relatorio',
            'DataReferencia': 'ano_referencia',
            'Mediana': 'previsao'
        })
        
        df['ano_referencia'] = df['ano_referencia'].astype(int)
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        
        return df
    except Exception as e:
        st.warning(f"Focus indisponível: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_currency_realtime() -> pd.DataFrame:
    """Cotações em tempo real via Yahoo Finance"""
    try:
        tickers = {"USDBRL=X": "USDBRL", "EURBRL=X": "EURBRL"}
        dados = {}
        
        for ticker, nome in tickers.items():
            info = yf.Ticker(ticker).fast_info
            preco = info['last_price']
            anterior = info['previous_close']
            variacao = ((preco - anterior) / anterior) * 100
            
            dados[nome] = {'bid': preco, 'pctChange': variacao}
        
        return pd.DataFrame.from_dict(dados, orient='index')
    except Exception as e:
        st.warning(f"Erro cotações: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_cambio_historico() -> pd.DataFrame:
    """Histórico completo de câmbio"""
    try:
        df = yf.download(
            ["USDBRL=X", "EURBRL=X"],
            start="1994-07-01",
            progress=False
        )['Close']
        
        # Normalização de fuso
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo').tz_localize(None)
        
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        return df.ffill()
    except Exception as e:
        st.warning(f"Erro histórico câmbio: {e}")
        return pd.DataFrame()

def processar_dataframe_comum(df: pd.DataFrame) -> pd.DataFrame:
    """Processamento padrão com cálculos acumulados"""
    df = df.sort_values('data_date', ascending=True).dropna(subset=['valor', 'data_date'])
    
    df['mes_num'] = df['data_date'].dt.month
    meses = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
             7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    
    # Cálculos
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12, min_periods=1).apply(np.prod, raw=True) - 1) * 100
    
    return df.sort_values('data_date', ascending=False)

@st.cache_data(ttl=3600)
def get_macro_real() -> Tuple[Dict, Dict]:
    """Carrega indicadores macroeconômicos com retry"""
    kpis = {}
    historico = {}
    mapa_meses = {f"{i:02d}": nome[:3].lower() 
                  for i, nome in enumerate(['Jan','Fev','Mar','Abr','Mai','Jun',
                                           'Jul','Ago','Set','Out','Nov','Dez'], 1)}
    
    for nome, codigo in APIConfig.SERIES_MACRO.items():
        @retry_request
        def fetch_serie():
            url = f"{APIConfig.BCB_BASE.format(codigo)}/ultimos/60?formato=json"
            resp = requests.get(url, headers=APIConfig.HEADERS, verify=False, timeout=5)
            resp.raise_for_status()
            return resp.json()
        
        try:
            dados = fetch_serie()
            if not dados:
                continue
            
            df = pd.DataFrame(dados)
            df['valor'] = pd.to_numeric(df['valor'])
            df['data_dt'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
            
            # Processamento para gráfico
            df_chart = df.copy()
            if nome == 'PIB':
                df_chart['valor'] /= 1_000_000
            elif nome in ['Balança Com.', 'Trans. Correntes', 'IDP']:
                df_chart['valor'] /= 1_000
            elif 'Primário' in nome or 'Nominal' in nome:
                df_chart['valor'] = (df_chart['valor'] * -1) / 1_000
            
            historico[nome] = df_chart
            
            # KPI
            ultimo = df.iloc[-1]
            ano_atual = ultimo['data_dt'].year
            mes_str = ultimo['data'].split('/')[1]
            data_curta = f"{mapa_meses[mes_str]}/{str(ano_atual)[2:]}"
            
            if nome == 'PIB':
                val_kpi = ultimo['valor'] / 1_000_000
            elif nome == 'Dívida Líq.':
                val_kpi = ultimo['valor']
            else:
                df_ano = df[df['data_dt'].dt.year == ano_atual]
                soma = df_ano['valor'].sum()
                val_kpi = (soma * -1 if 'Primário' in nome or 'Nominal' in nome else soma) / 1_000
            
            kpis[nome] = {'valor': val_kpi, 'data': data_curta, 'ano': ano_atual}
            
        except Exception as e:
            st.warning(f"Erro ao carregar {nome}: {e}")
            continue
    
    return kpis, historico

# --- CÁLCULO DE CORREÇÃO ---
def calcular_correcao(df: pd.DataFrame, valor: float, 
                     data_ini: str, data_fim: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Calcula correção monetária entre duas datas"""
    is_reverso = data_ini > data_fim
    periodo_inicio = min(data_ini, data_fim)
    periodo_fim = max(data_ini, data_fim)
    
    mask = (df['D2C'] >= periodo_inicio) & (df['D2C'] <= periodo_fim)
    df_periodo = df.loc[mask]
    
    if df_periodo.empty:
        return None, "⚠️ Período sem dados suficientes"
    
    fator = df_periodo['fator'].prod()
    valor_final = valor / fator if is_reverso else valor * fator
    percentual = (fator - 1) * 100
    
    return {
        'valor_final': valor_final,
        'percentual': percentual,
        'fator': fator,
        'is_reverso': is_reverso,
        'meses': len(df_periodo)
    }, None

# ==============================================================================
# INTERFACE - SIDEBAR
# ==============================================================================

try:
    st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
except:
    st.sidebar.markdown("## 📊 VPL CONSULTORIA")

st.sidebar.header("⚙️ Configurações")

# Seleção de índice
INDICES = {
    "IPCA (Inflação Oficial)": ("1737", "63", "IPCA"),
    "INPC (Salários)": ("1736", "44", "INPC"),
    "IGP-M (Aluguéis)": (None, "189", "IGP-M"),
    "SELIC (Taxa Básica)": (None, "4390", "SELIC"),
    "CDI (Investimentos)": (None, "4391", "CDI")
}

tipo_indice = st.sidebar.selectbox("Selecione o Indicador", list(INDICES.keys()))

# Carrega dados
with st.spinner("🔄 Carregando dados..."):
    config = INDICES[tipo_indice]
    
    if config[0]:  # Sidra
        df = get_sidra_data(config[0], config[1])
    else:  # BCB
        df = get_bcb_data(config[1])
    
    cor_tema = CORES[config[2]]

if df.empty:
    st.error("❌ Erro ao carregar dados. Tente novamente.")
    st.stop()

# --- CALCULADORA ---
st.sidebar.divider()
st.sidebar.subheader("🧮 Calculadora de Correção")

valor_input = st.sidebar.number_input(
    "Valor (R$)",
    min_value=0.01,
    value=1000.00,
    step=100.00,
    format="%.2f"
)

lista_anos = sorted(df['ano'].unique(), reverse=True)
meses_nome = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
mapa_meses = {m: f"{i:02d}" for i, m in enumerate(meses_nome, 1)}

st.sidebar.markdown("**📅 Data Inicial**")
c1, c2 = st.sidebar.columns(2)
mes_ini = c1.selectbox("Mês", meses_nome, index=0, key="mes_ini", label_visibility="collapsed")
ano_ini = c2.selectbox("Ano", lista_anos, index=min(1, len(lista_anos)-1), key="ano_ini", label_visibility="collapsed")

st.sidebar.markdown("**🎯 Data Final**")
c3, c4 = st.sidebar.columns(2)
mes_fim = c3.selectbox("Mês", meses_nome, index=11, key="mes_fim", label_visibility="collapsed")
ano_fim = c4.selectbox("Ano", lista_anos, index=0, key="ano_fim", label_visibility="collapsed")

if st.sidebar.button("🚀 Calcular", type="primary", use_container_width=True):
    code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
    code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
    
    resultado, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
    
    if erro:
        st.sidebar.error(erro)
    else:
        st.sidebar.divider()
        nome_idx = config[2]
        tipo_op = "💰 Rendimento" if nome_idx in ["SELIC", "CDI"] else "📈 Correção"
        label = "🔻 Descapitalização" if resultado['is_reverso'] else f"{tipo_op} ({nome_idx})"
        
        st.sidebar.markdown(f"**{label}**")
        st.sidebar.markdown(
            f"<h1 style='color: {cor_tema}; margin:0;'>R$ {resultado['valor_final']:,.2f}</h1>",
            unsafe_allow_html=True
        )
        
        col_res1, col_res2 = st.sidebar.columns(2)
        col_res1.metric("Total", f"{resultado['percentual']:.2f}%")
        col_res2.metric("Fator", f"{resultado['fator']:.6f}")
        st.sidebar.caption(f"Período: {resultado['meses']} meses")

# ==============================================================================
# ÁREA PRINCIPAL
# ==============================================================================

# EXPANDER 1: FOCUS + CÂMBIO
with st.expander("🔭 **Expectativas de Mercado (Focus) & Câmbio em Tempo Real**", expanded=False):
    col_top1, col_top2 = st.columns([2, 1])
    
    # FOCUS
    df_focus = get_focus_data()
    ano_atual = date.today().year
    
    with col_top1:
        if not df_focus.empty:
            ultima_data = df_focus['data_relatorio'].max()
            df_last = df_focus[df_focus['data_relatorio'] == ultima_data]
            data_str = pd.to_datetime(ultima_data).strftime('%d/%m/%Y')
            
            st.markdown(f"#### 📊 Boletim Focus ({data_str})")
            
            # Destaques
            df_atual = df_last[df_last['ano_referencia'] == ano_atual]
            pivot = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='mean')
            
            fc1, fc2, fc3, fc4 = st.columns(4)
            
            def safe_get(idx):
                return pivot.loc[idx, 'previsao'] if idx in pivot.index else 0
            
            fc1.metric(f"IPCA {ano_atual}", formatar_numero(safe_get('IPCA'), 'percentual'))
            fc2.metric(f"Selic {ano_atual}", formatar_numero(safe_get('Selic'), 'percentual'))
            fc3.metric(f"PIB {ano_atual}", formatar_numero(safe_get('PIB Total'), 'percentual'))
            fc4.metric(f"Dólar {ano_atual}", f"R$ {safe_get('Câmbio'):.2f}")
            
            # Tabela completa
            st.divider()
            st.markdown("##### 📅 Projeções Macroeconômicas")
            
            anos_proj = [ano_atual + i for i in range(3)]
            df_table = df_last[df_last['ano_referencia'].isin(anos_proj)]
            pivot_multi = df_table.pivot_table(
                index='Indicador',
                columns='ano_referencia',
                values='previsao'
            )
            
            # Formatação inteligente
            df_display = pivot_multi.copy()
            for col in df_display.columns:
                df_display[col] = df_display.apply(lambda row: 
                    f"R$ {row[col]:.2f}" if 'Câmbio' in row.name
                    else f"US$ {row[col]:.2f} B" if any(x in row.name for x in ['comercial', 'Conta', 'Investimento'])
                    else f"{row[col]:.2f}%", axis=1
                )
            
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("⚠️ Dados do Focus temporariamente indisponíveis")
    
    # CÂMBIO REAL TIME
    with col_top2:
        st.markdown("#### 💱 Câmbio (Agora)")
        df_moedas = get_currency_realtime()
        
        if not df_moedas.empty:
            mc1, mc2 = st.columns(2)
            try:
                usd = df_moedas.loc['USDBRL']
                eur = df_moedas.loc['EURBRL']
                
                mc1.metric("Dólar", f"R$ {float(usd['bid']):.2f}", f"{float(usd['pctChange']):.2f}%")
                mc2.metric("Euro", f"R$ {float(eur['bid']):.2f}", f"{float(eur['pctChange']):.2f}%")
            except:
                st.info("Aguardando atualização...")
        else:
            st.info("Carregando cotações...")

# EXPANDER 2: MACRO REALIZADO
with st.expander("🧩 **Conjuntura Macroeconômica (Dados Realizados BCB)**", expanded=False):
    st.markdown("Monitoramento dos principais indicadores oficiais da economia brasileira")
    
    kpis, historico = get_macro_real()
    
    if kpis:
        # Função helper
        def get_kpi(nome):
            return kpis.get(nome, {'valor': 0, 'data': '-', 'ano': '-'})
        
        # SEÇÃO 1: Atividade & Fiscal
        st.markdown("#### 🏛️ Atividade & Fiscal")
        c1, c2, c3, c4 = st.columns(4)
        
        k_pib, k_div, k_pri, k_nom = [get_kpi(x) for x in 
            ['PIB', 'Dívida Líq.', 'Res. Primário', 'Res. Nominal']]
        
        c1.metric(f"PIB 12m ({k_pib['data']})", formatar_numero(k_pib['valor'], 'trilhao'))
        c2.metric(f"Dívida Líquida ({k_div['data']})", formatar_numero(k_div['valor'], 'percentual'))
        c3.metric(f"Primário YTD ({k_pri['ano']})", formatar_numero(k_pri['valor'], 'bilhao'))
        c4.metric(f"Nominal YTD ({k_nom['ano']})", formatar_numero(k_nom['valor'], 'bilhao'))
        
        st.divider()
        
        # SEÇÃO 2: Setor Externo
        st.markdown("#### 🚢 Setor Externo (Acumulado Ano)")
        c5, c6, c7 = st.columns(3)
        
        k_bal, k_tra, k_idp = [get_kpi(x) for x in 
            ['Balança Com.', 'Trans. Correntes', 'IDP']]
        
        c5.metric(f"Balança Com. ({k_bal['data']})", formatar_numero(k_bal['valor'], 'dolar_bi'))
        c6.metric(f"Trans. Correntes ({k_tra['data']})", formatar_numero(k_tra['valor'], 'dolar_bi'))
        c7.metric(f"IDP ({k_idp['data']})", formatar_numero(k_idp['valor'], 'dolar_bi'))
        
        st.divider()
        
        # GRÁFICOS
        st.markdown("#### 📈 Evolução Temporal (5 Anos)")
        tab_ativ, tab_fisc, tab_ext = st.tabs(["Atividade", "Fiscal", "Externo"])
        
        def plot_serie(nome, cor, tipo='line'):
            if nome not in historico:
                st.info(f"Dados de {nome} indisponíveis")
                return
            
            df_plot = historico[nome]
            
            if tipo == 'line':
                fig = px.line(df_plot, x='data_dt', y='valor')
            else:
                fig = px.area(df_plot, x='data_dt', y='valor')
            
            fig.update_traces(line_color=cor, fillcolor=f"rgba{tuple(list(int(cor.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.3])}")
            fig.update_layout(
                title=nome,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=40, b=20),
                height=280,
                xaxis_title=None,
                yaxis_title=None,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_ativ:
            col_a1, col_a2 = st.columns(2)
            with col_a1: plot_serie('PIB', '#00D9FF')
            with col_a2: plot_serie('Dívida Líq.', '#FFD93D')
        
        with tab_fisc:
            st.caption("💡 Valores positivos = Superávit | Negativos = Déficit")
            col_f1, col_f2 = st.columns(2)
            with col_f1: plot_serie('Res. Primário', '#00FFA3', 'area')
            with col_f2: plot_serie('Res. Nominal', '#FF6B6B', 'area')
        
        with tab_ext:
            st.caption("💡 Fluxos mensais em Bilhões de Dólares (US$)")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1: plot_serie('Balança Com.', '#00D9FF', 'area')
            with col_e2: plot_serie('Trans. Correntes', '#FF6B6B', 'area')
            with col_e3: plot_serie('IDP', '#00FFA3', 'area')
    else:
        st.warning("⚠️ Dados macroeconômicos temporariamente indisponíveis")

# EXPANDER 3: HISTÓRICO DE CÂMBIO
with st.expander("💸 **Histórico Completo de Câmbio (1994-2025)**", expanded=False):
    st.markdown("Evolução do Dólar e Euro desde o início do Plano Real")
    
    df_cambio = get_cambio_historico()
    
    if not df_cambio.empty:
        # Resumo atual
        ultimo = df_cambio.iloc[-1]
        penultimo = df_cambio.iloc[-2]
        data_atual = df_cambio.index[-1].strftime('%d/%m/%Y')
        
        st.markdown(f"**📅 Fechamento: {data_atual}**")
        col_cam1, col_cam2 = st.columns(2)
        
        usd_val = ultimo['Dólar']
        usd_var = ((usd_val - penultimo['Dólar']) / penultimo['Dólar']) * 100
        eur_val = ultimo['Euro']
        eur_var = ((eur_val - penultimo['Euro']) / penultimo['Euro']) * 100
        
        col_cam1.metric("Dólar", f"R$ {usd_val:.2f}", f"{usd_var:.2f}%")
        col_cam2.metric("Euro", f"R$ {eur_val:.2f}", f"{eur_var:.2f}%")
        
        st.divider()
        
        # Abas
        tab_graf, tab_matriz, tab_tabela = st.tabs(["📈 Gráfico Interativo", "🗓️ Matriz de Retornos", "📋 Dados Diários"])
        
        with tab_graf:
            cores_cam = {"Dólar": "#00FFA3", "Euro": "#00D9FF"}
            
            fig_cam = px.line(
                df_cambio,
                x=df_cambio.index,
                y=['Dólar', 'Euro'],
                color_discrete_map=cores_cam
            )
            
            fig_cam.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E0E0E0"),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(0,0,0,0)"
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=450,
                xaxis_title=None,
                yaxis_title="Cotação (R$)"
            )
            
            fig_cam.update_xaxes(showgrid=False, rangeslider_visible=True)
            fig_cam.update_yaxes(showgrid=True, gridcolor='#333333', tickprefix="R$ ")
            
            st.plotly_chart(fig_cam, use_container_width=True)
        
        with tab_matriz:
            st.caption("📊 Variação percentual mensal das cotações")
            
            moeda_sel = st.radio(
                "Selecione a moeda:",
                ["Dólar", "Euro"],
                horizontal=True,
                key="moeda_matriz"
            )
            
            # Processamento
            df_mensal = df_cambio[[moeda_sel]].resample('ME').last()
            df_ret = df_mensal.pct_change() * 100
            df_ret['ano'] = df_ret.index.year
            df_ret['mes'] = df_ret.index.month
            
            meses_ord = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                        7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
            df_ret['mes_nome'] = df_ret['mes'].map(meses_ord)
            
            try:
                matriz = df_ret.pivot(index='ano', columns='mes_nome', values=moeda_sel)
                ordem_meses = list(meses_ord.values())
                matriz = matriz[ordem_meses].sort_index(ascending=False)
                
                st.dataframe(
                    matriz.style.background_gradient(
                        cmap=cmap_custom,
                        vmin=-8,
                        vmax=8,
                        axis=None
                    ).format("{:.2f}%"),
                    use_container_width=True,
                    height=450
                )
            except Exception as e:
                st.warning(f"⚠️ Dados insuficientes: {e}")
        
        with tab_tabela:
            df_view = df_cambio.sort_index(ascending=False).reset_index()
            df_view.columns = ['Data', 'Dólar', 'Euro']
            df_view['Data'] = pd.to_datetime(df_view['Data']).dt.strftime('%d/%m/%Y')
            
            # Download
            csv_data = df_view.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv_data,
                "historico_cambio.csv",
                "text/csv",
                use_container_width=True
            )
            
            st.dataframe(
                df_view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Dólar": st.column_config.NumberColumn(format="R$ %.4f"),
                    "Euro": st.column_config.NumberColumn(format="R$ %.4f")
                },
                height=450
            )
    else:
        st.warning("⚠️ Histórico de câmbio temporariamente indisponível")

# ==============================================================================
# PAINEL PRINCIPAL DO ÍNDICE SELECIONADO
# ==============================================================================

st.title(f"📊 {config[2]} - Análise Completa")
st.markdown(f"**Última atualização:** {df.iloc[0]['data_fmt']}")

# KPIs Principais
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Taxa do Mês",
    formatar_numero(df.iloc[0]['valor'], 'percentual'),
    help="Variação no último mês disponível"
)

kpi2.metric(
    "Acumulado 12 Meses",
    formatar_numero(df.iloc[0]['acum_12m'], 'percentual'),
    help="Variação acumulada nos últimos 12 meses"
)

kpi3.metric(
    "Acumulado no Ano",
    formatar_numero(df.iloc[0]['acum_ano'], 'percentual'),
    help="Variação acumulada no ano corrente (YTD)"
)

kpi4.metric(
    "Histórico desde",
    df['ano'].min(),
    help="Primeiro ano disponível na série histórica"
)

# Tabs principais
tab1, tab2, tab3 = st.tabs([
    "📈 Visualização Gráfica",
    "🗓️ Matriz Histórica",
    "📋 Dados Detalhados"
])

# TAB 1: GRÁFICO
with tab1:
    st.markdown("#### Evolução do Indicador nos Últimos Anos")
    
    # Filtros
    col_filtro1, col_filtro2 = st.columns([3, 1])
    
    with col_filtro1:
        anos_disponiveis = sorted(df['ano'].astype(int).unique())
        ano_inicio_grafico = st.select_slider(
            "Período de análise (a partir de):",
            options=anos_disponiveis,
            value=max(anos_disponiveis) - 10 if len(anos_disponiveis) > 10 else min(anos_disponiveis)
        )
    
    with col_filtro2:
        tipo_viz = st.selectbox(
            "Visualização:",
            ["Acumulado 12 Meses", "Mensal", "Acumulado no Ano"]
        )
    
    # Preparação dos dados
    df_chart = df[df['ano'].astype(int) >= ano_inicio_grafico].sort_values('data_date')
    
    coluna_y = {
        "Acumulado 12 Meses": "acum_12m",
        "Mensal": "valor",
        "Acumulado no Ano": "acum_ano"
    }[tipo_viz]
    
    # Gráfico
    fig_principal = go.Figure()
    
    fig_principal.add_trace(go.Scatter(
        x=df_chart['data_date'],
        y=df_chart[coluna_y],
        mode='lines',
        name=config[2],
        line=dict(color=cor_tema, width=3),
        fill='tozeroy',
        fillcolor=f'rgba({int(cor_tema[1:3], 16)}, {int(cor_tema[3:5], 16)}, {int(cor_tema[5:7], 16)}, 0.2)',
        hovertemplate='<b>%{x|%b/%Y}</b><br>Valor: %{y:.2f}%<extra></extra>'
    ))
    
    # Linha de referência
    fig_principal.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5
    )
    
    fig_principal.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E0E0E0", size=12),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=20, b=0),
        height=500,
        xaxis_title=None,
        yaxis_title="Percentual (%)",
        showlegend=False
    )
    
    fig_principal.update_xaxes(showgrid=False)
    fig_principal.update_yaxes(showgrid=True, gridcolor='#333333', ticksuffix="%")
    
    st.plotly_chart(fig_principal, use_container_width=True)
    
    # Estatísticas do período
    st.divider()
    st.markdown("##### 📊 Estatísticas do Período Selecionado")
    
    estat1, estat2, estat3, estat4 = st.columns(4)
    
    estat1.metric(
        "Média Mensal",
        formatar_numero(df_chart['valor'].mean(), 'percentual')
    )
    estat2.metric(
        "Mediana",
        formatar_numero(df_chart['valor'].median(), 'percentual')
    )
    estat3.metric(
        "Máxima",
        formatar_numero(df_chart['valor'].max(), 'percentual')
    )
    estat4.metric(
        "Mínima",
        formatar_numero(df_chart['valor'].min(), 'percentual')
    )

# TAB 2: MATRIZ
with tab2:
    st.markdown("#### Matriz de Variações Mensais")
    st.caption("💡 Cores: Verde = Alta | Branco = Neutro | Vermelho = Baixa")
    
    try:
        matriz_principal = df.pivot(index='ano', columns='mes_nome', values='valor')
        ordem_meses = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
        matriz_principal = matriz_principal[ordem_meses].sort_index(ascending=False)
        
        st.dataframe(
            matriz_principal.style.background_gradient(
                cmap=cmap_custom,
                vmin=-2,
                vmax=2,
                axis=None
            ).format("{:.2f}%", na_rep="-"),
            use_container_width=True,
            height=500
        )
        
        # Análise por ano
        st.divider()
        st.markdown("##### 📅 Acumulado por Ano Completo")
        
        df_anual = df.copy()
        df_anual['ano_int'] = df_anual['ano'].astype(int)
        
        # Pega apenas anos completos (12 meses)
        anos_completos = df_anual.groupby('ano_int').size()
        anos_completos = anos_completos[anos_completos >= 12].index.tolist()
        
        df_anual_filtrado = df_anual[df_anual['ano_int'].isin(anos_completos)]
        
        # Calcula acumulado correto por ano
        acum_anual = df_anual_filtrado.groupby('ano_int').apply(
            lambda x: ((x['fator'].prod() - 1) * 100)
        ).reset_index()
        acum_anual.columns = ['Ano', 'Acumulado (%)']
        acum_anual = acum_anual.sort_values('Ano', ascending=False)
        
        # Gráfico de barras
        fig_anual = px.bar(
            acum_anual,
            x='Ano',
            y='Acumulado (%)',
            color='Acumulado (%)',
            color_continuous_scale=['#FF6B6B', '#FFFFFF', '#4ECDC4'],
            color_continuous_midpoint=0
        )
        
        fig_anual.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        
        fig_anual.update_traces(
            hovertemplate='<b>%{x}</b><br>Acumulado: %{y:.2f}%<extra></extra>'
        )
        
        st.plotly_chart(fig_anual, use_container_width=True)
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao gerar matriz: {e}")

# TAB 3: TABELA
with tab3:
    st.markdown("#### Dados Históricos Completos")
    
    # Seletor de período
    col_tab1, col_tab2 = st.columns([3, 1])
    
    with col_tab1:
        anos_tabela = st.multiselect(
            "Filtrar por ano:",
            options=sorted(df['ano'].unique(), reverse=True),
            default=sorted(df['ano'].unique(), reverse=True)[:3]
        )
    
    with col_tab2:
        ordenacao = st.selectbox(
            "Ordenar por:",
            ["Mais Recente", "Mais Antigo", "Maior Valor", "Menor Valor"]
        )
    
    # Filtragem
    df_tabela = df[df['ano'].isin(anos_tabela)].copy()
    
    # Ordenação
    if ordenacao == "Mais Recente":
        df_tabela = df_tabela.sort_values('data_date', ascending=False)
    elif ordenacao == "Mais Antigo":
        df_tabela = df_tabela.sort_values('data_date', ascending=True)
    elif ordenacao == "Maior Valor":
        df_tabela = df_tabela.sort_values('valor', ascending=False)
    else:  # Menor Valor
        df_tabela = df_tabela.sort_values('valor', ascending=True)
    
    # Preparação para exibição
    df_exibir = df_tabela[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].copy()
    df_exibir.columns = ['Período', 'Mensal (%)', 'Acum. Ano (%)', 'Acum. 12M (%)']
    
    # Download
    csv_principal = df_exibir.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download dos Dados (CSV)",
        csv_principal,
        f"{config[2]}_historico_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True
    )
    
    # Exibição
    st.dataframe(
        df_exibir,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mensal (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Acum. Ano (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Acum. 12M (%)": st.column_config.NumberColumn(format="%.2f%%")
        },
        height=500
    )
    
    # Resumo estatístico
    st.divider()
    st.markdown("##### 📈 Resumo Estatístico do Período Filtrado")
    
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    
    col_stat1.metric("Média", f"{df_tabela['valor'].mean():.2f}%")
    col_stat2.metric("Mediana", f"{df_tabela['valor'].median():.2f}%")
    col_stat3.metric("Desvio Padrão", f"{df_tabela['valor'].std():.2f}%")
    col_stat4.metric("Máximo", f"{df_tabela['valor'].max():.2f}%")
    col_stat5.metric("Mínimo", f"{df_tabela['valor'].min():.2f}%")

# ==============================================================================
# RODAPÉ
# ==============================================================================

st.divider()
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("""
    **📚 Fontes de Dados:**
    - IBGE/Sidra (IPCA, INPC)
    - Banco Central (BCB/SGS)
    - Yahoo Finance (Câmbio)
    """)

with col_footer2:
    st.markdown("""
    **🔄 Atualização:**
    - Índices: Mensal
    - Câmbio: Tempo Real
    - Macro: Mensal (BCB)
    """)

with col_footer3:
    st.markdown("""
    **💼 VPL Consultoria**
    
    Inteligência Financeira
    
    © 2025
    """)
