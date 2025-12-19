import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
import time
from datetime import date, datetime
from typing import Dict, Tuple, Optional, Any
import logging

# =============================================================================
# 1. CONFIGURAÇÃO E CONSTANTES (A "Verdade" do Sistema)
# =============================================================================

# Configuração de Logs para Debug (fundamental para não engolir erros)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    """Centraliza todas as constantes e configurações de API."""
    # URLs
    BCB_API_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados"
    FOCUS_API_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
    
    # Headers para simular navegador real (evita bloqueios)
    REQUEST_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }

    # Catálogo de Indicadores
    INDICES = {
        "IPCA": {"source": "sidra", "table": "1737", "variable": "63", "name": "IPCA (Inflação)", "color": "#00D9FF"},
        "INPC": {"source": "sidra", "table": "1736", "variable": "44", "name": "INPC (Salários)", "color": "#00FFA3"},
        "IGP-M": {"source": "bcb", "code": "189", "name": "IGP-M (Aluguéis)", "color": "#FF6B6B"},
        "SELIC": {"source": "bcb", "code": "4390", "name": "SELIC (Juros)", "color": "#FFD93D"},
        "CDI": {"source": "bcb", "code": "4391", "name": "CDI (Investimentos)", "color": "#A8E6CF"}
    }

    MACRO_SERIES = {
        'PIB (R$ Bi)': {'code': 4382, 'type': 'last'},
        'Dívida Líq. (% PIB)': {'code': 4513, 'type': 'last'},
        'Res. Primário (% PIB)': {'code': 5793, 'type': 'invert'},
        'Balança Com. (US$ Mi)': {'code': 22707, 'type': 'sum_12m'},
    }

# =============================================================================
# 2. CORE & UTILS (Ferramentas do Desenvolvedor)
# =============================================================================

def safe_request(url: str, params: dict = None, timeout: int = 10) -> Optional[Any]:
    """
    Wrapper robusto para requisições HTTP.
    Removemos o verify=False inseguro, mas tratamos exceções especificas.
    """
    try:
        response = requests.get(
            url, 
            headers=AppConfig.REQUEST_HEADERS, 
            params=params, 
            timeout=timeout,
            verify=True # Voltamos para True por segurança. Se falhar, o log avisa.
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.SSLError:
        logging.warning(f"Erro de SSL ao acessar {url}. Tentando fallback inseguro...")
        # Fallback controlado apenas se SSL falhar
        try:
            return requests.get(url, headers=AppConfig.REQUEST_HEADERS, verify=False, timeout=timeout).json()
        except Exception as e:
            logging.error(f"Falha total no request: {e}")
            return None
    except Exception as e:
        logging.error(f"Erro na requisição {url}: {e}")
        return None

def hex_to_rgba(hex_color: str, opacity: float = 0.2) -> str:
    """Utilitário visual para gráficos."""
    hex_color = hex_color.lstrip('#')
    return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {opacity})"

# =============================================================================
# 3. DATA LAYER (Busca e Padronização)
# =============================================================================

@st.cache_data(ttl=86400) # Cache longo para dados estruturais (SIDRA/BCB)
def fetch_sidra_series(table_code: str, variable_code: str, periods: str = "last 120") -> pd.DataFrame:
    """Busca dados do IBGE/SIDRA e retorna DataFrame padronizado: [data_date, valor, ano]"""
    try:
        raw = sidrapy.get_table(
            table_code=table_code, territorial_level="1",
            ibge_territorial_code="all", variable=variable_code,
            period=periods
        )
        if raw.empty or 'V' not in raw.columns: 
            return pd.DataFrame()

        df = raw.iloc[1:].copy()
        df = df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'})
        
        # Tratamento seguro de tipos
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['data_date'].dt.year
        
        return df[['data_date', 'valor', 'ano']].dropna().sort_values('data_date')
    except Exception as e:
        logging.error(f"Erro SIDRA: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_bcb_series(code: str) -> pd.DataFrame:
    """Busca dados do BCB e padroniza."""
    url = AppConfig.BCB_API_BASE.format(code) + "?formato=json"
    data = safe_request(url)
    
    if not data: return pd.DataFrame()

    df = pd.DataFrame(data)
    df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
    df['ano'] = df['data_date'].dt.year
    
    return df[['data_date', 'valor', 'ano']].dropna().sort_values('data_date')

@st.cache_data(ttl=3600) # Cache médio para Focus
def fetch_focus_expectations() -> pd.DataFrame:
    """Busca relatório Focus do Banco Central."""
    url = f"{AppConfig.FOCUS_API_URL}?$top=1000&$orderby=Data desc&$format=json"
    data = safe_request(url)
    
    if not data or 'value' not in data: return pd.DataFrame()
    
    df = pd.DataFrame(data['value'])
    cols_map = {'Indicador': 'indicador', 'Data': 'data_relatorio', 'DataReferencia': 'ano_ref', 'Mediana': 'valor'}
    
    if not set(cols_map.keys()).issubset(df.columns): return pd.DataFrame()
    
    df = df.rename(columns=cols_map)
    df['ano_ref'] = pd.to_numeric(df['ano_ref'], errors='coerce')
    return df

# =============================================================================
# 4. BUSINESS LOGIC & INTELLIGENCE (O Cérebro)
# =============================================================================

def enrich_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona cálculos financeiros (acumulados, fatores) ao DataFrame base."""
    if df.empty: return df
    
    df = df.sort_values('data_date')
    df['fator'] = 1 + (df['valor'] / 100)
    
    # Acumulado no Ano
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    
    # Acumulado 12 Meses (Rolling Window)
    df['acum_12m'] = (df['fator'].rolling(12).apply(np.prod, raw=True) - 1) * 100
    
    # Dados auxiliares para UI
    df['mes_nome'] = df['data_date'].dt.strftime('%b')
    df['fmt_data'] = df['data_date'].dt.strftime('%b/%Y')
    
    return df.sort_values('data_date', ascending=False)

def generate_market_insight(df: pd.DataFrame, indice_nome: str) -> str:
    """
    NOVO: Analisa os dados e gera um texto de diagnóstico econômico.
    """
    if df.empty or len(df) < 13: return "Dados insuficientes para análise."
    
    atual = df.iloc[0]
    anterior = df.iloc[1]
    media_12m = df.iloc[:12]['valor'].mean()
    
    tendencia = "estável"
    if atual['valor'] > anterior['valor'] * 1.05: tendencia = "aceleração"
    elif atual['valor'] < anterior['valor'] * 0.95: tendencia = "desaceleração"
    
    comparacao_media = "acima" if atual['valor'] > media_12m else "abaixo"
    
    return (f"O {indice_nome} apresenta **{tendencia}** na margem ({atual['valor']:.2f}% vs {anterior['valor']:.2f}%). "
            f"No acumulado de 12 meses, o índice está em {atual['acum_12m']:.2f}%, rodando {comparacao_media} "
            f"da média mensal recente ({media_12m:.2f}%).")

# =============================================================================
# 5. UI LAYER (Interface Gráfica)
# =============================================================================

def render_sidebar():
    st.sidebar.markdown("## 📊 VPL Consultoria")
    st.sidebar.caption("Sistema de Inteligência Financeira")
    
    # Seletor Principal
    indicador_key = st.sidebar.selectbox("Indicador Principal", list(AppConfig.INDICES.keys()))
    config = AppConfig.INDICES[indicador_key]
    
    # Carga de Dados (Lazy Load)
    with st.spinner(f"Processando {config['name']}..."):
        if config['source'] == 'sidra':
            df_raw = fetch_sidra_series(config['table'], config['variable'])
        else:
            df_raw = fetch_bcb_series(config['code'])
            
        df_processed = enrich_timeseries(df_raw)
        
    if not df_processed.empty:
        st.sidebar.success(f"Dados atualizados: {df_processed.iloc[0]['fmt_data']}")
    else:
        st.sidebar.error("Falha na conexão com as fontes de dados.")
        st.stop()
        
    return df_processed, config

def main():
    # 1. Sidebar e Carga
    df, config = render_sidebar()
    
    # 2. Cabeçalho Principal com KPI
    st.title(f"{config['name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    ultimo = df.iloc[0]
    
    col1.metric("Taxa Mensal", f"{ultimo['valor']:.2f}%", delta=f"{ultimo['valor'] - df.iloc[1]['valor']:.2f} p.p.")
    col2.metric("Acumulado 12M", f"{ultimo['acum_12m']:.2f}%")
    col3.metric("Acumulado Ano", f"{ultimo['acum_ano']:.2f}%")
    col4.metric("Série Desde", int(ultimo['ano'])) # Conversão explícita para int

    # 3. Área de Inteligência (Diagnóstico)
    st.markdown("### 🧠 Análise de Conjuntura")
    with st.container():
        st.info(generate_market_insight(df, config['name']))

    # 4. Visualização
    tab_graf, tab_matriz, tab_dados = st.tabs(["📈 Tendência", "🗓️ Sazonalidade", "💾 Dados Brutos"])
    
    with tab_graf:
        fig = px.area(df, x='data_date', y='acum_12m', title="Curva de Tendência (12 Meses)")
        fig.update_traces(line_color=config['color'], fillcolor=hex_to_rgba(config['color']))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_matriz:
        # Pivotagem Segura
        pivot = df.pivot_table(index='ano', columns='mes_nome', values='valor', sort=False)
        # Ordenação de meses correta (não alfabética)
        meses_ordem = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Mapeamento para PT-BR se necessário, ou ajuste no Data Layer
        st.dataframe(pivot.style.background_gradient(cmap='RdYlGn_r', axis=None), use_container_width=True)

if __name__ == "__main__":
    main()
