import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import requests
import yfinance as yf
import time
import logging
import traceback
import hashlib
import json
from pathlib import Path
from functools import wraps
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Any

# --- CONFIGURAÇÃO INICIAL ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurações da página
st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTES E CONFIGURAÇÕES ---
class Config:
    """Configurações centralizadas da aplicação"""
    # Timeouts e retries
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 3
    
    # Cache TTL (em segundos)
    TTL_SHORT = 300      # 5 minutos (dados em tempo real)
    TTL_MEDIUM = 3600    # 1 hora (dados diários)
    TTL_LONG = 86400     # 1 dia (dados históricos)
    
    # URLs das APIs
    BCB_API = "https://api.bcb.gov.br/dados/serie"
    FOCUS_API = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata"
    YAHOO_CURRENCIES = ["USDBRL=X", "EURBRL=X"]
    
    # Headers padrão
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8'
    }
    
    # Cores temáticas
    CORES = {
        'IPCA': '#00BFFF',      # Azul
        'INPC': '#00FF7F',      # Verde
        'IGP-M': '#FF6347',     # Vermelho
        'SELIC': '#FFD700',     # Dourado
        'CDI': '#FFFFFF',       # Branco
        'positivo': '#4CAF50',  # Verde sucesso
        'negativo': '#F44336',  # Vermelho alerta
        'neutro': '#607D8B'     # Cinza
    }

# Metadados das fontes
METADADOS_FONTES = {
    'IPCA': {
        'fonte': 'IBGE/SIDRA (Tabela 1737)',
        'atualizacao': 'Mensal',
        'qualidade': 'Alta (Oficial)',
        'link': 'https://sidra.ibge.gov.br/tabela/1737',
        'descricao': 'Índice Nacional de Preços ao Consumidor Amplo - inflação oficial'
    },
    'INPC': {
        'fonte': 'IBGE/SIDRA (Tabela 1736)',
        'atualizacao': 'Mensal',
        'qualidade': 'Alta (Oficial)',
        'link': 'https://sidra.ibge.gov.br/tabela/1736',
        'descricao': 'Índice Nacional de Preços ao Consumidor - para famílias de baixa renda'
    },
    'IGP-M': {
        'fonte': 'FGV/IBRE via BCB (SGS 189)',
        'atualizacao': 'Mensal',
        'qualidade': 'Alta',
        'link': 'https://www.bcb.gov.br/estatisticas/igpm',
        'descricao': 'Índice Geral de Preços do Mercado - usado em contratos e aluguéis'
    },
    'SELIC': {
        'fonte': 'Banco Central (SGS 4390)',
        'atualizacao': 'Diária',
        'qualidade': 'Alta (Oficial)',
        'link': 'https://www.bcb.gov.br/estabilidadefinanceira/historicotaxasjuros',
        'descricao': 'Taxa básica de juros da economia brasileira'
    },
    'CDI': {
        'fonte': 'CETIP via BCB (SGS 4391)',
        'atualizacao': 'Diária',
        'qualidade': 'Alta',
        'link': 'https://www.bcb.gov.br/estatisticas/taxascdi',
        'descricao': 'Certificado de Depósito Interbancário - benchmark de investimentos'
    }
}

# --- DECORADORES E UTILITÁRIOS ---
def log_errors(func):
    """Decorador para log de erros"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Erro em {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    return wrapper

def criar_checksum(df: pd.DataFrame, fonte: str) -> str:
    """Cria checksum para verificação de integridade"""
    if df.empty:
        return ""
    data_str = pd.util.hash_pandas_object(df).sum()
    return hashlib.md5(str(data_str).encode()).hexdigest()

def verificar_dados(df: pd.DataFrame, nome: str) -> List[str]:
    """Verifica qualidade dos dados"""
    alertas = []
    
    if df.empty:
        return ["⚠️ Dados vazios"]
    
    # Verifica dados missing
    if 'valor' in df.columns:
        missing = df['valor'].isna().sum()
        if missing > 0:
            alertas.append(f"⚠️ {missing} valores missing")
    
    # Verifica atualização
    if 'data_date' in df.columns and not df.empty:
        ultima_data = df['data_date'].max()
        dias_atraso = (datetime.now() - ultima_data).days
        if dias_atraso > 60:
            alertas.append(f"⚠️ Dados podem estar desatualizados ({dias_atraso} dias)")
    
    # Verifica outliers (para alguns indicadores)
    if 'valor' in df.columns and len(df) > 12:
        media = df['valor'].abs().mean()
        std = df['valor'].abs().std()
        outliers = df[abs(df['valor']) > media + 3*std]
        if len(outliers) > 0:
            alertas.append(f"⚠️ {len(outliers)} possíveis outliers")
    
    return alertas

# --- FUNÇÕES DE CARGA DE DADOS ---
@st.cache_data(ttl=Config.TTL_LONG)
@log_errors
def carregar_dados_sidra(codigo_tabela: str, codigo_variavel: str) -> pd.DataFrame:
    """Carrega dados do SIDRA/IBGE"""
    try:
        dados = sidrapy.get_table(
            table_code=codigo_tabela,
            territorial_level="1",
            ibge_territorial_code="all",
            variable=codigo_variavel,
            period="last 360"
        )
        
        if dados.empty or len(dados) < 2:
            return pd.DataFrame()
        
        df = dados.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['D2C'].str.slice(0, 4)
        
        return processar_dataframe(df)
    except Exception as e:
        logger.error(f"Erro SIDRA {codigo_tabela}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=Config.TTL_MEDIUM)
@log_errors
def carregar_dados_bcb(codigo_serie: str) -> pd.DataFrame:
    """Carrega dados do Banco Central (SGS)"""
    for tentativa in range(Config.MAX_RETRIES):
        try:
            url = f"{Config.BCB_API}/bcdata.sgs.{codigo_serie}/dados?formato=json"
            response = requests.get(
                url, 
                headers=Config.HEADERS, 
                timeout=Config.REQUEST_TIMEOUT,
                verify=False
            )
            response.raise_for_status()
            
            dados = response.json()
            if not dados:
                return pd.DataFrame()
            
            df = pd.DataFrame(dados)
            df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            df['D2C'] = df['data_date'].dt.strftime('%Y%m')
            df['ano'] = df['data_date'].dt.strftime('%Y')
            
            df = processar_dataframe(df)
            
            # Verifica integridade
            alertas = verificar_dados(df, f"BCB_{codigo_serie}")
            if alertas:
                logger.warning(f"Alertas BCB {codigo_serie}: {alertas}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Tentativa {tentativa + 1} falhou para BCB {codigo_serie}: {e}")
            if tentativa < Config.MAX_RETRIES - 1:
                time.sleep(2 ** tentativa)
    
    return pd.DataFrame()

@st.cache_data(ttl=Config.TTL_SHORT)
@log_errors
def carregar_focus() -> pd.DataFrame:
    """Carrega expectativas do Boletim Focus"""
    try:
        url = f"{Config.FOCUS_API}/ExpectativasMercadoAnuais?$top=1000&$orderby=Data%20desc&$format=json"
        response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        dados = response.json()
        df = pd.DataFrame(dados['value'])
        
        indicadores = [
            'IPCA', 'PIB Total', 'Selic', 'Câmbio', 'IGP-M',
            'IPCA Administrados', 'Conta corrente', 'Balança comercial',
            'Investimento direto no país', 'Dívida líquida do setor público',
            'Resultado primário', 'Resultado nominal'
        ]
        
        df = df[df['Indicador'].isin(indicadores)]
        df = df.rename(columns={
            'Data': 'data_relatorio',
            'DataReferencia': 'ano_referencia',
            'Mediana': 'previsao'
        })
        
        df['ano_referencia'] = pd.to_numeric(df['ano_referencia'], errors='coerce')
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'], errors='coerce')
        df['previsao'] = pd.to_numeric(df['previsao'], errors='coerce')
        
        return df.dropna(subset=['previsao', 'ano_referencia'])
    except Exception as e:
        logger.error(f"Erro Focus: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=Config.TTL_SHORT)
@log_errors
def carregar_cotacoes_tempo_real() -> pd.DataFrame:
    """Carrega cotações de moedas em tempo real"""
    try:
        dados = {}
        for ticker in Config.YAHOO_CURRENCIES:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.fast_info
            
            if info and info.get('last_price'):
                preco_atual = info['last_price']
                fechamento_anterior = info.get('previous_close', preco_atual)
                variacao = ((preco_atual - fechamento_anterior) / fechamento_anterior) * 100
                
                moeda = ticker.replace("=X", "")
                dados[moeda] = {
                    'cotacao': preco_atual,
                    'variacao': variacao,
                    'atualizado': datetime.now().strftime('%H:%M')
                }
        
        if dados:
            df = pd.DataFrame.from_dict(dados, orient='index')
            df.index.name = 'moeda'
            return df
        
    except Exception as e:
        logger.error(f"Erro cotações tempo real: {e}")
    
    return pd.DataFrame()

@st.cache_data(ttl=Config.TTL_LONG)
@log_errors
def carregar_historico_cambio() -> pd.DataFrame:
    """Carrega histórico completo de câmbio"""
    try:
        df = yf.download(Config.YAHOO_CURRENCIES, start="1994-07-01", progress=False)
        
        if df.empty:
            return pd.DataFrame()
        
        df = df['Close'].copy()
        
        # Correção de fuso horário
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo')
        df.index = df.index.tz_localize(None)
        
        # Filtra datas futuras (erros do Yahoo)
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        return df.ffill().dropna()
    except Exception as e:
        logger.error(f"Erro histórico câmbio: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=Config.TTL_MEDIUM)
@log_errors
def carregar_macro_real() -> Tuple[Dict, Dict]:
    """Carrega dados macroeconômicos realizados"""
    series_bcb = {
        'PIB': 4382,
        'Dívida Líq.': 4513,
        'Res. Primário': 5793,
        'Res. Nominal': 5811,
        'Balança Com.': 22707,
        'Trans. Correntes': 22724,
        'IDP': 22885
    }
    
    mapa_meses = {
        '01': 'jan', '02': 'fev', '03': 'mar', '04': 'abr',
        '05': 'mai', '06': 'jun', '07': 'jul', '08': 'ago',
        '09': 'set', '10': 'out', '11': 'nov', '12': 'dez'
    }
    
    kpis = {}
    historico = {}
    
    for nome, codigo in series_bcb.items():
        for tentativa in range(Config.MAX_RETRIES):
            try:
                url = f"{Config.BCB_API}/bcdata.sgs.{codigo}/dados/ultimos/60?formato=json"
                response = requests.get(
                    url,
                    headers=Config.HEADERS,
                    timeout=Config.REQUEST_TIMEOUT,
                    verify=False
                )
                response.raise_for_status()
                
                dados = response.json()
                if not dados:
                    continue
                
                df = pd.DataFrame(dados)
                df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                df['data_dt'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
                df = df.dropna()
                
                if df.empty:
                    continue
                
                # Prepara histórico para gráficos
                df_chart = df.copy()
                
                # Ajustes de escala
                if nome == 'PIB':
                    df_chart['valor'] = df_chart['valor'] / 1_000_000  # Para trilhões
                elif nome in ['Balança Com.', 'Trans. Correntes', 'IDP']:
                    df_chart['valor'] = df_chart['valor'] / 1_000  # Para bilhões
                elif 'Primário' in nome or 'Nominal' in nome:
                    df_chart['valor'] = (df_chart['valor'] * -1) / 1_000  # Inverte sinal e escala
                
                historico[nome] = df_chart
                
                # Prepara KPIs
                ultimo = df.iloc[-1]
                ano_atual = ultimo['data_dt'].year
                mes_str = ultimo['data'].split('/')[1]
                data_ref = f"{mapa_meses.get(mes_str, mes_str)}/{str(ano_atual)[2:]}"
                
                # Calcula valor do KPI
                if nome == 'PIB':
                    valor_kpi = ultimo['valor'] / 1_000_000
                elif nome == 'Dívida Líq.':
                    valor_kpi = ultimo['valor']
                else:
                    # Soma acumulada do ano
                    df_ano = df[df['data_dt'].dt.year == ano_atual]
                    soma = df_ano['valor'].sum()
                    if 'Primário' in nome or 'Nominal' in nome:
                        valor_kpi = (soma * -1) / 1_000
                    else:
                        valor_kpi = soma / 1_000
                
                kpis[nome] = {
                    'valor': round(valor_kpi, 2),
                    'data': data_ref
