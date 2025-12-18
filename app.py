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
import urllib3
from pathlib import Path
from functools import wraps
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Any

# --- CONFIGURAÇÃO INICIAL ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Desabilita avisos de SSL inseguro (Necessário para APIs do Gov)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
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
        'neutro': '#607D8B',    # Cinza
        'automatico': '#4CAF50',# Verde para automático
        'manual': '#FF9800'     # Laranja para manual
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

# --- SISTEMA DE STATUS DA FONTE DOS DADOS ---
class StatusFonte:
    """Gerencia o status da fonte dos dados (automático/manual)"""
    
    @staticmethod
    def inicializar():
        """Inicializa o status das fontes no session_state"""
        if 'fontes_status' not in st.session_state:
            st.session_state['fontes_status'] = {
                'focus': {'tipo': None, 'data': None, 'fonte': None},
                'cambio_tempo_real': {'tipo': None, 'data': None, 'fonte': None},
                'cambio_historico': {'tipo': None, 'data': None, 'fonte': None},
                'macro': {'tipo': None, 'data': None, 'fonte': None},
                'indicador_principal': {'tipo': None, 'data': None, 'fonte': None}
            }
    
    @staticmethod
    def atualizar(fonte_nome: str, tipo: str, dados_fonte: str = None):
        """Atualiza o status de uma fonte"""
        StatusFonte.inicializar()
        st.session_state['fontes_status'][fonte_nome] = {
            'tipo': tipo,
            'data': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'fonte': dados_fonte or tipo
        }
    
    @staticmethod
    def obter_status(fonte_nome: str) -> Dict:
        """Obtém o status de uma fonte"""
        StatusFonte.inicializar()
        return st.session_state['fontes_status'].get(fonte_nome, {})
    
    @staticmethod
    def criar_badge(tipo: str, fonte: str = None) -> str:
        """Cria um badge HTML para mostrar o status da fonte"""
        if tipo == 'automático':
            cor = Config.CORES['automatico']
            texto = "🔄 Automático"
            tooltip = f"Fonte: {fonte}" if fonte else "Dados obtidos automaticamente da API"
        elif tipo == 'manual':
            cor = Config.CORES['manual']
            texto = "📝 Manual"
            tooltip = f"Fonte: {fonte}" if fonte else "Dados inseridos manualmente"
        else:
            cor = Config.CORES['neutro']
            texto = "❓ Desconhecido"
            tooltip = "Fonte dos dados não identificada"
        
        return f"""
        <span style="background-color: {cor}20; color: {cor}; 
                     padding: 2px 8px; border-radius: 10px; border: 1px solid {cor}80;
                     font-size: 0.8em; cursor: help;" title="{tooltip}">
            {texto}
        </span>
        """

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
    
    return alertas

def processar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Processamento comum para dataframes de séries temporais"""
    if df.empty:
        return df
    
    # Garante ordenação
    df = df.sort_values('data_date', ascending=True)
    
    # Cria colunas auxiliares
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    
    # Cálculos acumulados
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12, min_periods=1).apply(np.prod, raw=True) - 1) * 100
    
    return df.sort_values('data_date', ascending=False)

# --- FUNÇÕES DE CARGA DE DADOS COM STATUS ---

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
        
        # Atualiza status - sempre automático para SIDRA
        StatusFonte.atualizar('indicador_principal', 'automático', 'IBGE/SIDRA')
        
        return processar_dataframe(df)
    except Exception as e:
        logger.error(f"Erro SIDRA {codigo_tabela}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=Config.TTL_LONG)
def carregar_historico_cambio() -> Tuple[pd.DataFrame, str, str]:
    """Carrega histórico de câmbio (Recriada)"""
    try:
        # Tenta pegar via Yahoo Finance (dados longos)
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="2000-01-01", progress=False)
        if df.empty: raise ValueError("Vazio")
        
        df = df['Close'].copy()
        df.index = df.index.tz_localize(None) # Remove timezone
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        df = df.ffill().dropna()
        
        StatusFonte.atualizar('cambio_historico', 'automático', 'Yahoo Finance')
        return df, 'automático', 'Yahoo Finance'
    except:
        # Fallback simples
        StatusFonte.atualizar('cambio_historico', 'manual', 'Dados Indisponíveis')
        return pd.DataFrame(), 'manual', 'Erro na Carga'

@st.cache_data(ttl=Config.TTL_MEDIUM)
@log_errors
def carregar_dados_bcb(codigo_serie: str) -> pd.DataFrame:
    """Carrega dados do Banco Central (SGS) com tratamento SSL"""
    for tentativa in range(Config.MAX_RETRIES):
        try:
            url = f"{Config.BCB_API}/bcdata.sgs.{codigo_serie}/dados?formato=json"
            response = requests.get(
                url, 
                headers=Config.HEADERS, 
                timeout=Config.REQUEST_TIMEOUT,
                verify=False # Importante para BCB
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
            
            # Atualiza status - sempre automático para BCB
            StatusFonte.atualizar('indicador_principal', 'automático', 'Banco Central (SGS)')
            
            return df
            
        except Exception as e:
            logger.warning(f"Tentativa {tentativa + 1} falhou para BCB {codigo_serie}: {e}")
            if tentativa < Config.MAX_RETRIES - 1:
                time.sleep(2 ** tentativa)
    
    return pd.DataFrame()

@st.cache_data(ttl=Config.TTL_SHORT)
def carregar_focus_otimizado() -> pd.DataFrame:
    """Busca APENAS o último relatório disponível do Focus com segurança"""
    try:
        url_base = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
        
        # Pega apenas a data mais recente
        query_data = "?$top=1&$select=Data&$orderby=Data desc&$format=json"
        
        resp_data = requests.get(
            url_base + query_data, 
            headers=Config.HEADERS, 
            timeout=5, 
            verify=False
        )
        resp_data.raise_for_status()
        ultima_data = resp_data.json()['value'][0]['Data']
        
        # Baixar apenas os dados dessa data específica
        indicadores_str = "'IPCA','Selic','PIB Total','Câmbio','IGP-M'"
        query_final = (
            f"?$filter=Data eq '{ultima_data}' and Indicador in ({indicadores_str})"
            f"&$select=Indicador,DataReferencia,Mediana"
            f"&$format=json"
        )
        
        response = requests.get(
            url_base + query_final, 
            headers=Config.HEADERS, 
            timeout=10, 
            verify=False
        )
        dados = response.json()['value']
        
        df = pd.DataFrame(dados)
        df = df.rename(columns={'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'})
        df['ano_referencia'] = df['ano_referencia'].astype(int)
        
        # Filtra anos futuros relevantes
        ano_atual = datetime.now().year
        df = df[df['ano_referencia'].isin([ano_atual, ano_atual + 1, ano_atual + 2])]
        
        return df
        
    except Exception as e:
        logger.error(f"Erro Focus Otimizado: {e}")
        return pd.DataFrame()

def criar_fallback_focus() -> pd.DataFrame:
    """Cria dataframe com dados manuais do Focus"""
    dados_focus = {
        'IPCA': {2025: 3.80, 2026: 3.50, 2027: 3.50},
        'Selic': {2025: 9.00, 2026: 8.50, 2027: 8.00},
        'Câmbio': {2025: 5.40, 2026: 5.50, 2027: 5.50},
        'PIB Total': {2025: 2.26, 2026: 1.81, 2027: 1.81},
        'IGP-M': {2025: 4.20, 2026: 3.80, 2027: 3.80}
    }
    
    registros = []
    hoje = datetime.now().date()
    
    for indicador, valores in dados_focus.items():
        for ano, valor in valores.items():
            registros.append({
                'Indicador': indicador,
                'ano_referencia': ano,
                'previsao': valor,
                'data_relatorio': pd.Timestamp(hoje)
            })
    return pd.DataFrame(registros)

def carregar_dados_focus() -> Tuple[pd.DataFrame, str, str]:
    """Wrapper que decide entre API Otimizada e Fallback"""
    df_api = carregar_focus_otimizado()
    if not df_api.empty:
        StatusFonte.atualizar('focus', 'automático', 'BCB/Olinda (Otimizado)')
        return df_api, 'automático', 'BCB/Olinda'
    
    df_manual = criar_fallback_focus()
    StatusFonte.atualizar('focus', 'manual', 'Base VPL (Manual)')
    return df_manual, 'manual', 'Base VPL'
        
@st.cache_data(ttl=60)
@log_errors
def carregar_cotacoes_tempo_real() -> Tuple[pd.DataFrame, str, str]:
    """Carrega cotações via AwesomeAPI com Headers Corretos"""
    url = "https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL"
    try:
        response = requests.get(url, headers=Config.HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        dados_formatados = {}
        mapa_nomes = {'USDBRL': 'Dólar', 'EURBRL': 'Euro'}
        
        for par, info in data.items():
            nome = mapa_nomes.get(par, par)
            dados_formatados[nome] = {
                'cotacao': float(info['bid']),
                'variacao': float(info['pctChange']),
                'atualizado': datetime.now().strftime('%H:%M')
            }
            
        df = pd.DataFrame.from_dict(dados_formatados, orient='index')
        StatusFonte.atualizar('cambio_tempo_real', 'automático', 'AwesomeAPI')
        return df, 'automático', 'AwesomeAPI'
        
    except Exception as e:
        logger.error(f"Erro AwesomeAPI: {e}")
        dados_manual = {
            'Dólar': {'cotacao': 5.80, 'variacao': 0.0, 'atualizado': 'Manual'},
            'Euro': {'cotacao': 6.10, 'variacao': 0.0, 'atualizado': 'Manual'}
        }
        return pd.DataFrame.from_dict(dados_manual, orient='index'), 'manual', 'Base VPL'

@st.cache_data(ttl=Config.TTL_MEDIUM)
@log_errors
def carregar_macro_real() -> Tuple[Dict, Dict, str, str]:
    """Carrega dados macroeconômicos realizados"""
    series_bcb = {
        'PIB': 4382, 'Dívida Líq.': 4513, 'Res. Primário': 5793,
        'Res. Nominal': 5811, 'Balança Com.': 22707,
        'Trans. Correntes': 22724, 'IDP': 22885
    }
    
    mapa_meses = {'01': 'jan', '02': 'fev', '03': 'mar', '04': 'abr', '05': 'mai', '06': 'jun', '07': 'jul', '08': 'ago', '09': 'set', '10': 'out', '11': 'nov', '12': 'dez'}
    kpis = {}
    historico = {}
    dados_validos = 0
    
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
                if not dados: continue
                
                df = pd.DataFrame(dados)
                df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                df['data_dt'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
                df = df.dropna()
                if df.empty: continue
                
                # Prepara histórico
                df_chart = df.copy()
                if nome == 'PIB': df_chart['valor'] /= 1_000_000
                elif nome in ['Balança Com.', 'Trans. Correntes', 'IDP']: df_chart['valor'] /= 1_000
                elif 'Primário' in nome or 'Nominal' in nome: df_chart['valor'] = (df_chart['valor'] * -1) / 1_000
                historico[nome] = df_chart
                
                # Prepara KPIs
                ultimo = df.iloc[-1]
                ano_atual = ultimo['data_dt'].year
                mes_str = ultimo['data'].split('/')[1]
                data_ref = f"{mapa_meses.get(mes_str, mes_str)}/{str(ano_atual)[2:]}"
                
                if nome == 'PIB': valor_kpi = ultimo['valor'] / 1_000_000
                elif nome == 'Dívida Líq.': valor_kpi = ultimo['valor']
                else:
                    df_ano = df[df['data_dt'].dt.year == ano_atual]
                    soma = df_ano['valor'].sum()
                    if 'Primário' in nome or 'Nominal' in nome: valor_kpi = (soma * -1) / 1_000
                    else: valor_kpi = soma / 1_000
                
                kpis[nome] = {'valor': round(valor_kpi, 2), 'data': data_ref, 'ano': ano_atual}
                dados_validos += 1
                break
                
            except Exception as e:
                if tentativa < Config.MAX_RETRIES - 1: time.sleep(2 ** tentativa)
                continue
    
    if dados_validos >= 5:
        StatusFonte.atualizar('macro', 'automático', 'Banco Central (SGS)')
        return kpis, historico, 'automático', 'Banco Central (SGS)'
    else:
        StatusFonte.atualizar('macro', 'manual', 'Base VPL Consultoria')
        return kpis, historico, 'manual', 'Base VPL Consultoria'

# --- FUNÇÕES DE CÁLCULO ---
def calcular_correcao(df: pd.DataFrame, valor: float, data_ini: str, data_fim: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Calcula correção monetária entre duas datas"""
    if df.empty:
        return None, "Dados insuficientes"
    
    # Determina direção do cálculo
    reverso = data_ini > data_fim
    if reverso:
        inicio, fim = data_fim, data_ini
    else:
        inicio, fim = data_ini, data_fim
    
    # Filtra período
    mask = (df['D2C'] >= inicio) & (df['D2C'] <= fim)
    df_periodo = df.loc[mask].copy()
    
    if df_periodo.empty or len(df_periodo) < 2:
        return None, "Período sem dados suficientes"
    
    # Calcula fator acumulado
    fator_acumulado = df_periodo['fator'].prod()
    
    # Calcula valor final
    if reverso:
        valor_final = valor / fator_acumulado
    else:
        valor_final = valor * fator_acumulado
    
    pct_total = (fator_acumulado - 1) * 100
    
    return {
        'valor_final': valor_final,
        'percentual': pct_total,
        'fator': fator_acumulado,
        'is_reverso': reverso,
        'periodo_dias': len(df_periodo),
        'taxa_mensal_media': (fator_acumulado ** (1/len(df_periodo)) - 1) * 100
    }, None

# --- COMPONENTES DE UI ---
def criar_header():
    """Cria header da aplicação com indicadores de fonte"""
    StatusFonte.inicializar()
    fontes_status = st.session_state.get('fontes_status', {})
    
    contador = {'automático': 0, 'manual': 0, 'total': 0}
    for status in fontes_status.values():
        if status.get('tipo'):
            contador[status['tipo']] += 1
            contador['total'] += 1
    
    percentual_automatico = (contador['automático'] / contador['total'] * 100) if contador['total'] > 0 else 0
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #003366 0%, #0066cc 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
            <div>
                <h1 style='color: white; margin: 0;'>VPL CONSULTORIA</h1>
                <p style='color: #E0E0E0; margin: 5px 0 0 0; font-size: 1.1em;'>
                    📊 Inteligência Macroeconômica & Correção Monetária
                </p>
            </div>
            <div style='text-align: right;'>
                <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;'>
                    <div style='color: #4CAF50; font-size: 0.9em;'>
                        <span style='font-weight: bold;'>📡 Status Fontes:</span>
                        <span style='margin-left: 10px;'>{contador['automático']} auto</span>
                        <span style='margin-left: 5px;'>{contador['manual']} manual</span>
                    </div>
                    <div style='color: #CCCCCC; font-size: 0.9em; margin-top: 5px;'>
                        {percentual_automatico:.0f}% automático • {datetime.now().strftime("%d/%m/%Y %H:%M")}
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def criar_sidebar():
    """Cria sidebar com configurações"""
    try:
        st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
    except:
        st.sidebar.markdown("<h3 style='text-align:center'>VPL CONSULTORIA</h3>", unsafe_allow_html=True)
    
    st.sidebar.header("🔧 Configurações")
    indicadores = ["IPCA (Inflação Oficial)", "INPC (Salários)", "IGP-M (Aluguéis)", "SELIC (Taxa Básica)", "CDI (Investimentos)"]
    tipo_indice = st.sidebar.selectbox("Selecione o Indicador", indicadores)
    
    st.sidebar.divider()
    st.sidebar.subheader("🧮 Calculadora")
    valor_input = st.sidebar.number_input("Valor (R$)", value=1000.00, step=100.00, format="%.2f")
    return tipo_indice, valor_input

def criar_painel_focus(df_focus: pd.DataFrame, tipo_focus: str, fonte_focus: str, cotacoes: pd.DataFrame, tipo_cambio: str, fonte_cambio: str):
    """Cria painel de expectativas de mercado"""
    with st.expander("🔭 Expectativas de Mercado - Boletim Focus", expanded=False):
        col_titulo, col_badge = st.columns([3, 1])
        with col_titulo: st.markdown("### 📊 Projeções Macroeconômicas")
        with col_badge: st.markdown(StatusFonte.criar_badge(tipo_focus, fonte_focus), unsafe_allow_html=True)
        
        if df_focus.empty:
            st.warning("⚠️ Dados do Focus indisponíveis")
            return
            
        ano_atual = datetime.now().year
        st.markdown("##### 🎯 Indicadores Chave")
        
        df_atual = df_focus[df_focus['ano_referencia'] == ano_atual].copy()
        if not df_atual.empty:
            pivot_atual = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='mean')
            col1, col2, col3, col4 = st.columns(4)
            if 'IPCA' in pivot_atual.index: col1.metric("📈 IPCA 2025", f"{pivot_atual.loc['IPCA', 'previsao']:.2f}%")
            if 'Selic' in pivot_atual.index: col2.metric("🏦 Selic 2025", f"{pivot_atual.loc['Selic', 'previsao']:.2f}%")
            if 'PIB Total' in pivot_atual.index: col3.metric("📊 PIB 2025", f"{pivot_atual.loc['PIB Total', 'previsao']:.2f}%")
            if 'Câmbio' in pivot_atual.index: col4.metric("💵 Dólar 2025", f"R$ {pivot_atual.loc['Câmbio', 'previsao']:.2f}")

        st.divider()
        col_dados, col_cambio = st.columns([3, 1])
        
        with col_dados:
            st.markdown("##### 📅 Projeções Anuais (2025-2027)")
            df_tabela = df_focus.copy()
            if not df_tabela.empty:
                pivot_completo = df_tabela.pivot_table(index='Indicador', columns='ano_referencia', values='previsao', aggfunc='mean')
                st.dataframe(pivot_completo.style.format("{:.2f}"), use_container_width=True, height=500)
        
        with col_cambio:
            col_t, col_b = st.columns([2, 1])
            with col_t: st.markdown("##### 💱 Câmbio Agora")
            with col_b: st.markdown(StatusFonte.criar_badge(tipo_cambio, fonte_cambio), unsafe_allow_html=True)
            
            if not cotacoes.empty:
                for moeda, dados in cotacoes.iterrows():
                    cor = "#4CAF50" if dados['variacao'] >= 0 else "#F44336"
                    st.markdown(f"""
                    <div style='background: rgba(0,0,0,0.05); padding: 10px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid {cor};'>
                        <div style='font-size: 0.9em; font-weight: bold;'>{moeda}</div>
                        <div style='font-size: 1.5em; font-weight: bold;'>R$ {dados['cotacao']:.2f}</div>
                        <div style='color: {cor}; font-size: 0.8em;'>{dados['variacao']:+.2f}%</div>
                    </div>""", unsafe_allow_html=True)
            
            if st.button("🔄 Atualizar", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

def criar_painel_macro(kpis: Dict, historico: Dict, tipo_macro: str, fonte_macro: str):
    """Cria painel de conjuntura macroeconômica"""
    with st.expander("🧩 Conjuntura Macroeconômica (Dados Oficiais)", expanded=False):
        col_titulo, col_badge = st.columns([3, 1])
        with col_titulo: st.markdown("### 📈 Dados Macroeconômicos Realizados")
        with col_badge: st.markdown(StatusFonte.criar_badge(tipo_macro, fonte_macro), unsafe_allow_html=True)
        
        if not kpis:
            st.warning("Dados macroeconômicos indisponíveis.")
            return

        st.markdown("##### 🏛️ Atividade & Fiscal")
        col1, col2, col3, col4 = st.columns(4)
        def kpi(k): return kpis.get(k, {'valor': 0, 'data': '-'})
        
        col1.metric(f"PIB 12m ({kpi('PIB')['data']})", f"R$ {kpi('PIB')['valor']:.2f} Tri")
        col2.metric(f"Dívida Líq. ({kpi('Dívida Líq.')['data']})", f"{kpi('Dívida Líq.')['valor']:.1f}% PIB")
        col3.metric("Primário (YTD)", f"R$ {kpi('Res. Primário')['valor']:.1f} Bi")
        col4.metric("Nominal (YTD)", f"R$ {kpi('Res. Nominal')['valor']:.1f} Bi")
        
        st.divider()
        st.markdown("##### 📊 Tendências")
        tab1, tab2 = st.tabs(["Atividade", "Externo"])
        with tab1:
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(historico.get('PIB'), x='data_dt', y='valor', title='PIB'), use_container_width=True)
            c2.plotly_chart(px.line(historico.get('Dívida Líq.'), x='data_dt', y='valor', title='Dívida'), use_container_width=True)

def criar_painel_principal(df: pd.DataFrame, tipo_indice: str, cor_tema: str):
    """Cria painel principal"""
    if df.empty: return
    
    status = StatusFonte.obter_status('indicador_principal')
    st.title(f"📊 Painel: {tipo_indice.split()[0]}")
    st.markdown(StatusFonte.criar_badge(status.get('tipo'), status.get('fonte')), unsafe_allow_html=True)
    
    ultimo = df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Taxa do Mês", f"{ultimo['valor']:.2f}%")
    c2.metric("Acumulado 12M", f"{ultimo['acum_12m']:.2f}%")
    c3.metric("Acumulado Ano", f"{ultimo['acum_ano']:.2f}%")
    
    tab1, tab2 = st.tabs(["Gráfico", "Tabela"])
    with tab1:
        fig = px.line(df, x='data_date', y='acum_12m', title=f"Acumulado 12M - {tipo_indice.split()[0]}")
        fig.update_traces(line_color=cor_tema)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.dataframe(df[['data_fmt', 'valor', 'acum_12m']].head(24), use_container_width=True)

def criar_calculadora_sidebar(df: pd.DataFrame, tipo_indice: str, valor_input: float):
    if df.empty: return
    anos = sorted(df['ano'].unique(), reverse=True)
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    mapa_meses = {m: f"{i+1:02d}" for i, m in enumerate(meses)}
    
    st.sidebar.markdown("**📅 Período**")
    c1, c2 = st.sidebar.columns(2)
    mes_ini = c1.selectbox("Mês Ini", meses, key='mi')
    ano_ini = c2.selectbox("Ano Ini", anos, key='ai')
    
    c3, c4 = st.sidebar.columns(2)
    mes_fim = c3.selectbox("Mês Fim", meses, index=len(meses)-1, key='mf')
    ano_fim = c4.selectbox("Ano Fim", anos, key='af')
    
    if st.sidebar.button("🚀 Calcular", type="primary"):
        code_ini, code_fim = f"{ano_ini}{mapa_meses[mes_ini]}", f"{ano_fim}{mapa_meses[mes_fim]}"
        res, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
        
        if erro: st.error(erro)
        else:
            st.sidebar.success(f"R$ {res['valor_final']:,.2f}")
            st.sidebar.caption(f"Correção total: {res['percentual']:.2f}%")

def criar_footer():
    st.markdown("---")
    st.caption("VPL Consultoria • Dados oficiais")

# --- FUNÇÃO PRINCIPAL ---
def main():
    criar_header()
    with st.sidebar:
        tipo_indice, valor_input = criar_sidebar()
    
    with st.spinner("Conectando às APIs..."):
        # Mapa de carregadores
        mapa = {
            "IPCA": lambda: carregar_dados_sidra("1737", "63"),
            "INPC": lambda: carregar_dados_sidra("1736", "44"),
            "IGP-M": lambda: carregar_dados_bcb("189"),
            "SELIC": lambda: carregar_dados_bcb("4390"),
            "CDI": lambda: carregar_dados_bcb("4391")
        }
        
        # Execução
        nome = tipo_indice.split()[0]
        df_ind = mapa.get(nome)() if mapa.get(nome) else pd.DataFrame()
        cor = Config.CORES.get(nome, '#333')
        
        df_focus, t_focus, f_focus = carregar_dados_focus()
        df_cot, t_cot, f_cot = carregar_cotacoes_tempo_real()
        kpis, hist, t_macro, f_macro = carregar_macro_real()
        df_cambio, t_cambio, f_cambio = carregar_historico_cambio()
    
    if df_ind.empty: st.error("Erro ao carregar indicador principal.")
    
    criar_calculadora_sidebar(df_ind, tipo_indice, valor_input)
    criar_painel_focus(df_focus, t_focus, f_focus, df_cot, t_cot, f_cot)
    criar_painel_macro(kpis, hist, t_macro, f_macro)
    criar_painel_cambio(df_cambio, t_cambio, f_cambio)
    criar_painel_principal(df_ind, tipo_indice, cor)
    criar_footer()

    # CSS
    st.markdown("""
    <style>
        .stBadge { background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
