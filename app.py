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
import json
import os

class GerenciadorDados:
    """Gerencia o cache local em disco (Lazy ETL)"""
    ARQUIVO_CACHE = "vpl_dados_cache.json"
    TTL_CACHE_DISCO = 12 * 3600  # 12 horas em segundos

    @staticmethod
    def carregar_do_disco():
        """Tenta carregar dados do arquivo local"""
        if not os.path.exists(GerenciadorDados.ARQUIVO_CACHE):
            return None
        
        try:
            # Verifica se o arquivo é recente
            tempo_modificacao = os.path.getmtime(GerenciadorDados.ARQUIVO_CACHE)
            idade_arquivo = time.time() - tempo_modificacao
            
            if idade_arquivo > GerenciadorDados.TTL_CACHE_DISCO:
                logger.info("Cache em disco expirado.")
                return None
                
            logger.info("Carregando dados do disco (Cache Local)...")
            with open(GerenciadorDados.ARQUIVO_CACHE, 'r') as f:
                dados_serializados = json.load(f)
                return dados_serializados
        except Exception as e:
            logger.error(f"Erro ao ler cache: {e}")
            return None

    @staticmethod
    def salvar_no_disco(dados_dict):
        """Salva os dados processados no disco para o próximo uso"""
        try:
            # Precisamos converter DataFrames para JSON/Dict antes de salvar
            # Esta é uma simplificação; idealmente serializamos cada objeto
            with open(GerenciadorDados.ARQUIVO_CACHE, 'w') as f:
                json.dump(dados_dict, f)
            logger.info("Dados salvos no cache local com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")

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
    
    return alertas

# --- FUNÇÕES DE CARGA DE DADOS COM STATUS ---
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

def carregar_dados_focus() -> Tuple[pd.DataFrame, str, str]:
    """Wrapper que decide entre API Otimizada e Fallback"""
    # 1. Tenta a API Otimizada
    df_api = carregar_focus_otimizado()
    
    if not df_api.empty:
        StatusFonte.atualizar('focus', 'automático', 'BCB/Olinda (Otimizado)')
        return df_api, 'automático', 'BCB/Olinda'
    
    # 2. Se falhar, usa o fallback manual
    df_manual = criar_fallback_focus()
    StatusFonte.atualizar('focus', 'manual', 'Base VPL (Manual)')
    return df_manual, 'manual', 'Base VPL'

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
            
            # Atualiza status - sempre automático para BCB
            StatusFonte.atualizar('indicador_principal', 'automático', 'Banco Central (SGS)')
            
            return df
            
        except Exception as e:
            logger.warning(f"Tentativa {tentativa + 1} falhou para BCB {codigo_serie}: {e}")
            if tentativa < Config.MAX_RETRIES - 1:
                time.sleep(2 ** tentativa)
    
    return pd.DataFrame()

# --- SISTEMA DE FOCUS COM FALLBACK E STATUS ---
@st.cache_data(ttl=Config.TTL_SHORT)
@log_errors
def carregar_focus_api() -> pd.DataFrame:
    """Tenta carregar dados do Focus via API oficial"""
    try:
        # URL para projeções anuais do Focus
        url = f"{Config.FOCUS_API}/ExpectativasMercadoAnuais"
        
        params = {
            '$top': 1000,
            '$format': 'json',
            '$orderby': 'Data desc',
            '$select': 'Indicador,Data,DataReferencia,Mediana'
        }
        
        response = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        dados = response.json()
        df = pd.DataFrame(dados['value'])
        
        if df.empty:
            return pd.DataFrame()
        
        # Filtra indicadores relevantes
        indicadores_relevantes = [
            'IPCA', 'IPCA Administrados', 'IGP-M', 'Selic', 'Câmbio',
            'PIB Total', 'Balança comercial', 'Conta corrente',
            'Investimento direto no país', 'Dívida líquida do setor público',
            'Resultado primário', 'Resultado nominal'
        ]
        
        df = df[df['Indicador'].isin(indicadores_relevantes)].copy()
        
        # Renomeia e converte
        df = df.rename(columns={
            'Data': 'data_relatorio',
            'DataReferencia': 'ano_referencia',
            'Mediana': 'previsao'
        })
        
        df['ano_referencia'] = pd.to_numeric(df['ano_referencia'], errors='coerce')
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'], errors='coerce')
        df['previsao'] = pd.to_numeric(df['previsao'], errors='coerce')
        
        # Remove duplicatas
        df = df.sort_values('data_relatorio', ascending=False)
        df = df.drop_duplicates(subset=['Indicador', 'ano_referencia'], keep='first')
        
        # Filtra anos recentes
        ano_atual = datetime.now().year
        anos_validos = list(range(ano_atual, ano_atual + 4))
        df = df[df['ano_referencia'].isin(anos_validos)]
        
        logger.info(f"Focus API carregado: {len(df)} registros")
        
        return df
        
    except Exception as e:
        logger.error(f"Erro API Focus: {e}")
        return pd.DataFrame()

def criar_fallback_focus() -> pd.DataFrame:
    """Cria dataframe com dados manuais atualizados do Focus"""
    
    # Dados atualizados baseados na imagem fornecida
    dados_focus = {
        'IPCA': {2025: 3.80, 2026: 3.50, 2027: 3.50},
        'IPCA Administrados': {2025: 5.32, 2026: 3.75, 2027: 3.61},
        'IGP-M': {2025: 4.20, 2026: 3.80, 2027: 3.80},
        'Selic': {2025: 9.00, 2026: 8.50, 2027: 8.00},
        'Câmbio': {2025: 5.40, 2026: 5.50, 2027: 5.50},
        'PIB Total': {2025: 2.26, 2026: 1.81, 2027: 1.81},
        'Dívida líquida do setor público': {2025: 65.97, 2026: 70.13, 2027: 73.70},
        'Resultado primário': {2025: -0.48, 2026: -0.57, 2027: -0.40},
        'Resultado nominal': {2025: -8.35, 2026: -8.65, 2027: -7.92},
        'Balança comercial': {2025: 229.7, 2026: 234.2, 2027: 241.9},
        'Conta corrente': {2025: -72.7, 2026: -66.0, 2027: -65.0},
        'Investimento direto no país': {2025: 75.0, 2026: 73.5, 2027: 77.5}
    }
    
    # Converte para DataFrame
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
    
    df = pd.DataFrame(registros)
    
    return df

@st.cache_data(ttl=Config.TTL_SHORT)
def carregar_focus_otimizado() -> pd.DataFrame:
    """Busca APENAS o último relatório disponível do Focus"""
    try:
        # 1. Descobrir qual a data do último relatório disponível
        url_base = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
        
        # Pega apenas a data mais recente (top=1 ordenado por data desc)
        query_data = "?$top=1&$select=Data&$orderby=Data desc&$format=json"
        resp_data = requests.get(url_base + query_data, timeout=5)
        ultima_data = resp_data.json()['value'][0]['Data']
        
        # 2. Baixar apenas os dados dessa data específica
        # Filtra pela data E pelos indicadores que queremos para economizar banda
        indicadores_str = "'IPCA','Selic','PIB Total','Câmbio','IGP-M'" # Adicione os outros aqui
        query_final = (
            f"?$filter=Data eq '{ultima_data}' and Indicador in ({indicadores_str})"
            f"&$select=Indicador,DataReferencia,Mediana"
            f"&$format=json"
        )
        
        response = requests.get(url_base + query_final, timeout=10)
        dados = response.json()['value']
        
        df = pd.DataFrame(dados)
        
        # Tratamento (similar ao seu, mas agora o DF é leve)
        df = df.rename(columns={'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'})
        df['ano_referencia'] = df['ano_referencia'].astype(int)
        
        # Filtra anos futuros relevantes (ex: ano atual + 2)
        ano_atual = datetime.now().year
        df = df[df['ano_referencia'].isin([ano_atual, ano_atual + 1, ano_atual + 2])]
        
        return df
        
    except Exception as e:
        logger.error(f"Erro Focus Otimizado: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=60)
@log_errors
def carregar_cotacoes_tempo_real() -> Tuple[pd.DataFrame, str, str]:
    """Carrega cotações via AwesomeAPI (Muito mais rápido que Yahoo)"""
    url = "https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL"
    try:
        response = requests.get(url, timeout=5)
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
        # Fallback manual se a API falhar
        dados_manual = {
            'Dólar': {'cotacao': 5.80, 'variacao': 0.0, 'atualizado': 'Manual'},
            'Euro': {'cotacao': 6.10, 'variacao': 0.0, 'atualizado': 'Manual'}
        }
        return pd.DataFrame.from_dict(dados_manual, orient='index'), 'manual', 'Base VPL'
        
@st.cache_data(ttl=Config.TTL_LONG)
def get_sgs_data(codigos: dict, data_inicio: str = '01/01/2000') -> pd.DataFrame:
    """
    Busca múltiplas séries do BCB de uma vez.
    Ex: codigos = {'Selic': 432, 'IPCA': 433}
    """
    df_final = pd.DataFrame()
    
    for nome, codigo in codigos.items():
        url = f'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json&dataInicial={data_inicio}'
        try:
            df = pd.read_json(url)
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df = df.set_index('data')
            df = df.rename(columns={'valor': nome})
            
            if df_final.empty:
                df_final = df
            else:
                df_final = df_final.join(df, how='outer')
        except Exception as e:
            logger.error(f"Erro ao baixar série {codigo}: {e}")
            
    return df_final

@st.cache_data(ttl=Config.TTL_MEDIUM)
@log_errors
def carregar_macro_real() -> Tuple[Dict, Dict, str, str]:
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
                    df_chart['valor'] = df_chart['valor'] / 1_000_000
                elif nome in ['Balança Com.', 'Trans. Correntes', 'IDP']:
                    df_chart['valor'] = df_chart['valor'] / 1_000
                elif 'Primário' in nome or 'Nominal' in nome:
                    df_chart['valor'] = (df_chart['valor'] * -1) / 1_000
                
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
                    df_ano = df[df['data_dt'].dt.year == ano_atual]
                    soma = df_ano['valor'].sum()
                    if 'Primário' in nome or 'Nominal' in nome:
                        valor_kpi = (soma * -1) / 1_000
                    else:
                        valor_kpi = soma / 1_000
                
                kpis[nome] = {
                    'valor': round(valor_kpi, 2),
                    'data': data_ref,
                    'ano': ano_atual
                }
                
                dados_validos += 1
                break
                
            except Exception as e:
                logger.warning(f"Tentativa {tentativa + 1} para {nome} falhou: {e}")
                if tentativa < Config.MAX_RETRIES - 1:
                    time.sleep(2 ** tentativa)
                continue
    
    # Determina fonte baseado no sucesso
    if dados_validos >= 5:  # Se obteve pelo menos 5 de 7 indicadores
        StatusFonte.atualizar('macro', 'automático', 'Banco Central (SGS)')
        return kpis, historico, 'automático', 'Banco Central (SGS)'
    else:
        logger.warning(f"Apenas {dados_validos}/7 indicadores macro carregados")
        StatusFonte.atualizar('macro', 'manual', 'Base VPL Consultoria')
        return kpis, historico, 'manual', 'Base VPL Consultoria'

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

# --- COMPONENTES DE UI COM INDICADORES DE FONTE ---
def criar_header():
    """Cria header da aplicação com indicadores de fonte"""
    # Inicializa sistema de status
    StatusFonte.inicializar()
    
    # Obtém status de todas as fontes
    fontes_status = st.session_state.get('fontes_status', {})
    
    # Conta fontes automáticas vs manuais
    contador = {'automático': 0, 'manual': 0, 'total': 0}
    for status in fontes_status.values():
        if status.get('tipo'):
            contador[status['tipo']] += 1
            contador['total'] += 1
    
    # Calcula porcentagem de dados automáticos
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
    """Cria sidebar com configurações e calculadora"""
    # Logo
    try:
        st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
    except:
        st.sidebar.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <h3 style='color: #003366;'>VPL CONSULTORIA</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.header("🔧 Configurações")
    
    # Seletor de indicador
    indicadores = [
        "IPCA (Inflação Oficial)",
        "INPC (Salários)",
        "IGP-M (Aluguéis)",
        "SELIC (Taxa Básica)",
        "CDI (Investimentos)"
    ]
    
    tipo_indice = st.sidebar.selectbox("Selecione o Indicador", indicadores)
    
    # Calculadora
    st.sidebar.divider()
    st.sidebar.subheader("🧮 Calculadora")
    
    valor_input = st.sidebar.number_input(
        "Valor (R$)",
        value=1000.00,
        step=100.00,
        format="%.2f",
        help="Valor a ser corrigido"
    )
    
    return tipo_indice, valor_input

def criar_painel_focus(df_focus: pd.DataFrame, tipo_focus: str, fonte_focus: str, cotacoes: pd.DataFrame, tipo_cambio: str, fonte_cambio: str):
    """Cria painel de expectativas de mercado com indicador de fonte"""
    
    with st.expander("🔭 Expectativas de Mercado - Boletim Focus", expanded=False):
        
        # Header com badge de fonte
        col_titulo, col_badge = st.columns([3, 1])
        
        with col_titulo:
            st.markdown("### 📊 Projeções Macroeconômicas")
        
        with col_badge:
            # Badge indicando fonte dos dados
            badge_focus = StatusFonte.criar_badge(tipo_focus, fonte_focus)
            st.markdown(badge_focus, unsafe_allow_html=True)
            st.caption(f"Atualizado: {datetime.now().strftime('%d/%m/%Y')}")
        
        # Verifica se temos dados
        if df_focus.empty:
            st.warning("⚠️ Dados do Focus temporariamente indisponíveis")
            return
        
        # Ano atual para referência
        ano_atual = datetime.now().year
        
        # KPIs Principais em cards
        st.markdown("##### 🎯 Indicadores Chave")
        
        # Filtra ano atual e prepara pivot
        df_atual = df_focus[df_focus['ano_referencia'] == ano_atual].copy()
        if not df_atual.empty:
            pivot_atual = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='mean')
            
            # 4 KPIs principais
            col1, col2, col3, col4 = st.columns(4)
            
            # IPCA
            if 'IPCA' in pivot_atual.index:
                valor_ipca = pivot_atual.loc['IPCA', 'previsao']
                col1.metric("📈 IPCA 2025", f"{valor_ipca:.2f}%")
            
            # Selic
            if 'Selic' in pivot_atual.index:
                valor_selic = pivot_atual.loc['Selic', 'previsao']
                col2.metric("🏦 Selic 2025", f"{valor_selic:.2f}%")
            
            # PIB
            if 'PIB Total' in pivot_atual.index:
                valor_pib = pivot_atual.loc['PIB Total', 'previsao']
                col3.metric("📊 PIB 2025", f"{valor_pib:.2f}%")
            
            # Câmbio
            if 'Câmbio' in pivot_atual.index:
                valor_cambio = pivot_atual.loc['Câmbio', 'previsao']
                col4.metric("💵 Dólar 2025", f"R$ {valor_cambio:.2f}")
        
        st.divider()
        
        # Layout principal
        col_dados, col_cambio = st.columns([3, 1])
        
        with col_dados:
            # Tabela completa de projeções
            st.markdown("##### 📅 Projeções Anuais (2025-2027)")
            
            # Pivot table com anos como colunas
            anos_exibir = [ano_atual, ano_atual + 1, ano_atual + 2]
            df_tabela = df_focus[df_focus['ano_referencia'].isin(anos_exibir)].copy()
            
            if not df_tabela.empty:
                pivot_completo = df_tabela.pivot_table(
                    index='Indicador', 
                    columns='ano_referencia', 
                    values='previsao',
                    aggfunc='mean'
                )
                
                # Ordena os indicadores de forma lógica
                ordem_indicadores = [
                    'IPCA',
                    'IPCA Administrados',
                    'IGP-M',
                    'Selic',
                    'Câmbio',
                    'PIB Total',
                    'Dívida líquida do setor público',
                    'Resultado primário',
                    'Resultado nominal',
                    'Balança comercial',
                    'Conta corrente',
                    'Investimento direto no país'
                ]
                
                # Filtra apenas indicadores presentes
                ordem_filtrada = [i for i in ordem_indicadores if i in pivot_completo.index]
                pivot_completo = pivot_completo.reindex(ordem_filtrada)
                
                # Formata os valores
                df_formatado = pivot_completo.copy()
                
                for col in df_formatado.columns:
                    df_formatado[col] = df_formatado.apply(
                        lambda row: formatar_valor_focus(row.name, row[col]),
                        axis=1
                    )
                
                # Exibe a tabela com formatação condicional
                st.dataframe(
                    df_formatado,
                    use_container_width=True,
                    height=500
                )
                
                # Legenda com fonte
                fonte_texto = f"Dados obtidos via {fonte_focus}" if tipo_focus == 'automático' else f"Dados de referência ({fonte_focus})"
                st.caption(f"""
                **Legenda:** IPCA/IGP-M/Selic/PIB/Dívida/Primário/Nominal = % | 
                Câmbio = R$/US$ | Balança/Conta/IDP = US$ Bilhões
                **Fonte:** {fonte_texto}
                """)
            else:
                st.info("Sem dados de projeções para os próximos anos")
        
        with col_cambio:
            # Painel de câmbio em tempo real
            col_cambio_titulo, col_cambio_badge = st.columns([2, 1])
            with col_cambio_titulo:
                st.markdown("##### 💱 Câmbio Agora")
            with col_cambio_badge:
                badge_cambio = StatusFonte.criar_badge(tipo_cambio, fonte_cambio)
                st.markdown(badge_cambio, unsafe_allow_html=True)
            
            if not cotacoes.empty:
                # Layout para moedas
                for moeda, dados in cotacoes.iterrows():
                    with st.container():
                        variacao = dados['variacao']
                        cor_valor = "#4CAF50" if variacao >= 0 else "#F44336"
                        emoji = "🟢" if variacao >= 0 else "🔴"
                        
                        st.markdown(f"""
                        <div style='background: rgba(0,0,0,0.05); padding: 15px; 
                                    border-radius: 10px; margin-bottom: 10px; border-left: 4px solid {cor_valor};'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div style='font-size: 1.1em; font-weight: bold;'>{moeda}</div>
                                <div style='font-size: 0.8em; color: #666;'>
                                    {dados['atualizado']}
                                </div>
                            </div>
                            <div style='font-size: 1.8em; font-weight: bold; margin: 5px 0;'>
                                R$ {dados['cotacao']:.2f}
                            </div>
                            <div style='color: {cor_valor}; font-weight: bold;'>
                                {emoji} {variacao:+.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Atualizando cotações...")
            
            # Botão para atualizar
            if st.button("🔄 Atualizar Cotações", use_container_width=True, key="btn_atualizar_cambio"):
                st.cache_data.clear()
                st.rerun()

def formatar_valor_focus(indicador: str, valor: float) -> str:
    """Formata valores do Focus conforme o indicador"""
    if pd.isna(valor):
        return "-"
    
    if 'Câmbio' in indicador:
        return f"R$ {valor:.2f}"
    elif any(x in indicador for x in ['comercial', 'Conta corrente', 'Investimento']):
        return f"US$ {valor:.1f} B"
    else:
        return f"{valor:.2f}%"

def criar_painel_macro(kpis: Dict, historico: Dict, tipo_macro: str, fonte_macro: str):
    """Cria painel de conjuntura macroeconômica com indicador de fonte"""
    with st.expander("🧩 Conjuntura Macroeconômica (Dados Oficiais)", expanded=False):
        
        # Header com badge de fonte
        col_titulo, col_badge = st.columns([3, 1])
        with col_titulo:
            st.markdown("### 📈 Dados Macroeconômicos Realizados")
        with col_badge:
            badge_macro = StatusFonte.criar_badge(tipo_macro, fonte_macro)
            st.markdown(badge_macro, unsafe_allow_html=True)
        
        if not kpis:
            st.warning("Não foi possível carregar os dados macroeconômicos.")
            return
        
        # KPIs - Atividade & Fiscal
        st.markdown("##### 🏛️ Atividade & Fiscal")
        col1, col2, col3, col4 = st.columns(4)
        
        def get_kpi(chave):
            return kpis.get(chave, {'valor': 0, 'data': '-', 'ano': '-'})
        
        with col1:
            kpi = get_kpi('PIB')
            col1.metric(f"PIB 12m ({kpi['data']})", f"R$ {kpi['valor']:.2f} Tri")
        
        with col2:
            kpi = get_kpi('Dívida Líq.')
            col2.metric(f"Dívida Líq. ({kpi['data']})", f"{kpi['valor']:.1f}% PIB")
        
        with col3:
            kpi = get_kpi('Res. Primário')
            col3.metric(f"Primário (YTD {kpi['ano']})", f"R$ {kpi['valor']:.1f} Bi")
        
        with col4:
            kpi = get_kpi('Res. Nominal')
            col4.metric(f"Nominal (YTD {kpi['ano']})", f"R$ {kpi['valor']:.1f} Bi")
        
        st.divider()
        
        # KPIs - Setor Externo
        st.markdown("##### 🚢 Setor Externo")
        col5, col6, col7 = st.columns(3)
        
        with col5:
            kpi = get_kpi('Balança Com.')
            col5.metric(f"Balança Com. ({kpi['data']})", f"US$ {kpi['valor']:.1f} Bi")
        
        with col6:
            kpi = get_kpi('Trans. Correntes')
            col6.metric(f"Trans. Correntes ({kpi['data']})", f"US$ {kpi['valor']:.1f} Bi")
        
        with col7:
            kpi = get_kpi('IDP')
            col7.metric(f"IDP ({kpi['data']})", f"US$ {kpi['valor']:.1f} Bi")
        
        st.divider()
        
        # Gráficos
        st.markdown("##### 📊 Tendências (Últimos 5 Anos)")
        
        # Adiciona badge de fonte para gráficos
        st.markdown(f"*Fonte dos dados: {fonte_macro}*", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Atividade", "Fiscal", "Externo"])
        
        with tab1:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                plotar_serie_temporal(historico.get('PIB'), 'PIB', '#00BFFF')
            with col_a2:
                plotar_serie_temporal(historico.get('Dívida Líq.'), 'Dívida Líquida', '#FFD700')
        
        with tab2:
            st.caption("Valores positivos = Superávit | Negativos = Déficit")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                plotar_serie_temporal(historico.get('Res. Primário'), 'Resultado Primário', '#00FF7F', 'area')
            with col_f2:
                plotar_serie_temporal(historico.get('Res. Nominal'), 'Resultado Nominal', '#FF6347', 'area')
        
        with tab3:
            st.caption("Fluxos mensais em US$ Bilhões")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                plotar_serie_temporal(historico.get('Balança Com.'), 'Balança Comercial', '#00BFFF', 'area')
            with col_e2:
                plotar_serie_temporal(historico.get('Trans. Correntes'), 'Transações Correntes', '#FF6347', 'area')
            with col_e3:
                plotar_serie_temporal(historico.get('IDP'), 'IDP', '#00FF7F', 'area')

def plotar_serie_temporal(df: pd.DataFrame, titulo: str, cor: str, tipo: str = 'line'):
    """Plota série temporal"""
    if df is None or df.empty:
        st.info(f"📊 {titulo}: Dados indisponíveis")
        return
    
    if tipo == 'line':
        fig = px.line(df, x='data_dt', y='valor', title=titulo)
    else:
        fig = px.area(df, x='data_dt', y='valor', title=titulo)
    
    fig.update_traces(line_color=cor, fill='tozeroy' if tipo == 'area' else None)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        height=250,
        hovermode="x unified",
        showlegend=False
    )
    fig.update_xaxes(title=None, showgrid=False)
    fig.update_yaxes(title=None, showgrid=True, gridcolor='#333')
    
    st.plotly_chart(fig, use_container_width=True)

def criar_painel_cambio(df_cambio: pd.DataFrame, tipo_cambio: str, fonte_cambio: str):
    """Cria painel de histórico de câmbio com indicador de fonte"""
    with st.expander("💸 Histórico de Câmbio (desde 1994)", expanded=False):
        
        # Header com badge de fonte
        col_titulo, col_badge = st.columns([3, 1])
        with col_titulo:
            st.markdown("### 📈 Histórico de Cotações")
        with col_badge:
            badge_cambio = StatusFonte.criar_badge(tipo_cambio, fonte_cambio)
            st.markdown(badge_cambio, unsafe_allow_html=True)
        
        if df_cambio.empty:
            st.warning("Histórico de câmbio indisponível")
            return
        
        # Resumo do topo
        ultimo = df_cambio.iloc[-1]
        penultimo = df_cambio.iloc[-2] if len(df_cambio) > 1 else ultimo
        data_ref = df_cambio.index[-1].strftime('%d/%m/%Y')
        
        st.markdown(f"**📅 Último Fechamento: {data_ref}**")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            var_usd = ((ultimo['Dólar'] - penultimo['Dólar']) / penultimo['Dólar']) * 100
            cor_usd = Config.CORES['positivo'] if var_usd >= 0 else Config.CORES['negativo']
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 0.9em; color: #666;'>Dólar</div>
                <div style='font-size: 1.5em; font-weight: bold;'>R$ {ultimo['Dólar']:.2f}</div>
                <div style='color: {cor_usd};'>{var_usd:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            var_eur = ((ultimo['Euro'] - penultimo['Euro']) / penultimo['Euro']) * 100
            cor_eur = Config.CORES['positivo'] if var_eur >= 0 else Config.CORES['negativo']
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 0.9em; color: #666;'>Euro</div>
                <div style='font-size: 1.5em; font-weight: bold;'>R$ {ultimo['Euro']:.2f}</div>
                <div style='color: {cor_eur};'>{var_eur:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Informação da fonte
        fonte_texto = f"Dados obtidos via {fonte_cambio}" if tipo_cambio == 'automático' else f"Dados de referência ({fonte_cambio})"
        st.caption(f"*{fonte_texto}*")
        
        st.divider()
        
        # Abas
        tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Matriz Mensal", "📋 Dados Diários"])
        
        with tab1:
            fig = px.line(df_cambio, x=df_cambio.index, y=['Dólar', 'Euro'],
                         color_discrete_map={'Dólar': '#00FF7F', 'Euro': '#00BFFF'})
            
            fig.update_layout(
                template="plotly_dark",
                title="Evolução Histórica das Cotações",
                xaxis_title=None,
                yaxis_title="Cotação (R$)",
                hovermode="x unified",
                height=500
            )
            fig.update_yaxes(tickprefix="R$ ")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            moeda_selecionada = st.radio("Moeda:", ["Dólar", "Euro"], horizontal=True)
            
            # Cria matriz de retornos mensais
            df_mensal = df_cambio[[moeda_selecionada]].resample('ME').last()
            df_retornos = df_mensal.pct_change() * 100
            df_retornos['ano'] = df_retornos.index.year
            df_retornos['mes'] = df_retornos.index.month_name().str.slice(0, 3)
            
            # Mapeamento para português
            meses_pt = {
                'Jan': 'Jan', 'Feb': 'Fev', 'Mar': 'Mar', 'Apr': 'Abr',
                'May': 'Mai', 'Jun': 'Jun', 'Jul': 'Jul', 'Aug': 'Ago',
                'Sep': 'Set', 'Oct': 'Out', 'Nov': 'Nov', 'Dec': 'Dez'
            }
            df_retornos['mes'] = df_retornos['mes'].map(meses_pt)
            
            try:
                matriz = df_retornos.pivot(index='ano', columns='mes', values=moeda_selecionada)
                ordem_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                matriz = matriz[ordem_meses].sort_index(ascending=False)
                
                # Gradiente de cores
                cmap = LinearSegmentedColormap.from_list(
                    "rdylgn", ["#FFB3B3", "#FFFFFF", "#B3FFB3"]
                )
                
                st.dataframe(
                    matriz.style.background_gradient(cmap=cmap, vmin=-5, vmax=5)
                    .format("{:.2f}%"),
                    use_container_width=True,
                    height=500
                )
            except Exception as e:
                st.info("Matriz indisponível para o período selecionado")
        
        with tab3:
            # Botão de download
            csv = df_cambio.reset_index().to_csv(index=False)
            st.download_button(
                "📥 Baixar CSV Completo",
                csv,
                "historico_cambio.csv",
                "text/csv",
                key='download-cambio'
            )
            
            # Visualização dos dados
            df_view = df_cambio.tail(100).copy()
            df_view.index = df_view.index.strftime('%d/%m/%Y')
            st.dataframe(
                df_view.style.format({'Dólar': 'R$ {:.4f}', 'Euro': 'R$ {:.4f}'}),
                use_container_width=True
            )

def criar_painel_principal(df: pd.DataFrame, tipo_indice: str, cor_tema: str):
    """Cria painel principal com dados do indicador selecionado"""
    if df.empty:
        st.error("❌ Erro ao carregar dados do indicador")
        return
    
    # Obtém status da fonte do indicador principal
    status_indicador = StatusFonte.obter_status('indicador_principal')
    tipo_fonte = status_indicador.get('tipo', 'desconhecido')
    fonte_indicador = status_indicador.get('fonte', 'Fonte não identificada')
    
    # Header do painel
    nome_indice = tipo_indice.split()[0]
    st.title(f"📊 Painel: {nome_indice}")
    
    # Badge de fonte do indicador
    col_titulo, col_badge = st.columns([3, 1])
    with col_titulo:
        st.markdown(f"**Dados históricos atualizados até:** {df.iloc[0]['data_fmt']}")
    with col_badge:
        badge_indicador = StatusFonte.criar_badge(tipo_fonte, fonte_indicador)
        st.markdown(badge_indicador, unsafe_allow_html=True)
    
    # Metadados
    if nome_indice in METADADOS_FONTES:
        meta = METADADOS_FONTES[nome_indice]
        with st.expander("ℹ️ Metadados e Fonte", expanded=False):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Fonte Oficial:** {meta['fonte']}")
            col1.markdown(f"**Atualização:** {meta['atualizacao']}")
            col2.markdown(f"**Qualidade:** {meta['qualidade']}")
            col2.markdown(f"[🔗 Documentação oficial]({meta['link']})")
            st.caption(f"*{meta['descricao']}*")
            
            # Status atual
            st.markdown("---")
            st.markdown(f"**Status Atual:** Dados obtidos via {fonte_indicador} ({tipo_fonte})")
    
    # Alertas de qualidade
    alertas = verificar_dados(df, nome_indice)
    if alertas:
        with st.container():
            st.warning(" | ".join(alertas))
    
    # KPIs
    ultimo = df.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Taxa do Mês", f"{ultimo['valor']:.2f}%")
    col2.metric("Acumulado 12M", f"{ultimo['acum_12m']:.2f}%")
    col3.metric("Acumulado Ano", f"{ultimo['acum_ano']:.2f}%")
    col4.metric("Início da Série", df['ano'].min())
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Matriz Anual", "📋 Tabela"])
    
    with tab1:
        # Gráfico de série temporal
        df_grafico = df.sort_values('data_date').copy()
        
        # Filtra para séries muito longas
        if len(df_grafico) > 200:
            df_grafico = df_grafico[df_grafico['data_date'].dt.year >= 2000]
        
        fig = px.line(
            df_grafico,
            x='data_date',
            y='acum_12m',
            title=f"Histórico de 12 Meses - {nome_indice}"
        )
        
        fig.update_traces(line_color=cor_tema, line_width=3)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            xaxis_title=None,
            yaxis_title="Acumulado 12M (%)",
            height=500
        )
        fig.update_yaxes(ticksuffix="%")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Matriz anual
        try:
            matriz = df.pivot(index='ano', columns='mes_nome', values='valor')
            ordem_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                          'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            matriz = matriz[ordem_meses].sort_index(ascending=False)
            
            # Gradiente personalizado
            cmap = LinearSegmentedColormap.from_list(
                "pastel", ["#FFB3B3", "#FFFFFF", "#B3FFB3"]
            )
            
            st.dataframe(
                matriz.style.background_gradient(cmap=cmap, axis=None, vmin=-1.5, vmax=1.5)
                .format("{:.2f}"),
                use_container_width=True,
                height=500
            )
        except:
            st.info("Matriz indisponível para este indicador")
    
    with tab3:
        # Botão de download
        csv = df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].to_csv(index=False)
        st.download_button(
            "📥 Baixar Dados",
            csv,
            f"{nome_indice.lower()}_historico.csv",
            "text/csv"
        )
        
        # Tabela
        st.dataframe(
            df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].head(50),
            column_config={
                'data_fmt': 'Período',
                'valor': st.column_config.NumberColumn('Taxa (%)', format='%.2f'),
                'acum_ano': st.column_config.NumberColumn('Acum. Ano (%)', format='%.2f'),
                'acum_12m': st.column_config.NumberColumn('Acum. 12M (%)', format='%.2f')
            },
            hide_index=True,
            use_container_width=True
        )

def criar_calculadora_sidebar(df: pd.DataFrame, tipo_indice: str, valor_input: float):
    """Cria calculadora na sidebar"""
    if df.empty:
        return
    
    # Listas para seleção
    anos = sorted(df['ano'].unique(), reverse=True)
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    mapa_meses = {m: f"{i+1:02d}" for i, m in enumerate(meses)}
    
    # Seleção de datas
    st.sidebar.markdown("**📅 Data de Referência**")
    col_ini1, col_ini2 = st.sidebar.columns(2)
    with col_ini1:
        mes_ini = st.selectbox("Mês Inicial", meses, key='mes_ini', label_visibility="collapsed")
    with col_ini2:
        ano_ini = st.selectbox("Ano Inicial", anos, key='ano_ini', label_visibility="collapsed")
    
    st.sidebar.markdown("**🎯 Data Alvo**")
    col_fim1, col_fim2 = st.sidebar.columns(2)
    with col_fim1:
        mes_fim = st.selectbox("Mês Final", meses, index=9, key='mes_fim', label_visibility="collapsed")
    with col_fim2:
        ano_fim = st.selectbox("Ano Final", anos, key='ano_fim', label_visibility="collapsed")
    
    # Botão de cálculo
    if st.sidebar.button("🚀 Calcular Correção", type="primary", use_container_width=True):
        with st.sidebar:
            code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
            code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
            
            resultado, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
            
            if erro:
                st.error(f"❌ {erro}")
            else:
                st.sidebar.divider()
                
                # Obtém status da fonte para mostrar
                status_indicador = StatusFonte.obter_status('indicador_principal')
                fonte_texto = status_indicador.get('fonte', 'Fonte não identificada')
                
                # Exibe resultado
                nome_indice = tipo_indice.split()[0]
                operacao = "Descapitalização" if resultado['is_reverso'] else "Correção"
                
                st.sidebar.markdown(f"**{operacao} ({nome_indice})**")
                st.sidebar.markdown(
                    f"<h2 style='color: {Config.CORES.get(nome_indice, '#FFFFFF')};'>"
                    f"R$ {resultado['valor_final']:,.2f}</h2>",
                    unsafe_allow_html=True
                )
                
                # Badge da fonte
                tipo_fonte = status_indicador.get('tipo', 'desconhecido')
                if tipo_fonte != 'desconhecido':
                    badge = StatusFonte.criar_badge(tipo_fonte, fonte_texto)
                    st.sidebar.markdown(badge, unsafe_allow_html=True)
                
                # Detalhes
                st.sidebar.markdown(f"**Variação Total:** {resultado['percentual']:+.2f}%")
                st.sidebar.markdown(f"**Fator de Correção:** {resultado['fator']:.6f}")
                if resultado.get('taxa_mensal_media'):
                    st.sidebar.markdown(f"**Taxa Mensal Média:** {resultado['taxa_mensal_media']:.2f}%")
                
                # Informações adicionais
                st.sidebar.caption(f"Período: {mes_ini}/{ano_ini} → {mes_fim}/{ano_fim}")

def criar_footer():
    """Cria footer da aplicação"""
    st.markdown("---")
    
    # Painel de status das fontes
    with st.expander("📡 Status das Fontes de Dados", expanded=False):
        fontes_status = st.session_state.get('fontes_status', {})
        
        if fontes_status:
            # Cria tabela de status
            dados_status = []
            for fonte, status in fontes_status.items():
                dados_status.append({
                    'Fonte': fonte.replace('_', ' ').title(),
                    'Tipo': status.get('tipo', '❓ Desconhecido'),
                    'Origem': status.get('fonte', '-'),
                    'Última Atualização': status.get('data', '-')
                })
            
            df_status = pd.DataFrame(dados_status)
            
            # Aplica formatação condicional
            def color_tipo(val):
                if val == 'automático':
                    return f'background-color: {Config.CORES["automatico"]}20; color: {Config.CORES["automatico"]}'
                elif val == 'manual':
                    return f'background-color: {Config.CORES["manual"]}20; color: {Config.CORES["manual"]}'
                return ''
            
            st.dataframe(
                df_status.style.applymap(color_tipo, subset=['Tipo']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Status das fontes não disponível")
    
    # Footer principal
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <p>VPL Consultoria Financeira • Dados oficiais IBGE e Banco Central</p>
            <p>ℹ️ Para uso informativo. Consulte um profissional para decisões financeiras.</p>
            <p>📧 suporte@vplconsultoria.com • 📞 (11) 99999-9999</p>
        </div>
        """, unsafe_allow_html=True)

def criar_painel_controle_fontes():
    """Cria painel de controle de fontes na sidebar"""
    with st.sidebar.expander("⚙️ Controle de Fontes", expanded=False):
        st.markdown("**Configurar Fontes de Dados**")
        
        # Opção para forçar dados manuais
        forcar_manual = st.checkbox(
            "Usar dados manuais quando disponível",
            value=False,
            help="Prioriza dados de referência em vez de APIs externas"
        )
        
        if forcar_manual:
            st.info("⚠️ Modo manual ativado. Alguns dados podem não estar atualizados.")
        
        # Botão para atualizar todas as fontes
        if st.button("🔄 Atualizar Todas as Fontes", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Status atual resumido
        st.markdown("---")
        st.markdown("**Status Atual:**")
        
        fontes_status = st.session_state.get('fontes_status', {})
        for fonte, status in fontes_status.items():
            if status.get('tipo'):
                cor = Config.CORES['automatico'] if status['tipo'] == 'automático' else Config.CORES['manual']
                emoji = "🟢" if status['tipo'] == 'automático' else "🟠"
                st.markdown(f"{emoji} {fonte}: **{status['tipo']}**")
                st.caption(f"  {status.get('fonte', '')}")

# --- FUNÇÃO PRINCIPAL ---
def main():
    """Função principal da aplicação"""
    
    # Header com sistema de status
    criar_header()
    
    # Sidebar
    with st.sidebar:
        tipo_indice, valor_input = criar_sidebar()
        criar_painel_controle_fontes()
    
    # Variáveis de dados
    df_indicador = pd.DataFrame()
    df_focus = pd.DataFrame()
    
    # Bloco de Carregamento de Dados
    with st.spinner(f"📥 Carregando dados de inteligência..."):
        
        # A. Carga do Indicador Principal
        mapa_carregamento = {
            "IPCA": lambda: carregar_dados_sidra("1737", "63"),
            "INPC": lambda: carregar_dados_sidra("1736", "44"),
            "IGP-M": lambda: carregar_dados_bcb("189"),
            "SELIC": lambda: carregar_dados_bcb("4390"),
            "CDI": lambda: carregar_dados_bcb("4391")
        }
        
        nome_indice = tipo_indice.split()[0]
        carregador = mapa_carregamento.get(nome_indice)
        
        if carregador:
            df_indicador = carregador()
            cor_tema = Config.CORES.get(nome_indice, '#FFFFFF')
            
        # B. Carrega dados complementares (Usando as funções corrigidas)
        # Otimização: AwesomeAPI é muito rápida, não precisa de cache de disco complexo
        df_focus, tipo_focus, fonte_focus = carregar_dados_focus()
        df_cotacoes, tipo_cambio_real, fonte_cambio_real = carregar_cotacoes_tempo_real()
        
        # C. Carrega históricos
        kpis_macro, historico_macro, tipo_macro, fonte_macro = carregar_macro_real()
        df_cambio_hist, tipo_cambio_hist, fonte_cambio_hist = carregar_historico_cambio()
    
    # Verifica se o carregamento principal funcionou
    if df_indicador.empty:
        st.error("Não foi possível carregar o indicador principal. Verifique a conexão.")
    
    # Calculadora na sidebar
    criar_calculadora_sidebar(df_indicador, tipo_indice, valor_input)
    
    # Painéis de dados
    criar_painel_focus(df_focus, tipo_focus, fonte_focus, df_cotacoes, tipo_cambio_real, fonte_cambio_real)
    criar_painel_macro(kpis_macro, historico_macro, tipo_macro, fonte_macro)
    criar_painel_cambio(df_cambio_hist, tipo_cambio_hist, fonte_cambio_hist)
    
    # Painel principal
    criar_painel_principal(df_indicador, tipo_indice, cor_tema)
    
    # Footer
    criar_footer()
    
    # Estilos CSS
    st.markdown("""
    <style>
        .stBadge { background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
        div[data-testid="stVerticalBlock"] > div[style*="border-left"] { transition: all 0.3s ease; }
        div[data-testid="stVerticalBlock"] > div[style*="border-left"]:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .dataframe td { font-size: 0.9em !important; }
    </style>
    """, unsafe_allow_html=True)
    
# --- EXECUÇÃO ---
if __name__ == "__main__":
    main()
