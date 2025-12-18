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
                    'data': data_ref,
                    'ano': ano_atual
                }
                
                break  # Sucesso, sai do loop de tentativas
                
            except Exception as e:
                logger.warning(f"Tentativa {tentativa + 1} para {nome} falhou: {e}")
                if tentativa < Config.MAX_RETRIES - 1:
                    time.sleep(2 ** tentativa)
                continue
    
    return kpis, historico

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

# --- COMPONENTES DE UI ---
def criar_header():
    """Cria header da aplicação"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #003366 0%, #0066cc 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>VPL CONSULTORIA</h1>
        <p style='color: #E0E0E0; margin: 5px 0 0 0; font-size: 1.1em;'>
            📊 Inteligência Macroeconômica & Correção Monetária
        </p>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 10px;'>
            <span style='color: #4CAF50; font-weight: bold; background-color: rgba(255,255,255,0.1); 
                         padding: 5px 10px; border-radius: 5px;'>✓ DADOS OFICIAIS</span>
            <span style='color: #CCCCCC; font-size: 0.9em;'>
                Última atualização: {}
            </span>
        </div>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)

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

def criar_painel_focus(df_focus: pd.DataFrame, cotacoes: pd.DataFrame):
    """Cria painel de expectativas de mercado"""
    with st.expander("🔭 Expectativas de Mercado (Focus) & Câmbio", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not df_focus.empty:
                ultima_data = df_focus['data_relatorio'].max()
                df_recente = df_focus[df_focus['data_relatorio'] == ultima_data]
                ano_atual = datetime.now().year
                
                data_str = pd.to_datetime(ultima_data).strftime('%d/%m/%Y')
                st.markdown(f"**📅 Boletim Focus ({data_str})**")
                
                # KPIs principais
                df_ano = df_recente[df_recente['ano_referencia'] == ano_atual]
                pivot = df_ano.pivot_table(index='Indicador', values='previsao', aggfunc='mean')
                
                cols = st.columns(4)
                indicadores_kpi = ['IPCA', 'Selic', 'PIB Total', 'Câmbio']
                
                for idx, indicador in enumerate(indicadores_kpi):
                    with cols[idx]:
                        if indicador in pivot.index:
                            valor = pivot.loc[indicador, 'previsao']
                            if indicador == 'Câmbio':
                                cols[idx].metric(f"USD {ano_atual}", f"R$ {valor:.2f}")
                            else:
                                cols[idx].metric(f"{indicador} {ano_atual}", f"{valor:.2f}%")
                        else:
                            cols[idx].metric(f"{indicador} {ano_atual}", "-")
                
                # Tabela completa
                st.divider()
                st.markdown("##### 📊 Projeções Macroeconômicas")
                
                anos_exibir = [ano_atual, ano_atual + 1, ano_atual + 2]
                df_tabela = df_recente[df_recente['ano_referencia'].isin(anos_exibir)].copy()
                pivot_multi = df_tabela.pivot_table(index='Indicador', columns='ano_referencia', values='previsao')
                
                # Ordem lógica
                ordem = [
                    'IPCA', 'IGP-M', 'IPCA Administrados', 'Selic', 'Câmbio', 'PIB Total',
                    'Dívida líquida do setor público', 'Resultado primário', 'Resultado nominal',
                    'Balança comercial', 'Conta corrente', 'Investimento direto no país'
                ]
                ordem_filtrada = [x for x in ordem if x in pivot_multi.index]
                pivot_multi = pivot_multi.reindex(ordem_filtrada)
                
                # Formatação
                df_display = pivot_multi.copy()
                for col in df_display.columns:
                    df_display[col] = df_display.apply(
                        lambda row: formatar_valor_focus(row.name, row[col]), axis=1
                    )
                
                st.dataframe(df_display, use_container_width=True)
            else:
                st.warning("⚠️ Dados do Focus indisponíveis no momento")
        
        with col2:
            st.markdown("**💱 Câmbio em Tempo Real**")
            if not cotacoes.empty:
                cols_moeda = st.columns(2)
                for idx, (moeda, dados) in enumerate(cotacoes.iterrows()):
                    with cols_moeda[idx % 2]:
                        variacao = dados['variacao']
                        cor = Config.CORES['positivo'] if variacao >= 0 else Config.CORES['negativo']
                        st.markdown(f"""
                        <div style='background-color: rgba(0,0,0,0.1); padding: 10px; border-radius: 5px;'>
                            <div style='font-size: 0.9em; color: #666;'>{moeda}</div>
                            <div style='font-size: 1.3em; font-weight: bold;'>R$ {dados['cotacao']:.2f}</div>
                            <div style='color: {cor}; font-size: 0.9em;'>
                                {dados['variacao']:+.2f}% • {dados['atualizado']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("💡 Atualizando cotações...")

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

def criar_painel_macro(kpis: Dict, historico: Dict):
    """Cria painel de conjuntura macroeconômica"""
    with st.expander("🧩 Conjuntura Macroeconômica (Dados Oficiais)", expanded=False):
        st.markdown("Monitoramento dos principais indicadores da economia brasileira.")
        
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
        st.markdown("##### 📈 Tendências (Últimos 5 Anos)")
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

def criar_painel_cambio(df_cambio: pd.DataFrame):
    """Cria painel de histórico de câmbio"""
    with st.expander("💸 Histórico de Câmbio (desde 1994)", expanded=False):
        if df_cambio.empty:
            st.warning("Histórico de câmbio indisponível")
            return
        
        # Resumo do topo
        ultimo = df_cambio.iloc[-1]
        penultimo = df_cambio.iloc[-2] if len(df_cambio) > 1 else ultimo
        data_ref = df_cambio.index[-1].strftime('%d/%m/%Y')
        
        st.markdown(f"**📅 Fechamento: {data_ref}**")
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
    
    # Header do painel
    nome_indice = tipo_indice.split()[0]
    st.title(f"📊 Painel: {nome_indice}")
    
    # Metadados
    if nome_indice in METADADOS_FONTES:
        meta = METADADOS_FONTES[nome_indice]
        with st.expander("ℹ️ Metadados e Fonte", expanded=False):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Fonte:** {meta['fonte']}")
            col1.markdown(f"**Atualização:** {meta['atualizacao']}")
            col2.markdown(f"**Qualidade:** {meta['qualidade']}")
            col2.markdown(f"[🔗 Documentação oficial]({meta['link']})")
            st.caption(f"*{meta['descricao']}*")
    
    # Alertas de qualidade
    alertas = verificar_dados(df, nome_indice)
    if alertas:
        with st.container():
            st.warning(" | ".join(alertas))
    
    # KPIs
    ultimo = df.iloc[0]
    st.markdown(f"**📅 Dados atualizados até:** {ultimo['data_fmt']}")
    
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
                
                # Exibe resultado
                nome_indice = tipo_indice.split()[0]
                operacao = "Descapitalização" if resultado['is_reverso'] else "Correção"
                
                st.sidebar.markdown(f"**{operacao} ({nome_indice})**")
                st.sidebar.markdown(
                    f"<h2 style='color: {Config.CORES.get(nome_indice, '#FFFFFF')};'>"
                    f"R$ {resultado['valor_final']:,.2f}</h2>",
                    unsafe_allow_html=True
                )
                
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
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <p>VPL Consultoria Financeira • Dados oficiais IBGE e Banco Central</p>
            <p>ℹ️ Para uso informativo. Consulte um profissional para decisões financeiras.</p>
            <p>📧 suporte@vplconsultoria.com • 📞 (11) 99999-9999</p>
        </div>
        """, unsafe_allow_html=True)

# --- FUNÇÃO PRINCIPAL ---
def main():
    """Função principal da aplicação"""
    
    # Header
    criar_header()
    
    # Sidebar
    with st.sidebar:
        tipo_indice, valor_input = criar_sidebar()
    
    # Carrega dados do indicador selecionado
    with st.spinner(f"📥 Carregando dados de {tipo_indice}..."):
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
        else:
            st.error("Indicador não suportado")
            st.stop()
    
    if df_indicador.empty:
        st.error("Não foi possível carregar os dados. Tente novamente.")
        st.stop()
    
    # Carrega dados complementares em paralelo
    with st.spinner("🔄 Carregando dados complementares..."):
        # Dados que podem carregar em paralelo
        import threading
        
        resultados = {}
        def carregar_paralelo(nome, funcao):
            resultados[nome] = funcao()
        
        threads = [
            threading.Thread(target=carregar_paralelo, args=('focus', carregar_focus)),
            threading.Thread(target=carregar_paralelo, args=('cotacoes', carregar_cotacoes_tempo_real)),
            threading.Thread(target=carregar_paralelo, args=('cambio', carregar_historico_cambio)),
            threading.Thread(target=carregar_paralelo, args=('macro', lambda: carregar_macro_real()[0])),
            threading.Thread(target=carregar_paralelo, args=('macro_hist', lambda: carregar_macro_real()[1]))
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        df_focus = resultados.get('focus', pd.DataFrame())
        df_cotacoes = resultados.get('cotacoes', pd.DataFrame())
        df_cambio = resultados.get('cambio', pd.DataFrame())
        kpis_macro = resultados.get('macro', {})
        historico_macro = resultados.get('macro_hist', {})
    
    # Calculadora na sidebar
    criar_calculadora_sidebar(df_indicador, tipo_indice, valor_input)
    
    # Painéis de dados
    criar_painel_focus(df_focus, df_cotacoes)
    criar_painel_macro(kpis_macro, historico_macro)
    criar_painel_cambio(df_cambio)
    
    # Painel principal
    criar_painel_principal(df_indicador, tipo_indice, cor_tema)
    
    # Footer
    criar_footer()
    
    # Status do sistema (oculto por padrão)
    with st.sidebar.expander("🔧 Status do Sistema", expanded=False):
        st.caption(f"Cache: {len(st.cache_data.clear.callbacks)} funções")
        st.caption(f"Indicador: {nome_indice}")
        st.caption(f"Registros: {len(df_indicador)}")
        if not df_indicador.empty:
            st.caption(f"Período: {df_indicador['data_date'].min():%Y-%m} a {df_indicador['data_date'].max():%Y-%m}")

# --- EXECUÇÃO ---
if __name__ == "__main__":
    main()
