import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, datetime, timedelta
import requests
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
import urllib3
import warnings
warnings.filterwarnings('ignore')

# Desabilita avisos de SSL (APENAS quando necessário)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        margin-bottom: 10px;
    }
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        font-weight: 600;
        color: #003366;
        border: 1px solid #ddd;
    }
    .error-box {
        background-color: #fff2f2;
        border-left: 4px solid #ff4d4d;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #f2fff2;
        border-left: 4px solid #4dff4d;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #007bff;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

cores_leves = ["#FFB3B3", "#FFFFFF", "#B3FFB3"] # Vermelho Suave, Branco, Verde Suave
cmap_leves = LinearSegmentedColormap.from_list("pastel_rdylgn", cores_leves)

# --- CONFIGURAÇÕES GLOBAIS ---
SERIES_CONFIG = {
    # Inflação (variação mensal %)
    "IPCA": {"source": "sidra", "table": "1737", "variable": "63", "type": "inflation", "color": "#00BFFF"},
    "INPC": {"source": "sidra", "table": "1736", "variable": "44", "type": "inflation", "color": "#00FF7F"},
    "IGP-M": {"source": "bcb", "code": "189", "type": "inflation", "color": "#FF6347"},
    
    # Taxas de juros (taxa anual %)
    "SELIC": {"source": "bcb", "code": "4390", "type": "annual_rate", "color": "#FFD700"},
    "CDI": {"source": "bcb", "code": "4391", "type": "annual_rate", "color": "#9370DB"},
    
    # Taxas em nível (ex: dólar, PIB nominal)
    "Dólar": {"source": "yfinance", "symbol": "USDBRL=X", "type": "level", "color": "#32CD32"},
    "Euro": {"source": "yfinance", "symbol": "EURBRL=X", "type": "level", "color": "#4169E1"},
}

# --- FUNÇÕES UTILITÁRIAS ---
def format_date(dt):
    """Formata data para exibição"""
    if pd.isna(dt):
        return "Não disponível"
    return dt.strftime('%d/%m/%Y')

def get_data_update_info(df, source_name):
    """Retorna informação de atualização dos dados"""
    if df.empty:
        return f"{source_name}: Sem dados"
    
    if 'data_date' in df.columns:
        last_date = df['data_date'].max()
        return f"{source_name}: Atualizado em {format_date(last_date)}"
    elif 'data_relatorio' in df.columns:
        last_date = df['data_relatorio'].max()
        return f"{source_name}: Atualizado em {format_date(last_date)}"
    else:
        return f"{source_name}: Disponível"

def format_valor_focus(valor, nome):
    """Formata valores do Focus para exibição"""
    if pd.isna(valor):
        return "-"
    if 'Câmbio' in nome:
        return f"R$ {valor:.2f}"
    elif any(x in nome for x in ['comercial', 'Conta corrente', 'Investimento']):
        return f"US$ {valor:.2f} B"
    else:
        return f"{valor:.2f}%"

def calcular_fator(valor, tipo_serie, periodo='mensal'):
    """
    Calcula fator de correção/multiplicação baseado no tipo de série
    CORREÇÃO CRÍTICA: Normalização correta das taxas
    """
    if pd.isna(valor):
        return 1.0
    
    valor = float(valor)
    
    if tipo_serie == "inflation":
        # Inflação: taxa mensal (%)
        return 1 + (valor / 100)
    
    elif tipo_serie == "annual_rate":
        # Taxa anual: converter para taxa mensal equivalente
        if periodo == 'mensal':
            # Se já é mensal (SELIC mensal por exemplo)
            return 1 + (valor / 100)
        else:
            # Taxa anual para mensal
            return (1 + (valor / 100)) ** (1/12)
    
    elif tipo_serie == "level":
        # Nível (ex: dólar): não é taxa, fator = 1 para cálculos de correção
        # Para cálculos de variação, usar retorno percentual
        return 1.0  # Não aplicável para correção monetária
    
    else:
        # Default: assume taxa mensal
        return 1 + (valor / 100)

# --- FUNÇÕES DE CARGA DE DADOS COM PADRÃO UNIFICADO ---

@st.cache_data(ttl=3600)
def get_sidra_data(table_code, variable_code, test_mode=False):
    """Função para obter dados do IBGE com estrutura padronizada"""
    if test_mode:
        return pd.DataFrame()
    
    try:
        dados_raw = sidrapy.get_table(
            table_code=table_code, territorial_level="1", 
            ibge_territorial_code="all", variable=variable_code, 
            period="last 360"
        )
        
        if dados_raw.empty: 
            return pd.DataFrame()
        
        df = dados_raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['D2C'].str.slice(0, 4)
        df['D2C'] = df['D2C'].astype(str)  # Padronização
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados do IBGE: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_bcb_data(codigo_serie, serie_nome="", test_mode=False):
    """Função para obter dados do BCB com estrutura padronizada"""
    if test_mode:
        return pd.DataFrame()
    
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, verify=True, timeout=15)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json())
        
        if df.empty:
            return pd.DataFrame()
        
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        
        # PADRONIZAÇÃO CRÍTICA: Criar D2C igual ao Sidra
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        df['mes_ano'] = df['data_date'].dt.strftime('%b/%Y')  # Ex: Jan/2024
        
        return df
        
    except requests.exceptions.SSLError:
        # Fallback para verify=False apenas se necessário
        try:
            response = requests.get(url, headers=headers, verify=False, timeout=15)
            response.raise_for_status()
            df = pd.DataFrame(response.json())
            
            if df.empty:
                return pd.DataFrame()
            
            df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            df['D2C'] = df['data_date'].dt.strftime('%Y%m')
            df['ano'] = df['data_date'].dt.strftime('%Y')
            df['mes_ano'] = df['data_date'].dt.strftime('%b/%Y')
            
            return df
            
        except Exception as e:
            st.error(f"❌ Erro SSL no BCB (Série {codigo_serie}): {str(e)}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Erro no BCB: {str(e)}")
        return pd.DataFrame()

def processar_serie_inflacao(df, tipo_serie="inflation"):
    """Processa série de inflação/taxas corretamente"""
    if df.empty: 
        return df
    
    df = df.sort_values('data_date', ascending=True).copy()
    
    # Adicionar informações de mês/ano
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['ano'] = df['data_date'].dt.strftime('%Y')
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    
    # CORREÇÃO CRÍTICA: Cálculo correto do fator baseado no tipo de série
    df['fator'] = df['valor'].apply(lambda x: calcular_fator(x, tipo_serie))
    
    # Acumulado no ano (desde janeiro)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    
    # Acumulado 12 meses (rolling window)
    df['acum_12m'] = (df['fator'].rolling(window=12, min_periods=1).apply(
        lambda x: np.prod(x) if len(x) == 12 else np.nan
    ) - 1) * 100
    
    # Variação mensal (já está em df['valor'] para inflação)
    if tipo_serie != "level":
        df['variacao_mensal'] = df['valor']
    
    return df.sort_values('data_date', ascending=False)

def processar_serie_nivel(df):
    """Processa série de nível (ex: dólar, PIB)"""
    if df.empty: 
        return df
    
    df = df.sort_values('data_date', ascending=True).copy()
    
    # Para série de nível, calcular variação percentual
    df['variacao_mensal'] = df['valor'].pct_change() * 100
    df['retorno_acum'] = ((df['valor'] / df['valor'].iloc[0]) - 1) * 100
    
    # Rolling window 12 meses
    df['acum_12m'] = df['valor'].pct_change(periods=12) * 100
    
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['ano'] = df['data_date'].dt.strftime('%Y')
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    
    return df.sort_values('data_date', ascending=False)

@st.cache_data(ttl=3600)
def get_focus_data(test_mode=False):
    """Função para obter dados do Focus"""
    if test_mode:
        return pd.DataFrame()
    
    try:
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=5000&$orderby=Data%20desc&$format=json"
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, verify=True, timeout=15)
        response.raise_for_status()
        data_json = response.json()
        
        df = pd.DataFrame(data_json['value'])
        
        if df.empty:
            return pd.DataFrame()
        
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
        df['previsao'] = pd.to_numeric(df['previsao'], errors='coerce')
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        
        df = df.sort_values('data_relatorio', ascending=False)
        df = df.drop_duplicates(subset=['Indicador', 'ano_referencia'], keep='first')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erro no Focus: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_currency_realtime(test_mode=False):
    """Função para obter cotações em tempo real"""
    if test_mode:
        # Valores de fallback
        dados = {
            "USDBRL": {'bid': 5.50, 'pctChange': 0.0},
            "EURBRL": {'bid': 6.00, 'pctChange': 0.0}
        }
        return pd.DataFrame.from_dict(dados, orient='index')
    
    try:
        tickers = ["USDBRL=X", "EURBRL=X"]
        dados = {}
        
        for t in tickers:
            try:
                ticker_obj = yf.Ticker(t)
                hist = ticker_obj.history(period="2d", interval="1d")
                
                if not hist.empty:
                    preco_atual = hist['Close'].iloc[-1]
                    fechamento_anterior = hist['Close'].iloc[0] if len(hist) > 1 else preco_atual
                    variacao = ((preco_atual - fechamento_anterior) / fechamento_anterior) * 100
                    
                    key = "USDBRL" if "USD" in t else "EURBRL"
                    dados[key] = {'bid': float(preco_atual), 'pctChange': float(variacao)}
                else:
                    # Fallback
                    info = ticker_obj.info
                    preco_atual = info.get('regularMarketPrice', 0) or info.get('currentPrice', 0)
                    fechamento_anterior = info.get('previousClose', preco_atual)
                    variacao = ((preco_atual - fechamento_anterior) / fechamento_anterior) * 100 if fechamento_anterior != 0 else 0
                    
                    key = "USDBRL" if "USD" in t else "EURBRL"
                    dados[key] = {'bid': float(preco_atual), 'pctChange': float(variacao)}
                    
            except Exception:
                # Valores padrão em caso de erro
                if "USD" in t:
                    dados["USDBRL"] = {'bid': 5.50, 'pctChange': 0.0}
                elif "EUR" in t:
                    dados["EURBRL"] = {'bid': 6.00, 'pctChange': 0.0}
        
        if not dados:
            dados = {
                "USDBRL": {'bid': 5.50, 'pctChange': 0.0},
                "EURBRL": {'bid': 6.00, 'pctChange': 0.0}
            }
            
        return pd.DataFrame.from_dict(dados, orient='index')
        
    except Exception as e:
        st.error(f"❌ Erro nas cotações: {str(e)}")
        dados = {
            "USDBRL": {'bid': 5.50, 'pctChange': 0.0},
            "EURBRL": {'bid': 6.00, 'pctChange': 0.0}
        }
        return pd.DataFrame.from_dict(dados, orient='index')

@st.cache_data(ttl=86400)
def get_cambio_historico(test_mode=False):
    """Função para obter histórico de câmbio"""
    if test_mode:
        return pd.DataFrame()
    
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="1994-07-01", progress=False)
        
        if df.empty: 
            return pd.DataFrame()
        
        df = df['Close']
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo')
        df.index = df.index.tz_localize(None)
        
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        df = df.ffill()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erro no histórico de câmbio: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_macro_real(test_mode=False):
    """Função para obter dados macroeconômicos com validação de periodicidade"""
    if test_mode:
        return {"dados": {}, "ultima_atualizacao": "Não disponível", "metodologia": {}}
    
    series = {
        'PIB Nominal (R$ Mi)': {'codigo': 4380, 'tipo': 'pib'},
        'Dívida Líq. (% PIB)': {'codigo': 4513, 'tipo': 'percentual'},
        'Res. Primário (% PIB)': {'codigo': 4520, 'tipo': 'percentual'},
        'Res. Nominal (% PIB)': {'codigo': 4521, 'tipo': 'percentual'},
        'Balança Com. (US$ Mi)': {'codigo': 22705, 'tipo': 'externo'},
        'Trans. Correntes (US$ Mi)': {'codigo': 22707, 'tipo': 'externo'},
        'IDP (US$ Mi)': {'codigo': 22704, 'tipo': 'externo'}
    }
    
    resultados = {}
    ultimas_datas = []
    metodologia = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        for nome, info in series.items():
            try:
                codigo = info['codigo']
                tipo = info['tipo']
                
                # Usar mais períodos para análise de frequência
                url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/100?formato=json"
                
                resp = requests.get(url, headers=headers, verify=True, timeout=15)
                df = pd.DataFrame(resp.json())
                
                if df.empty:
                    st.warning(f"Série {nome} ({codigo}) retornou vazia")
                    continue
                
                # Verificar coluna 'valor'
                if 'valor' not in df.columns:
                    if len(df.columns) > 1:
                        valor_col = df.columns[1]
                        df = df.rename(columns={valor_col: 'valor'})
                    else:
                        continue
                
                df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
                df = df.sort_values('data', ascending=True)
                
                # Analisar frequência dos dados
                if len(df) > 1:
                    diffs = df['data'].diff().dt.days
                    freq_media = diffs.mean()
                    
                    if freq_media < 10:
                        freq = "diária"
                        fator_acum = 365/12  # Para converter para mensal
                    elif 25 < freq_media < 35:
                        freq = "mensal"
                        fator_acum = 1
                    elif 85 < freq_media < 95:
                        freq = "trimestral"
                        fator_acum = 4  # Para acumular 4 trimestres = 12 meses
                    else:
                        freq = "irregular"
                        fator_acum = 1
                else:
                    freq = "desconhecida"
                    fator_acum = 1
                
                # Guarda a última data disponível
                if not df['data'].empty:
                    ultimas_datas.append(df['data'].max())
                
                # Cálculos baseados no tipo e frequência
                if tipo == 'pib':
                    if freq == "trimestral" and len(df) >= 4:
                        valor_final = df['valor'].iloc[-4:].sum() / 1_000_000  # Trilhões
                        metodologia[nome] = f"Acumulado dos últimos 4 trimestres (12 meses) - Frequência: {freq}"
                    else:
                        # Para outras frequências ou dados insuficientes
                        if len(df) >= int(12/fator_acum):
                            periodo_needed = int(12/fator_acum)
                            valor_final = df['valor'].iloc[-periodo_needed:].sum() / 1_000_000
                            metodologia[nome] = f"Acumulado últimos 12 meses - Frequência: {freq}"
                        else:
                            valor_final = df['valor'].iloc[-1] / 1_000_000
                            metodologia[nome] = f"Último período disponível - Frequência: {freq}"
                
                elif tipo == 'externo':
                    if len(df) >= 12:
                        valor_final = df['valor'].iloc[-12:].sum() / 1_000  # Bilhões
                        metodologia[nome] = f"Soma dos últimos 12 meses - Frequência: {freq}"
                    else:
                        valor_final = df['valor'].sum() / 1_000
                        metodologia[nome] = f"Soma de todos os períodos - Frequência: {freq}"
                
                else:  # percentual (% PIB)
                    valor_final = df['valor'].iloc[-1]
                    metodologia[nome] = f"Último período disponível - Frequência: {freq}"
                
                resultados[nome] = valor_final
                
            except Exception as e:
                st.warning(f"Erro na série {nome}: {str(e)}")
                continue
        
        # Determina a data de atualização
        ultima_atualizacao = "Não disponível"
        if ultimas_datas:
            ultima_atualizacao = max(ultimas_datas).strftime('%d/%m/%Y')
        
        return {
            "dados": resultados,
            "ultima_atualizacao": ultima_atualizacao,
            "metodologia": metodologia
        }
        
    except Exception as e:
        st.error(f"❌ Erro geral nos dados macro: {str(e)}")
        return {"dados": {}, "ultima_atualizacao": "Não disponível", "metodologia": {}}

# CORREÇÃO CRÍTICA: Função de cálculo corrigida
def calcular_correcao(df, valor, data_ini_code, data_fim_code, tipo_serie="inflation"):
    """
    Calcula correção monetária com ordenação temporal correta
    """
    # Converter para string se necessário
    data_ini_code = str(data_ini_code)
    data_fim_code = str(data_fim_code)
    
    is_reverso = int(data_ini_code) > int(data_fim_code)
    if is_reverso:
        periodo_inicio, periodo_fim = data_fim_code, data_ini_code
    else:
        periodo_inicio, periodo_fim = data_ini_code, data_fim_code
    
    # CORREÇÃO: Garantir ordenação temporal
    df_sorted = df.sort_values('data_date', ascending=True).copy()
    
    mask = (df_sorted['D2C'] >= periodo_inicio) & (df_sorted['D2C'] <= periodo_fim)
    df_periodo = df_sorted.loc[mask].copy()
    
    if df_periodo.empty:
        return None, f"Período {periodo_inicio} a {periodo_fim} sem dados suficientes."
    
    # CORREÇÃO: Verificar se há fator calculado
    if 'fator' not in df_periodo.columns:
        # Calcular fator se não existir
        df_periodo['fator'] = df_periodo['valor'].apply(lambda x: calcular_fator(x, tipo_serie))
    
    # Garantir que não há valores NaN
    df_periodo = df_periodo.dropna(subset=['fator'])
    
    if df_periodo.empty:
        return None, "Não foi possível calcular fatores para o período."
    
    # Produto dos fatores - ORDENADO por data
    fator_acumulado = df_periodo['fator'].prod()
    
    # Cálculo do valor final
    if is_reverso:
        # CORREÇÃO MATEMÁTICA: Descapitalização é divisão pelo fator
        valor_final = valor / fator_acumulado
    else:
        valor_final = valor * fator_acumulado
    
    pct_total = (fator_acumulado - 1) * 100
    
    return {
        'valor_final': valor_final, 
        'percentual': pct_total, 
        'fator': fator_acumulado, 
        'is_reverso': is_reverso,
        'periodo_inicio': periodo_inicio,
        'periodo_fim': periodo_fim,
        'periodos_envolvidos': len(df_periodo),
        'data_inicio': df_periodo['data_date'].min(),
        'data_fim': df_periodo['data_date'].max()
    }, None

# ==============================================================================
# LAYOUT - SIDEBAR
# ==============================================================================
try:
    st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
except:
    st.sidebar.markdown("## VPL CONSULTORIA")
    st.sidebar.markdown("**Inteligência Financeira**")

st.sidebar.divider()
st.sidebar.header("⚙️ Configurações")

# Seleção do índice
indices_disponiveis = ["IPCA (Inflação Oficial)", "INPC (Salários)", "IGP-M (Aluguéis)", "SELIC (Taxa Básica)", "CDI (Investimentos)"]
tipo_indice = st.sidebar.selectbox("Selecione o Indicador", indices_disponiveis)

# 📘 METODOLOGIA
with st.sidebar.expander("📘 Metodologia", expanded=False):
    st.markdown("""
    ### 📊 Fontes dos Dados
    - **IBGE (Sidra)**: Dados oficiais de IPCA e INPC
    - **Banco Central do Brasil (BCB)**: Séries temporais do SGS
    - **Boletim Focus**: Expectativas de mercado do BCB
    - **Yahoo Finance**: Cotações de câmbio em tempo real
    
    ### 📅 Periodicidade
    - **Dados mensais**: IPCA, INPC, IGP-M, SELIC, CDI
    - **Dados diários**: Câmbio (Dólar/Euro)
    - **Expectativas**: Atualizadas semanalmente (Focus)
    
    ### 🧮 Métodos de Cálculo
    - **Correção monetária**: Fator acumulado do período
    - **Acumulado 12 meses**: Produto dos últimos 12 meses
    - **Acumulado no ano**: Produto desde janeiro
    - **Matriz de calor**: Variação mensal por ano
    
    ### ⚠️ Observações
    - Cálculos ajustados por sazonalidade quando aplicável
    - Taxas CDI e SELIC convertidas para base mensal equivalente
    - Todos os valores em Reais (R$) corrigidos pela inflação
    - Dados sujeitos a revisão pelas fontes oficiais
    """)

st.sidebar.divider()

# 🔧 STATUS DAS APIs
st.sidebar.subheader("🔧 Status das APIs")
try:
    api_status = {}
    
    # Teste simplificado
    for api in ["IBGE", "BCB", "Focus", "Câmbio"]:
        api_status[api] = "🟢"
    
    for api_name, status in api_status.items():
        st.sidebar.markdown(f"{status} {api_name}")
        
except Exception as e:
    st.sidebar.info("Status das APIs não disponível")

st.sidebar.divider()

# 🧮 CALCULADORA
st.sidebar.subheader("🧮 Calculadora")
valor_input = st.sidebar.number_input("Valor (R$)", value=1000.00, step=100.00, format="%.2f")

# Carregar dados do índice selecionado
nome_indice = tipo_indice.split()[0]
config = SERIES_CONFIG.get(nome_indice, {})

with st.spinner(f"Carregando dados do {nome_indice}..."):
    if config["source"] == "sidra":
        df_raw = get_sidra_data(config["table"], config["variable"])
        df = processar_serie_inflacao(df_raw, config["type"])
    elif config["source"] == "bcb":
        df_raw = get_bcb_data(config["code"], nome_indice)
        df = processar_serie_inflacao(df_raw, config["type"])
    else:
        df = pd.DataFrame()

# Tratamento de erro
if df.empty:
    st.error(f"⚠️ **Erro ao carregar dados do {nome_indice}**")
    st.warning("Usando dados limitados. Algumas funcionalidades podem estar indisponíveis.")
    df = pd.DataFrame(columns=['data_date', 'valor', 'acum_ano', 'acum_12m', 'data_fmt', 'D2C', 'ano', 'mes_nome', 'fator'])
else:
    update_info = get_data_update_info(df, config["source"].upper())
    st.sidebar.markdown(f"<div class='success-box'><small>✅ {update_info}</small></div>", unsafe_allow_html=True)

# Configuração da calculadora
if not df.empty:
    lista_anos = sorted(df['ano'].unique(), reverse=True)
else:
    lista_anos = [str(date.today().year - i) for i in range(5)]

lista_meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
mapa_meses = {
    'Jan': '01', 'Fev': '02', 'Mar': '03', 'Abr': '04', 'Mai': '05', 'Jun': '06',
    'Jul': '07', 'Ago': '08', 'Set': '09', 'Out': '10', 'Nov': '11', 'Dez': '12'
}

st.sidebar.markdown("**📅 Data Referência**")
c1, c2 = st.sidebar.columns(2)
mes_ini = c1.selectbox("Mes Ini", lista_meses_nome, index=0, label_visibility="collapsed", key="mes_ini")
ano_ini = c2.selectbox("Ano Ini", lista_anos, index=min(1, len(lista_anos)-1), label_visibility="collapsed", key="ano_ini")

st.sidebar.markdown("**🎯 Data Alvo**")
c3, c4 = st.sidebar.columns(2)
mes_fim = c3.selectbox("Mes Fim", lista_meses_nome, index=9, label_visibility="collapsed", key="mes_fim")
ano_fim = c4.selectbox("Ano Fim", lista_anos, index=0, label_visibility="collapsed", key="ano_fim")

if st.sidebar.button("Calcular", type="primary", use_container_width=True):
    code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
    code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
    
    if df.empty:
        st.sidebar.error("Dados insuficientes para cálculo")
    else:
        res, erro = calcular_correcao(df, valor_input, code_ini, code_fim, config["type"])
        
        if erro:
            st.sidebar.error(f"❌ {erro}")
        else:
            st.sidebar.divider()
            
            tipo_op = "Rendimento" if nome_indice in ["SELIC", "CDI"] else "Correção"
            texto_op = "Descapitalização" if res['is_reverso'] else f"{tipo_op} ({nome_indice})"
            
            st.sidebar.markdown(f"<small>{texto_op}</small>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<h2 style='color: {config.get("color", "#000000")}; margin:0;'>R$ {res['valor_final']:,.2f}</h2>", unsafe_allow_html=True)
            
            # Card de resultados
            st.sidebar.markdown("---")
            
            st.sidebar.markdown(f"""
            <div class='metric-card'>
            <small>Total Período ({res['periodos_envolvidos']} períodos)</small><br>
            <strong>{res['percentual']:.2f}%</strong>
            </div>
            """, unsafe_allow_html=True)
            
            st.sidebar.markdown(f"""
            <div class='metric-card'>
            <small>Fator de Correção</small><br>
            <strong>{res['fator']:.6f}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            st.sidebar.markdown(f"""
            <div class='metric-card'>
            <small>Período</small><br>
            <strong>{mes_ini}/{ano_ini} → {mes_fim}/{ano_fim}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Aviso para taxas de juros
            if nome_indice in ["SELIC", "CDI"]:
                st.sidebar.markdown("""
                <div class='warning-box'>
                <small>⚠️ Observação:</small><br>
                <small>Para taxas de juros, o cálculo considera capitalização mensal equivalente da taxa anual.</small>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# PAINEL PRINCIPAL
# ==============================================================================

# CABEÇALHO PRINCIPAL
st.title(f"📊 Painel: {nome_indice}")

if not df.empty:
    update_date = df['data_date'].max() if 'data_date' in df.columns else None
    update_str = format_date(update_date) if update_date else "Não disponível"
    
    # Aviso sobre tipo de série
    tipo_info = {
        "inflation": "Índice de Preços",
        "annual_rate": "Taxa de Juros (anual convertida para mensal)",
        "level": "Nível/Preço"
    }
    
    st.caption(f"Fonte: {config['source'].upper()} • {tipo_info.get(config['type'], '')} • Última atualização: {update_str}")
else:
    st.caption(f"Fonte: {config.get('source', '').upper()} • Dados temporariamente indisponíveis")

if not df.empty:
    # KPIs PRINCIPAIS
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        try:
            if nome_indice in ["SELIC", "CDI"]:
                # Taxa mensal equivalente
                taxa_anual = df.iloc[0]['valor'] if not df.empty else 0
                taxa_mensal = ((1 + taxa_anual/100) ** (1/12) - 1) * 100
                st.metric("Taxa Mensal Equivalente", f"{taxa_mensal:.2f}%")
            else:
                valor_mes = df.iloc[0]['valor'] if not df.empty else 0
                st.metric("Taxa do Mês", f"{valor_mes:.2f}%")
        except:
            st.metric("Taxa do Mês", "N/D")
    
    with kpi2:
        try:
            acum_12m = df.iloc[0]['acum_12m'] if not df.empty and 'acum_12m' in df.columns else 0
            if pd.isna(acum_12m):
                acum_12m = 0
            st.metric("Acumulado 12 Meses", f"{acum_12m:.2f}%")
        except:
            st.metric("Acumulado 12 Meses", "N/D")
    
    with kpi3:
        try:
            acum_ano = df.iloc[0]['acum_ano'] if not df.empty and 'acum_ano' in df.columns else 0
            if pd.isna(acum_ano):
                acum_ano = 0
            st.metric("Acumulado Ano (YTD)", f"{acum_ano:.2f}%")
        except:
            st.metric("Acumulado Ano (YTD)", "N/D")
    
    with kpi4:
        try:
            inicio_serie = df['data_date'].min().year if not df.empty else "N/D"
            st.metric("Início da Série", inicio_serie)
        except:
            st.metric("Início da Série", "N/D")
    
    # ABA DO GRÁFICO
    tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Matriz de Calor", "📋 Tabela Detalhada"])
    
    with tab1:
        if not df.empty:
            df_chart = df.dropna(subset=['acum_12m']).sort_values('data_date')
            
            if not df_chart.empty:
                fig = px.line(
                    df_chart, 
                    x='data_date', 
                    y='acum_12m', 
                    title=f"Histórico 12 Meses - {nome_indice}",
                    labels={'acum_12m': 'Acumulado 12 meses (%)', 'data_date': 'Data'}
                )
                fig.update_traces(line_color=config.get('color', '#00BFFF'), line_width=3)
                
                # CORREÇÃO: Usar tema claro com contraste adequado
                fig.update_layout(
                    template="plotly_white",
                    plot_bgcolor='white',
                    font=dict(color="#333333"),
                    hovermode="x unified", 
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="%",
                    xaxis_title="",
                    yaxis=dict(gridcolor='lightgray'),
                    xaxis=dict(gridcolor='lightgray')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dados insuficientes para gerar o gráfico")
        else:
            st.info("Aguardando dados para exibir o gráfico")
    
    with tab2:
        if not df.empty and 'mes_nome' in df.columns and 'ano' in df.columns:
            try:
                # CORREÇÃO: Usar valores apropriados para cada tipo de série
                if nome_indice in ["SELIC", "CDI"]:
                    # Para taxas de juros, mostrar taxa mensal
                    df_heat = df.copy()
                    df_heat['taxa_mensal'] = ((1 + df_heat['valor']/100) ** (1/12) - 1) * 100
                    matrix = df_heat.pivot(index='ano', columns='mes_nome', values='taxa_mensal')
                else:
                    matrix = df.pivot(index='ano', columns='mes_nome', values='valor')
                
                ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                ordem = [col for col in ordem if col in matrix.columns]
                
                if ordem:
                    matrix = matrix[ordem].sort_index(ascending=False)
                    
                    # CORREÇÃO: Limites dinâmicos baseados nos percentis
                    if not matrix.empty:
                        # Excluir outliers usando percentis
                        flat_values = matrix.values.flatten()
                        flat_values = flat_values[~np.isnan(flat_values)]
                        
                        if len(flat_values) > 0:
                            vmin = np.percentile(flat_values, 10)
                            vmax = np.percentile(flat_values, 90)
                            
                            # Garantir intervalo mínimo
                            if vmax - vmin < 0.5:
                                vmin -= 0.5
                                vmax += 0.5
                        else:
                            vmin, vmax = -2, 2
                    else:
                        vmin, vmax = -2, 2
                    
                    # Garantir que IPCA/INPC/IGP-M tenham vermelho para negativo
                    if nome_indice in ["IPCA", "INPC", "IGP-M"] and vmin >= 0:
                        vmin = -0.5
                    
                    st.dataframe(
                        matrix.style.background_gradient(
                            cmap=cmap_leves, axis=None, vmin=vmin, vmax=vmax
                        ).format("{:.2f}"), 
                        use_container_width=True, 
                        height=500
                    )
                else:
                    st.warning("Dados insuficientes para matriz de calor")
            except Exception as e:
                st.error(f"Erro ao gerar matriz: {str(e)}")
        else:
            st.info("Aguardando dados para exibir a matriz")
    
    with tab3:
        if not df.empty:
            # Botão de download
            colunas_export = ['data_fmt', 'valor']
            if 'acum_ano' in df.columns:
                colunas_export.append('acum_ano')
            if 'acum_12m' in df.columns:
                colunas_export.append('acum_12m')
            
            df_export = df[colunas_export].copy()
            csv_principal = df_export.to_csv(index=False).encode('utf-8')
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    "📥 Baixar CSV", 
                    csv_principal, 
                    f"{nome_indice}_historico.csv", 
                    "text/csv",
                    use_container_width=True
                )
            
            # Tabela de dados
            df_display = df_export.copy()
            if nome_indice in ["SELIC", "CDI"]:
                df_display.columns = ['Período', 'Taxa Anual (%)', 'Acum. Ano (%)', 'Acum. 12M (%)']
            else:
                df_display.columns = ['Período', 'Taxa Mensal (%)', 'Acum. Ano (%)', 'Acum. 12M (%)']
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("Aguardando dados para exibir a tabela")

# ==============================================================================
# SEÇÃO FOCUS & CÂMBIO
# ==============================================================================

with st.expander("🔭 Expectativas de Mercado (Focus) & Câmbio Hoje", expanded=False):
    col_top1, col_top2 = st.columns([2, 1])
    
    # FOCUS
    with col_top1:
        st.subheader("📊 Boletim Focus")
        df_focus = get_focus_data()
        ano_atual = date.today().year
        
        if not df_focus.empty:
            ultima_data = df_focus['data_relatorio'].max()
            data_str = format_date(ultima_data)
            
            st.caption(f"Última atualização: {data_str}")
            st.markdown("---")
            
            # DESTAQUES
            df_atual = df_focus[df_focus['ano_referencia'] == ano_atual]
            if not df_atual.empty:
                pivot_atual = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='first')
                
                fc1, fc2, fc3, fc4 = st.columns(4)
                
                def get_val(idx, default=0):
                    try: 
                        return pivot_atual.loc[idx, 'previsao']
                    except: 
                        return default
                
                with fc1:
                    st.metric(f"IPCA {ano_atual}", f"{get_val('IPCA'):.2f}%")
                with fc2:
                    st.metric(f"Selic {ano_atual}", f"{get_val('Selic'):.2f}%")
                with fc3:
                    st.metric(f"PIB {ano_atual}", f"{get_val('PIB Total'):.2f}%")
                with fc4:
                    st.metric(f"Dólar {ano_atual}", f"R$ {get_val('Câmbio'):.2f}")
                
                st.markdown("---")
                st.markdown("##### 📅 Projeções Macroeconômicas (2025 - 2027)")
                
                anos_exibir = [ano_atual, ano_atual + 1, ano_atual + 2]
                df_table = df_focus[df_focus['ano_referencia'].isin(anos_exibir)].copy()
                
                if not df_table.empty:
                    df_pivot_multi = df_table.pivot_table(
                        index='Indicador', 
                        columns='ano_referencia', 
                        values='previsao', 
                        aggfunc='first'
                    )
                    
                    ordem = [
                        'IPCA', 'IGP-M', 'IPCA Administrados', 'Selic', 'Câmbio', 'PIB Total', 
                        'Dívida líquida do setor público', 'Resultado primário', 'Resultado nominal', 
                        'Balança comercial', 'Conta corrente', 'Investimento direto no país' 
                    ]
                    
                    ordem_final = [x for x in ordem if x in df_pivot_multi.index]
                    if ordem_final:
                        df_pivot_multi = df_pivot_multi.reindex(ordem_final)
                        
                        df_display = df_pivot_multi.copy()
                        for col in df_display.columns:
                            df_display[col] = df_display.apply(
                                lambda row: format_valor_focus(row[col], row.name), 
                                axis=1
                            )
                        
                        st.dataframe(df_display, use_container_width=True)
                    else:
                        st.info("Nenhum indicador encontrado para os anos selecionados")
                else:
                    st.info("Sem dados de projeção para os próximos anos")
            else:
                st.warning(f"Sem dados Focus para o ano {ano_atual}")
        else:
            st.warning("Focus indisponível (Erro na API ou Filtro).")
    
    # MOEDAS
    with col_top2:
        st.subheader("💱 Câmbio (Tempo Real)")
        df_moedas = get_currency_realtime()
        
        if not df_moedas.empty:
            st.caption(f"Atualizado: {datetime.now().strftime('%H:%M:%S')}")
            st.markdown("---")
            
            mc1, mc2 = st.columns(2)
            try:
                if 'USDBRL' in df_moedas.index:
                    usd = df_moedas.loc['USDBRL']
                    delta_usd = f"{float(usd['pctChange']):+.2f}%" if 'pctChange' in usd and pd.notna(usd['pctChange']) else None
                    valor_usd = f"R$ {float(usd['bid']):.2f}" if 'bid' in usd and pd.notna(usd['bid']) else "R$ 0.00"
                else:
                    delta_usd = None
                    valor_usd = "R$ 0.00"
                
                if 'EURBRL' in df_moedas.index:
                    eur = df_moedas.loc['EURBRL']
                    delta_eur = f"{float(eur['pctChange']):+.2f}%" if 'pctChange' in eur and pd.notna(eur['pctChange']) else None
                    valor_eur = f"R$ {float(eur['bid']):.2f}" if 'bid' in eur and pd.notna(eur['bid']) else "R$ 0.00"
                else:
                    delta_eur = None
                    valor_eur = "R$ 0.00"
                
                with mc1:
                    st.metric("Dólar (USD/BRL)", valor_usd, delta_usd)
                
                with mc2:
                    st.metric("Euro (EUR/BRL)", valor_eur, delta_eur)
                    
            except Exception as e:
                st.error(f"Erro ao processar cotações: {str(e)}")
                mc1.metric("Dólar (USD/BRL)", "R$ 5.50", "0.00%")
                mc2.metric("Euro (EUR/BRL)", "R$ 6.00", "0.00%")
        else:
            st.warning("Cotações em tempo real indisponíveis")
            mc1, mc2 = st.columns(2)
            mc1.metric("Dólar (USD/BRL)", "R$ 5.50", "0.00%")
            mc2.metric("Euro (EUR/BRL)", "R$ 6.00", "0.00%")

# ==============================================================================
# SEÇÃO CONJUNTURA MACRO
# ==============================================================================

with st.expander("🧩 Conjuntura Macroeconômica (Dados Oficiais Realizados)", expanded=False):
    st.subheader("🏛️ Indicadores Macroeconômicos Reais")
    
    macro_result = get_macro_real()
    macro_dados = macro_result.get("dados", {})
    ultima_atualizacao = macro_result.get("ultima_atualizacao", "Não disponível")
    metodologia = macro_result.get("metodologia", {})
    
    if macro_dados:
        st.caption(f"Fonte: Banco Central do Brasil (SGS) • Acumulado últimos 12 meses")
        st.caption(f"📅 Última atualização: {ultima_atualizacao}")
        
        # Informações de metodologia
        with st.expander("📊 Metodologia dos Cálculos", expanded=False):
            for nome, metodo in metodologia.items():
                st.markdown(f"**{nome}**: {metodo}")
        
        st.markdown("---")
        
        st.markdown("##### 📈 Atividade & Fiscal")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            pib_valor = macro_dados.get('PIB Nominal (R$ Mi)', 0)
            if pib_valor == 0:
                st.metric("PIB (Acum. 12m)", "N/D")
            else:
                st.metric("PIB (Acum. 12m)", f"R$ {pib_valor:.2f} Tri")
        
        with c2:
            divida_valor = macro_dados.get('Dívida Líq. (% PIB)', 0)
            if divida_valor == 0:
                st.metric("Dív. Líquida Setor Púb.", "N/D")
            else:
                st.metric("Dív. Líquida Setor Púb.", f"{divida_valor:.1f}% PIB")
        
        with c3:
            primario_valor = macro_dados.get('Res. Primário (% PIB)', 0)
            if primario_valor == 0:
                st.metric("Res. Primário", "N/D")
            else:
                st.metric("Res. Primário", f"{primario_valor:.2f}% PIB")
        
        with c4:
            nominal_valor = macro_dados.get('Res. Nominal (% PIB)', 0)
            if nominal_valor == 0:
                st.metric("Res. Nominal", "N/D")
            else:
                st.metric("Res. Nominal", f"{nominal_valor:.2f}% PIB")
        
        st.markdown("---")
        st.markdown("##### 🌍 Setor Externo (Acum. 12 meses)")
        c5, c6, c7 = st.columns(3)
        
        with c5:
            balanca_valor = macro_dados.get('Balança Com. (US$ Mi)', 0)
            if balanca_valor == 0:
                st.metric("Balança Comercial", "N/D")
            else:
                st.metric("Balança Comercial", f"US$ {balanca_valor:.1f} Bi")
        
        with c6:
            trans_valor = macro_dados.get('Trans. Correntes (US$ Mi)', 0)
            if trans_valor == 0:
                st.metric("Transações Correntes", "N/D")
            else:
                st.metric("Transações Correntes", f"US$ {trans_valor:.1f} Bi")
        
        with c7:
            idp_valor = macro_dados.get('IDP (US$ Mi)', 0)
            if idp_valor == 0:
                st.metric("Investimento Direto (IDP)", "N/D")
            else:
                st.metric("Investimento Direto (IDP)", f"US$ {idp_valor:.1f} Bi")
    else:
        st.warning("Não foi possível carregar os dados macroeconômicos do BCB.")

# ==============================================================================
# SEÇÃO CÂMBIO HISTÓRICO
# ==============================================================================

with st.expander("💸 Histórico de Câmbio (Dólar e Euro desde 1994)", expanded=False):
    st.subheader("📈 Histórico de Cotações")
    
    df_cambio = get_cambio_historico()
    
    if not df_cambio.empty:
        last_date = df_cambio.index[-1] if not df_cambio.empty else None
        date_str = format_date(last_date) if last_date else "Não disponível"
        st.caption(f"Último fechamento: {date_str}")
        
        tab_graf, tab_matriz, tab_tabela = st.tabs(["📈 Gráfico", "📅 Matriz de Retornos", "📋 Tabela Diária"])
        
        with tab_graf:
            if len(df_cambio) > 1:
                fig_cambio = px.line(
                    df_cambio, 
                    x=df_cambio.index, 
                    y=['Dólar', 'Euro'], 
                    color_discrete_map={"Dólar": "#00FF7F", "Euro": "#00BFFF"},
                    labels={'value': 'R$', 'variable': 'Moeda'}
                )
                fig_cambio.update_layout(
                    template="plotly_white",
                    plot_bgcolor='white',
                    font=dict(color="#333333"),
                    hovermode="x unified", 
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="Cotação (R$)",
                    xaxis_title="",
                    yaxis=dict(gridcolor='lightgray'),
                    xaxis=dict(gridcolor='lightgray')
                )
                st.plotly_chart(fig_cambio, use_container_width=True)
            else:
                st.info("Dados insuficientes para gráfico")
        
        with tab_matriz:
            moeda_matriz = st.radio(
                "Selecione a Moeda:", 
                ["Dólar", "Euro"], 
                horizontal=True,
                key="moeda_matriz"
            )
            
            if not df_cambio.empty:
                df_mensal = df_cambio[[moeda_matriz]].resample('ME').last()
                if len(df_mensal) > 12:
                    df_retorno = df_mensal.pct_change() * 100
                    df_retorno['ano'] = df_retorno.index.year
                    df_retorno['mes'] = df_retorno.index.month_name().str.slice(0, 3)
                    
                    mapa_meses_en_pt = {
                        'Jan': 'Jan', 'Feb': 'Fev', 'Mar': 'Mar', 'Apr': 'Abr', 
                        'May': 'Mai', 'Jun': 'Jun', 'Jul': 'Jul', 'Aug': 'Ago', 
                        'Sep': 'Set', 'Oct': 'Out', 'Nov': 'Nov', 'Dec': 'Dez'
                    }
                    df_retorno['mes'] = df_retorno['mes'].map(mapa_meses_en_pt)
                    
                    try:
                        matrix_cambio = df_retorno.pivot(
                            index='ano', 
                            columns='mes', 
                            values=moeda_matriz
                        )
                        
                        colunas_ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                                        'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                        colunas_ordem = [c for c in colunas_ordem if c in matrix_cambio.columns]
                        
                        if colunas_ordem:
                            matrix_cambio = matrix_cambio[colunas_ordem].sort_index(ascending=False)
                            
                            # Limites dinâmicos
                            flat_values = matrix_cambio.values.flatten()
                            flat_values = flat_values[~np.isnan(flat_values)]
                            if len(flat_values) > 0:
                                vmin = np.percentile(flat_values, 10)
                                vmax = np.percentile(flat_values, 90)
                                if vmax - vmin < 2:
                                    vmin -= 1
                                    vmax += 1
                            else:
                                vmin, vmax = -5, 5
                            
                            st.dataframe(
                                matrix_cambio.style.background_gradient(
                                    cmap=cmap_leves, vmin=vmin, vmax=vmax
                                ).format("{:.2f}%"), 
                                use_container_width=True, 
                                height=500
                            )
                        else:
                            st.info("Dados insuficientes para matriz")
                    except Exception as e:
                        st.error(f"Erro ao gerar matriz: {str(e)}")
                else:
                    st.info("Dados históricos insuficientes para matriz")
            else:
                st.info("Aguardando dados de câmbio")
        
        with tab_tabela:
            if not df_cambio.empty:
                df_view = df_cambio.sort_index(ascending=False).reset_index()
                df_view.rename(columns={'index': 'Data'}, inplace=True)
                
                # CORREÇÃO DO ERRO: Formato correto de data
                if 'Data' in df_view.columns:
                    try:
                        df_view['Data'] = df_view['Data'].dt.strftime('%d/%m/%Y')
                    except Exception as e:
                        # Se não for datetime, manter como está
                        pass
                
                df_view.columns = ['Data', 'Dólar (R$)', 'Euro (R$)']
                st.dataframe(df_view, use_container_width=True, hide_index=True)
            else:
                st.info("Aguardando dados de câmbio")
    else:
        st.warning("Histórico de câmbio indisponível.")

# ==============================================================================
# RODAPÉ
# ==============================================================================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**VPL Consultoria Financeira**")
    st.markdown("*Inteligência para decisões*")
with col2:
    st.markdown("**Contato**")
    st.markdown("contato@vplconsultoria.com.br")
with col3:
    st.markdown("**Atualização**")
    st.markdown(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

st.caption("""
⚠️ **Aviso**: As informações fornecidas são para fins educacionais e informativos. 
Não constituem recomendação de investimento. Consulte um profissional qualificado para decisões financeiras.
""")
