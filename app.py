import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, datetime
import requests
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
import urllib3

# Desabilita avisos de SSL (Necessário para APIs do Governo)
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
</style>
""", unsafe_allow_html=True)

cores_leves = ["#FFB3B3", "#FFFFFF", "#B3FFB3"] # Vermelho Suave, Branco, Verde Suave
cmap_leves = LinearSegmentedColormap.from_list("pastel_rdylgn", cores_leves)

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

# ADICIONE ESTA FUNÇÃO AQUI - SOMENTE UMA VEZ!
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

# --- FUNÇÕES DE CARGA DE DADOS COM TRATAMENTO DE ERRO MELHORADO ---

@st.cache_data(show_spinner="Carregando dados do IBGE...")
def get_sidra_data(table_code, variable_code, test_mode=False):
    """Função para obter dados do IBGE com suporte a test_mode"""
    if test_mode:
        # Retorna dataframe vazio para teste rápido
        return pd.DataFrame()
    
    try:
        with st.spinner(f"Buscando dados do IBGE (Tabela {table_code})..."):
            dados_raw = sidrapy.get_table(
                table_code=table_code, territorial_level="1", 
                ibge_territorial_code="all", variable=variable_code, 
                period="last 360"
            )
        
        if dados_raw.empty: 
            st.warning(f"Tabela {table_code} do IBGE retornou vazia")
            return pd.DataFrame()
        
        df = dados_raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['D2C'].str.slice(0, 4)
        
        return processar_dataframe_comum(df)
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados do IBGE: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Carregando dados do BCB...")
def get_bcb_data(codigo_serie, serie_nome="", test_mode=False):
    """Função para obter dados do BCB com suporte a test_mode"""
    if test_mode:
        # Retorna dataframe vazio para teste rápido
        return pd.DataFrame()
    
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        with st.spinner(f"Buscando série {codigo_serie} do BCB..."):
            response = requests.get(url, headers=headers, verify=False, timeout=15)
            response.raise_for_status()
        
        df = pd.DataFrame(response.json())
        
        if df.empty:
            st.warning(f"Série {codigo_serie} do BCB retornou vazia")
            return pd.DataFrame()
        
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        
        return processar_dataframe_comum(df)
        
    except requests.exceptions.Timeout:
        st.error(f"⏰ Timeout ao acessar série {codigo_serie} do BCB")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"🔗 Erro de conexão com BCB (Série {codigo_serie}): {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erro inesperado no BCB: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Carregando expectativas de mercado...")
def get_focus_data(test_mode=False):
    """Função para obter dados do Focus com suporte a test_mode"""
    if test_mode:
        # Retorna dataframe vazio para teste rápido
        return pd.DataFrame()
    
    try:
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=5000&$orderby=Data%20desc&$format=json"
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        with st.spinner("Buscando Boletim Focus..."):
            response = requests.get(url, headers=headers, verify=False, timeout=15)
            response.raise_for_status()
            data_json = response.json()
        
        df = pd.DataFrame(data_json['value'])
        
        if df.empty:
            st.warning("Boletim Focus retornou vazio")
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
        
    except requests.exceptions.Timeout:
        st.error("⏰ Timeout ao acessar Boletim Focus")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"🔗 Erro de conexão com Focus: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erro inesperado no Focus: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner="Atualizando cotações...")
def get_currency_realtime(test_mode=False):
    """Função para obter cotações em tempo real com suporte a test_mode"""
    if test_mode:
        # Retorna dataframe vazio para teste rápido
        return pd.DataFrame()
    
    try:
        tickers = ["USDBRL=X", "EURBRL=X"]
        dados = {}
        
        with st.spinner("Consultando cotações em tempo real..."):
            for t in tickers:
                try:
                    ticker_obj = yf.Ticker(t)
                    info = ticker_obj.fast_info
                    preco_atual = info.get('last_price', 0)
                    fechamento_anterior = info.get('previous_close', preco_atual)
                    variacao = ((preco_atual - fechamento_anterior) / fechamento_anterior) * 100 if fechamento_anterior != 0 else 0
                    key = t.replace("=X", "") 
                    dados[key] = {'bid': preco_atual, 'pctChange': variacao}
                except Exception as e:
                    st.warning(f"Erro ao buscar {t}: {str(e)}")
        
        if not dados:
            st.warning("Não foi possível obter cotações em tempo real")
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(dados, orient='index')
        return df
        
    except Exception as e:
        st.error(f"❌ Erro nas cotações em tempo real: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner="Carregando histórico de câmbio...")
def get_cambio_historico(test_mode=False):
    """Função para obter histórico de câmbio com suporte a test_mode"""
    if test_mode:
        # Retorna dataframe vazio para teste rápido
        return pd.DataFrame()
    
    try:
        with st.spinner("Baixando histórico de câmbio (pode levar alguns segundos)..."):
            df = yf.download(["USDBRL=X", "EURBRL=X"], start="1994-07-01", progress=False)
        
        if df.empty: 
            st.warning("Histórico de câmbio retornou vazio")
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

@st.cache_data(ttl=3600, show_spinner="Carregando dados macroeconômicos...")
def get_macro_real(test_mode=False):
    """Função para obter dados macroeconômicos com suporte a test_mode"""
    if test_mode:
        # Retorna dicionário vazio para teste rápido
        return {}
    
    series = {
        'PIB (R$ Bi)': 4382,
        'Dívida Líq. (% PIB)': 4513,
        'Res. Primário (% PIB)': 5362,
        'Res. Nominal (% PIB)': 5360,
        'Balança Com. (US$ Mi)': 22707,
        'Trans. Correntes (US$ Mi)': 22724,
        'IDP (US$ Mi)': 22885
    }
    
    resultados = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        for nome, codigo in series.items():
            try:
                url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/13?formato=json"
                resp = requests.get(url, headers=headers, verify=False, timeout=10)
                df = pd.DataFrame(resp.json())
                
                if df.empty:
                    st.warning(f"Série {nome} ({codigo}) retornou vazia")
                    continue
                
                df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                
                if nome == 'PIB (R$ Bi)':
                    valor_final = df['valor'].iloc[-1] / 1_000_000 
                elif 'Balança' in nome or 'Trans.' in nome or 'IDP' in nome:
                    valor_final = df['valor'].iloc[-12:].sum() / 1_000
                elif 'Primário' in nome or 'Nominal' in nome:
                    valor_final = df['valor'].iloc[-1] * -1
                else:
                    valor_final = df['valor'].iloc[-1]
                
                resultados[nome] = valor_final
                
            except Exception as e:
                st.warning(f"Erro na série {nome}: {str(e)}")
                continue
        
        return resultados
        
    except Exception as e:
        st.error(f"❌ Erro geral nos dados macroeconômicos: {str(e)}")
        return {}

def processar_dataframe_comum(df):
    if df.empty: 
        return df
    
    df = df.sort_values('data_date', ascending=True)
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['ano'] = df['data_date'].dt.strftime('%Y')
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12, min_periods=12).apply(np.prod, raw=True) - 1) * 100
    
    return df.sort_values('data_date', ascending=False)

def calcular_correcao(df, valor, data_ini_code, data_fim_code):
    is_reverso = data_ini_code > data_fim_code
    if is_reverso:
        periodo_inicio, periodo_fim = data_fim_code, data_ini_code
    else:
        periodo_inicio, periodo_fim = data_ini_code, data_fim_code
    
    mask = (df['D2C'] >= periodo_inicio) & (df['D2C'] <= periodo_fim)
    df_periodo = df.loc[mask].copy()
    
    if df_periodo.empty:
        return None, "Período sem dados suficientes."
    
    fator_acumulado = df_periodo['fator'].prod()
    valor_final = valor / fator_acumulado if is_reverso else valor * fator_acumulado
    pct_total = (fator_acumulado - 1) * 100
    
    return {
        'valor_final': valor_final, 
        'percentual': pct_total, 
        'fator': fator_acumulado, 
        'is_reverso': is_reverso,
        'periodo_inicio': periodo_inicio,
        'periodo_fim': periodo_fim
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

tipo_indice = st.sidebar.selectbox(
    "Selecione o Indicador",
    ["IPCA (Inflação Oficial)", "INPC (Salários)", "IGP-M (Aluguéis)", "SELIC (Taxa Básica)", "CDI (Investimentos)"]
)

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

# 🔧 STATUS DAS APIs (ATUALIZADO)
st.sidebar.subheader("🔧 Status das APIs")
try:
    # Teste simplificado das APIs
    api_status = {}
    
    # IBGE
    try:
        df_test = get_sidra_data("1737", "63", test_mode=False)
        api_status["IBGE"] = not df_test.empty
    except:
        api_status["IBGE"] = False
    
    # BCB
    try:
        df_test = get_bcb_data("4390", test_mode=False)
        api_status["BCB"] = not df_test.empty
    except:
        api_status["BCB"] = False
    
    # Focus
    try:
        df_test = get_focus_data(test_mode=False)
        api_status["Focus"] = not df_test.empty
    except:
        api_status["Focus"] = False
    
    # Câmbio
    try:
        df_test = get_currency_realtime(test_mode=False)
        api_status["Câmbio"] = not df_test.empty
    except:
        api_status["Câmbio"] = False
    
    # Exibir status
    for api_name, status in api_status.items():
        icon = "🟢" if status else "🔴"
        st.sidebar.markdown(f"{icon} {api_name}")
        
except Exception as e:
    st.sidebar.info("Status das APIs não disponível")

st.sidebar.divider()

# 🧮 CALCULADORA
st.sidebar.subheader("🧮 Calculadora")
valor_input = st.sidebar.number_input("Valor (R$)", value=1000.00, step=100.00, format="%.2f")

# Carregar dados do índice selecionado
with st.spinner(f"Carregando dados do {tipo_indice.split()[0]}..."):
    if "IPCA" in tipo_indice:
        df = get_sidra_data("1737", "63")
        cor_tema = "#00BFFF"
        fonte = "IBGE/Sidra"
    elif "INPC" in tipo_indice:
        df = get_sidra_data("1736", "44")
        cor_tema = "#00FF7F"
        fonte = "IBGE/Sidra"
    elif "IGP-M" in tipo_indice:
        df = get_bcb_data("189", "IGP-M")
        cor_tema = "#FF6347"
        fonte = "BCB/SGS"
    elif "SELIC" in tipo_indice:
        df = get_bcb_data("4390", "SELIC")
        cor_tema = "#FFD700"
        fonte = "BCB/SGS"
    elif "CDI" in tipo_indice:
        df = get_bcb_data("4391", "CDI")
        cor_tema = "#9370DB"
        fonte = "BCB/SGS"

# Tratamento de erro explícito
if df.empty:
    error_container = st.container()
    with error_container:
        st.error(f"""
        ⚠️ **Erro ao carregar dados do índice principal**
        
        **Fonte:** {fonte}
        **Índice:** {tipo_indice.split()[0]}
        
        **Possíveis causas:**
        1. Conexão com {fonte} interrompida
        2. API temporariamente indisponível
        3. Limite de requisições excedido
        4. Dados em manutenção
        
        **Soluções:**
        - Tente recarregar a página (F5)
        - Verifique sua conexão com a internet
        - Tente novamente em alguns minutos
        - Use os dados históricos já carregados
        
        **Status atual:** ❌ Indisponível
        """)
    
    # Permite continuar com dados limitados
    st.warning("Usando dados limitados. Algumas funcionalidades podem estar indisponíveis.")
    
    # Cria um dataframe vazio com estrutura mínima
    df = pd.DataFrame(columns=['data_date', 'valor', 'acum_ano', 'acum_12m', 'data_fmt', 'D2C', 'ano', 'mes_nome'])
else:
    # Mostra informação de sucesso
    update_info = get_data_update_info(df, fonte)
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
        res, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
        
        if erro:
            st.sidebar.error(f"❌ {erro}")
        else:
            st.sidebar.divider()
            nome_indice = tipo_indice.split()[0]
            tipo_op = "Rendimento" if nome_indice in ["SELIC", "CDI"] else "Correção"
            texto_op = "Descapitalização" if res['is_reverso'] else f"{tipo_op} ({nome_indice})"
            
            st.sidebar.markdown(f"<small>{texto_op}</small>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<h2 style='color: {cor_tema}; margin:0;'>R$ {res['valor_final']:,.2f}</h2>", unsafe_allow_html=True)
            
            # Card de resultados
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"""
            <div class='metric-card'>
            <small>Total Período</small><br>
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

# ==============================================================================
# PAINEL PRINCIPAL
# ==============================================================================

# CABEÇALHO PRINCIPAL
st.title(f"📊 Painel: {tipo_indice.split()[0]}")
st.caption(f"Fonte: {fonte} • Atualização: {get_data_update_info(df, '')}")

if not df.empty:
    # KPIs PRINCIPAIS
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        try:
            valor_mes = df.iloc[0]['valor'] if not df.empty else 0
            st.metric("Taxa do Mês", f"{valor_mes:.2f}%")
        except:
            st.metric("Taxa do Mês", "N/D")
    
    with kpi2:
        try:
            acum_12m = df.iloc[0]['acum_12m'] if not df.empty and 'acum_12m' in df.columns else 0
            st.metric("Acumulado 12 Meses", f"{acum_12m:.2f}%")
        except:
            st.metric("Acumulado 12 Meses", "N/D")
    
    with kpi3:
        try:
            acum_ano = df.iloc[0]['acum_ano'] if not df.empty and 'acum_ano' in df.columns else 0
            st.metric("Acumulado Ano (YTD)", f"{acum_ano:.2f}%")
        except:
            st.metric("Acumulado Ano (YTD)", "N/D")
    
    with kpi4:
        try:
            inicio_serie = df['ano'].min() if not df.empty else "N/D"
            st.metric("Início da Série", inicio_serie)
        except:
            st.metric("Início da Série", "N/D")
    
    # ABA DO GRÁFICO
    tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Matriz de Calor", "📋 Tabela Detalhada"])
    
    with tab1:
        if not df.empty:
            df_chart = df.dropna(subset=['acum_12m']).sort_values('data_date')
            
            # Filtro para séries mais longas
            if any(idx in tipo_indice for idx in ["IGP-M", "SELIC", "CDI"]):
                df_chart = df_chart[df_chart['ano'].astype(int) >= 2000]
            
            if not df_chart.empty:
                fig = px.line(
                    df_chart, 
                    x='data_date', 
                    y='acum_12m', 
                    title=f"Histórico 12 Meses - {tipo_indice.split()[0]}",
                    labels={'acum_12m': 'Acumulado 12 meses (%)', 'data_date': 'Data'}
                )
                fig.update_traces(line_color=cor_tema, line_width=3)
                fig.update_layout(
                    template="plotly_dark", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#E0E0E0"), 
                    hovermode="x unified", 
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="%",
                    xaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dados insuficientes para gerar o gráfico")
        else:
            st.info("Aguardando dados para exibir o gráfico")
    
    with tab2:
        if not df.empty and 'mes_nome' in df.columns and 'ano' in df.columns:
            try:
                matrix = df.pivot(index='ano', columns='mes_nome', values='valor')
                ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                
                # Mantém apenas colunas existentes
                ordem = [col for col in ordem if col in matrix.columns]
                if ordem:
                    matrix = matrix[ordem].sort_index(ascending=False)
                    
                    # Define limites dinâmicos para o gradiente
                    vmin = matrix.min().min() if not matrix.empty else -1.5
                    vmax = matrix.max().max() if not matrix.empty else 1.5
                    
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
            csv_principal = df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].to_csv(index=False).encode('utf-8')
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    "📥 Baixar CSV", 
                    csv_principal, 
                    f"{tipo_indice.split()[0]}_historico.csv", 
                    "text/csv",
                    use_container_width=True
                )
            
            # Tabela de dados
            df_display = df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].copy()
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
                        
                        # Formatação - USANDO A FUNÇÃO format_valor_focus QUE JÁ ESTÁ DEFINIDA
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
            st.info("Tentando reconectar...")
    
    # MOEDAS
    with col_top2:
        st.subheader("💱 Câmbio (Tempo Real)")
        df_moedas = get_currency_realtime()
        
        if not df_moedas.empty:
            st.caption(f"Atualizado: {datetime.now().strftime('%H:%M:%S')}")
            st.markdown("---")
            
            mc1, mc2 = st.columns(2)
            try:
                usd = df_moedas.loc['USDBRL']
                eur = df_moedas.loc['EURBRL']
                
                with mc1:
                    delta_usd = f"{float(usd['pctChange']):+.2f}%" if 'pctChange' in usd else None
                    st.metric("Dólar (USD/BRL)", f"R$ {float(usd['bid']):.2f}", delta_usd)
                
                with mc2:
                    delta_eur = f"{float(eur['pctChange']):+.2f}%" if 'pctChange' in eur else None
                    st.metric("Euro (EUR/BRL)", f"R$ {float(eur['bid']):.2f}", delta_eur)
                    
            except Exception as e:
                st.error(f"Erro ao processar cotações: {str(e)}")
                st.info("Mostrando últimos valores disponíveis")
                st.dataframe(df_moedas, use_container_width=True)
        else:
            st.warning("Cotações em tempo real indisponíveis")
            st.info("Verificando conexão com Yahoo Finance...")

# ==============================================================================
# SEÇÃO CONJUNTURA MACRO
# ==============================================================================

with st.expander("🧩 Conjuntura Macroeconômica (Dados Oficiais Realizados)", expanded=False):
    st.subheader("🏛️ Indicadores Macroeconômicos Reais")
    
    with st.spinner("Carregando dados do BCB..."):
        macro_dados = get_macro_real()
    
    if macro_dados:
        st.caption("Fonte: Banco Central do Brasil (SGS) • Acumulado últimos 12 meses")
        st.markdown("---")
        
        st.markdown("##### 📈 Atividade & Fiscal")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.metric("PIB (Acum. 12m)", f"R$ {macro_dados.get('PIB (R$ Bi)', 0):.2f} Tri")
        with c2:
            st.metric("Dív. Líquida Setor Púb.", f"{macro_dados.get('Dívida Líq. (% PIB)', 0):.1f}% PIB")
        with c3:
            st.metric("Res. Primário", f"{macro_dados.get('Res. Primário (% PIB)', 0):.2f}% PIB")
        with c4:
            st.metric("Res. Nominal", f"{macro_dados.get('Res. Nominal (% PIB)', 0):.2f}% PIB")
        
        st.markdown("---")
        st.markdown("##### 🌍 Setor Externo")
        c5, c6, c7 = st.columns(3)
        
        with c5:
            st.metric("Balança Comercial", f"US$ {macro_dados.get('Balança Com. (US$ Mi)', 0):.1f} Bi")
        with c6:
            st.metric("Transações Correntes", f"US$ {macro_dados.get('Trans. Correntes (US$ Mi)', 0):.1f} Bi")
        with c7:
            st.metric("Investimento Direto (IDP)", f"US$ {macro_dados.get('IDP (US$ Mi)', 0):.1f} Bi")
    else:
        st.warning("Não foi possível carregar os dados macroeconômicos do BCB.")
        st.info("""
        Possíveis causas:
        1. API do BCB temporariamente indisponível
        2. Limite de requisições excedido
        3. Problemas de conectividade
        
        Tente novamente em alguns minutos.
        """)

# ==============================================================================
# SEÇÃO CÂMBIO HISTÓRICO
# ==============================================================================

with st.expander("💸 Histórico de Câmbio (Dólar e Euro desde 1994)", expanded=False):
    st.subheader("📈 Histórico de Cotações")
    
    with st.spinner("Carregando histórico de câmbio..."):
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
                    template="plotly_dark", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#E0E0E0"), 
                    hovermode="x unified", 
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="Cotação (R$)",
                    xaxis_title=""
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
                if len(df_mensal) > 12:  # Precisa de dados suficientes
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
                            st.dataframe(
                                matrix_cambio.style.background_gradient(
                                    cmap=cmap_leves, vmin=-5, vmax=5
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
                df_view['Data'] = df_view['Data'].dt.strftime('%d/%m/%Y')
                df_view.columns = ['Data', 'Dólar (R$)', 'Euro (R$)']
                st.dataframe(df_view, use_container_width=True, hide_index=True)
            else:
                st.info("Aguardando dados de câmbio")
    else:
        st.warning("Histórico de câmbio indisponível.")
        st.info("Verificando conexão com Yahoo Finance...")

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
