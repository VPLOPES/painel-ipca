import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
from datetime import date, datetime
import time
import functools
from typing import Dict, Tuple, Optional, Any

# =============================================================================
# 1. CAMADA DE CONFIGURAÇÃO E ESTILO
# =============================================================================

st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Otimizado
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

    /* Box de Insights */
    .insight-box {
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .insight-positive { background-color: #e6fffa; border-color: #38b2ac; color: #234e52; }
    .insight-negative { background-color: #fff5f5; border-color: #f56565; color: #742a2a; }
    .insight-neutral { background-color: #ebf8ff; border-color: #4299e1; color: #2c5282; }
</style>
""", unsafe_allow_html=True)

class Config:
    """Centraliza constantes e configurações"""
    # URLs
    BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados"
    FOCUS_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
    
    # Configuração de Requisição (Segurança e Timeout)
    HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    TIMEOUT = 10 # segundos
    
    # Metadados dos Índices
    INDICES = {
        "IPCA (Inflação Oficial)": {"source": "sidra", "table": "1737", "variable": "63", "code": "IPCA", "color": "#00D9FF"},
        "INPC (Salários)": {"source": "sidra", "table": "1736", "variable": "44", "code": "INPC", "color": "#00FFA3"},
        "IGP-M (Aluguéis)": {"source": "bcb", "bcb_code": "189", "code": "IGP-M", "color": "#FF6B6B"},
        "SELIC (Taxa Básica)": {"source": "bcb", "bcb_code": "4390", "code": "SELIC", "color": "#FFD93D"},
        "CDI (Investimentos)": {"source": "bcb", "bcb_code": "4391", "code": "CDI", "color": "#A8E6CF"}
    }
    
    # Códigos SGS (BCB)
    SERIES_MACRO = {
        'PIB (R$ Bi)': 4382, 'Dívida Líq. (% PIB)': 4513,
        'Res. Primário (% PIB)': 5793, 'Res. Nominal (% PIB)': 5811,
        'Balança Com. (US$ Mi)': 22707, 'Trans. Correntes (US$ Mi)': 22724,
        'IDP (US$ Mi)': 22885
    }

    # Cores
    CORES_MATRIZ = ["#FF6B6B", "#FFFFFF", "#4ECDC4"]
    CMAP_CUSTOM = LinearSegmentedColormap.from_list("custom", CORES_MATRIZ)

# =============================================================================
# 2. CAMADA DE UTILITÁRIOS (Helpers & Decorators)
# =============================================================================

def retry_request(max_attempts=3, delay=2.0):
    """Decorator robusto para retry com backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    time.sleep(delay)
                except Exception as e:
                    # Erros de lógica/parsing não devem ter retry
                    raise e
            # Se falhar após tentativas, retorna DataFrame vazio e loga
            st.warning(f"Falha na conexão externa após {max_attempts} tentativas: {str(last_exception)}")
            return pd.DataFrame() 
        return wrapper
    return decorator

def hex_to_rgba(hex_color: str, opacity: float = 0.2) -> str:
    """Converte HEX para RGBA para compatibilidade com Plotly"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {opacity})"
    return hex_color

# =============================================================================
# 3. CAMADA DE DADOS (Data Fetching & Cleaning)
# =============================================================================

@st.cache_data(ttl=86400) # Cache longo (24h) para dados estruturais (Sidra/IBGE)
@retry_request()
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
    except Exception as e:
        st.error(f"Erro ao processar dados do SIDRA (IBGE): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400) # Cache longo para histórico BCB
@retry_request()
def get_bcb_data(codigo_serie: str) -> pd.DataFrame:
    url = Config.BCB_BASE.format(codigo_serie) + "?formato=json"
    # SSL Verify=True é o padrão. Removido o disable_warnings.
    resp = requests.get(url, headers=Config.HEADERS, timeout=Config.TIMEOUT)
    resp.raise_for_status()
    
    data = resp.json()
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
    df['D2C'] = df['data_date'].dt.strftime('%Y%m')
    df['ano'] = df['data_date'].dt.strftime('%Y')
    
    return df.dropna(subset=['valor', 'data_date'])

@st.cache_data(ttl=3600) # Cache médio (1h) para Focus (atualiza semanalmente)
@retry_request()
def get_focus_data() -> pd.DataFrame:
    url = f"{Config.FOCUS_URL}?$top=5000&$orderby=Data%20desc&$format=json"
    resp = requests.get(url, headers=Config.HEADERS, timeout=Config.TIMEOUT)
    resp.raise_for_status()
    
    data = resp.json().get('value', [])
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Filtragem e Limpeza
    indicadores_interesse = [
        'IPCA', 'PIB Total', 'Selic', 'Câmbio', 'IGP-M',
        'Balança comercial', 'Investimento direto no país', 
        'Dívida líquida do setor público', 'Resultado primário', 'Resultado nominal'
    ]
    df = df[df['Indicador'].isin(indicadores_interesse)]
    
    renomear = {'Data': 'data_relatorio', 'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'}
    df = df.rename(columns=renomear)
    
    # Tipagem forte
    df['ano_referencia'] = pd.to_numeric(df['ano_referencia'], errors='coerce')
    df['previsao'] = pd.to_numeric(df['previsao'], errors='coerce')
    df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
    
    # Deduplicação (mantém o relatório mais recente)
    df = df.sort_values('data_relatorio', ascending=False)
    df = df.drop_duplicates(subset=['Indicador', 'ano_referencia'], keep='first')
    
    return df

@st.cache_data(ttl=300) # Cache curto (5 min) para Câmbio Realtime
def get_currency_realtime() -> pd.DataFrame:
    try:
        tickers = {"USDBRL=X": "USDBRL", "EURBRL=X": "EURBRL"}
        dados = {}
        for ticker, nome in tickers.items():
            info = yf.Ticker(ticker).fast_info
            # Validação se a API retornou dados válidos
            if info and hasattr(info, 'last_price'):
                dados[nome] = {
                    'bid': info['last_price'],
                    'pctChange': ((info['last_price'] - info['previous_close']) / info['previous_close']) * 100
                }
            else:
                # Fallback em caso de falha pontual do Yahoo
                dados[nome] = {'bid': 0.0, 'pctChange': 0.0}
        return pd.DataFrame.from_dict(dados, orient='index')
    except Exception as e:
        # Erro silencioso aqui é aceitável, mas logamos no console se necessário
        # print(f"Erro Yahoo Finance: {e}") 
        return pd.DataFrame()

@st.cache_data(ttl=43200) # Cache 12h para histórico longo
def get_cambio_historico() -> pd.DataFrame:
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="2000-01-01", progress=False)['Close']
        if df.empty: return pd.DataFrame()
        
        # Ajuste Fuso Horário
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo').tz_localize(None)
        
        # Filtra futuro (timezone mismatch prevention)
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        return df.ffill()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_macro_real() -> Dict[str, Any]:
    """Retorna dicionário com KPIs e Histórico Macro"""
    resultados = {}
    
    for nome, codigo in Config.SERIES_MACRO.items():
        # Reutiliza a lógica do get_bcb_data, mas aqui pegamos uma janela fixa
        # para garantir performance
        try:
            url = f"{Config.BCB_BASE.format(codigo)}/ultimos/24?formato=json"
            resp = requests.get(url, headers=Config.HEADERS, timeout=Config.TIMEOUT)
            resp.raise_for_status()
            dados = resp.json()
            
            if not dados: continue
            
            df = pd.DataFrame(dados)
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            
            # Lógica de Negócio: Normalização de Unidades
            if 'PIB' in nome and 'Dívida' not in nome and 'Res.' not in nome:
                val = df.iloc[-1]['valor'] / 1_000_000 # R$ Tri
            elif any(x in nome for x in ['Balança', 'Trans.', 'IDP']):
                val = df.iloc[-12:]['valor'].sum() / 1_000 # US$ Bi (Acum 12m)
            elif 'Primário' in nome or 'Nominal' in nome:
                val = df.iloc[-1]['valor'] * -1 # Inversão de sinal (Déficit/Superávit)
            else:
                val = df.iloc[-1]['valor'] # %
                
            resultados[nome] = val
        except Exception:
            # Em macro, se um indicador falhar, não queremos quebrar o loop
            resultados[nome] = None
            
    return resultados

# =============================================================================
# 4. CAMADA DE CÁLCULO E INTELIGÊNCIA (Business Logic)
# =============================================================================

def processar_dataframe_padrao(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza todos os DFs de índices para conter colunas de cálculo"""
    if df.empty: return df
    
    df = df.sort_values('data_date', ascending=True)
    
    # Enriquecimento de Datas
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                 7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    
    # Cálculos Financeiros
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12, min_periods=12).apply(np.prod, raw=True) - 1) * 100
    
    return df.sort_values('data_date', ascending=False)

def calcular_correcao_monetaria(df: pd.DataFrame, valor: float, data_ini: str, data_fim: str) -> Tuple[Dict, Optional[str]]:
    """Motor de cálculo de correção de valores"""
    try:
        is_reverso = int(data_ini) > int(data_fim)
        periodo_inicio = min(data_ini, data_fim)
        periodo_fim = max(data_ini, data_fim)
        
        # Filtra o período exato
        mask = (df['D2C'] >= periodo_inicio) & (df['D2C'] <= periodo_fim)
        df_periodo = df.loc[mask]
        
        if df_periodo.empty:
            return {}, "⚠️ Período selecionado não possui dados históricos suficientes."
        
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
    except Exception as e:
        return {}, f"Erro de cálculo: {str(e)}"

def gerar_diagnostico_economico(df_indice: pd.DataFrame, nome_indice: str) -> Tuple[str, str]:
    """
    Transforma dados em texto (Inteligência Econômica).
    Retorna: (HTML do card, Class CSS)
    """
    if df_indice.empty or len(df_indice) < 13:
        return "Dados insuficientes para diagnóstico.", "insight-neutral"
    
    atual_12m = df_indice.iloc[0]['acum_12m']
    anterior_12m = df_indice.iloc[1]['acum_12m']
    media_historica = df_indice['acum_12m'].mean()
    
    delta = atual_12m - anterior_12m
    
    # Lógica de Diagnóstico (Exemplo Simples)
    if delta > 0.5:
        analise = f"⚠️ **Atenção:** O {nome_indice} apresenta forte aceleração. A taxa em 12 meses subiu de {anterior_12m:.2f}% para {atual_12m:.2f}%, indicando pressão inflacionária acima da tendência de curto prazo."
        classe = "insight-negative"
    elif delta < -0.5:
        analise = f"✅ **Alívio:** O {nome_indice} está desacelerando consistentemente. Caiu de {anterior_12m:.2f}% para {atual_12m:.2f}%, o que pode sinalizar arrefecimento de preços."
        classe = "insight-positive"
    else:
        tendencia = "acima" if atual_12m > media_historica else "abaixo"
        analise = f"ℹ️ **Estabilidade:** O {nome_indice} mostra estabilidade no curto prazo. Atualmente em {atual_12m:.2f}%, opera {tendencia} da sua média histórica ({media_historica:.2f}%)."
        classe = "insight-neutral"
        
    return analise, classe

# =============================================================================
# 5. CAMADA DE INTERFACE (UI & Layout)
# =============================================================================

def render_sidebar():
    """Renderiza Sidebar e retorna configurações selecionadas"""
    try:
        st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
    except:
        st.sidebar.markdown("## 📊 VPL CONSULTORIA")
        
    st.sidebar.header("⚙️ Configurações")
    tipo_indice = st.sidebar.selectbox("Selecione o Indicador", list(Config.INDICES.keys()))
    config = Config.INDICES[tipo_indice]
    
    return tipo_indice, config

def render_calculadora(df: pd.DataFrame, config: Dict):
    """Renderiza a calculadora na sidebar"""
    st.sidebar.divider()
    st.sidebar.subheader("🧮 Calculadora")
    
    valor = st.sidebar.number_input("Valor (R$)", min_value=0.01, value=1000.00, step=100.00)
    
    # Seletores de Data
    lista_anos = sorted(df['ano'].unique(), reverse=True)
    meses_nome = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
    mapa_meses = {m: f"{i:02d}" for i, m in enumerate(meses_nome, 1)}
    
    c1, c2 = st.sidebar.columns(2)
    mi = c1.selectbox("Mês", meses_nome, index=0, key='mi', label_visibility="collapsed")
    ai = c2.selectbox("Ano", lista_anos, index=min(1, len(lista_anos)-1), key='ai', label_visibility="collapsed")
    
    st.sidebar.markdown("⬇️ Para")
    c3, c4 = st.sidebar.columns(2)
    mf = c3.selectbox("Mês", meses_nome, index=11, key='mf', label_visibility="collapsed")
    af = c4.selectbox("Ano", lista_anos, index=0, key='af', label_visibility="collapsed")
    
    if st.sidebar.button("🚀 Calcular", type="primary", use_container_width=True):
        code_ini = f"{ai}{mapa_meses[mi]}"
        code_fim = f"{af}{mapa_meses[mf]}"
        
        res, erro = calcular_correcao_monetaria(df, valor, code_ini, code_fim)
        
        if erro:
            st.sidebar.error(erro)
        else:
            st.sidebar.divider()
            lbl = "Descapitalização" if res['is_reverso'] else "Valor Corrigido"
            val_fmt = f"R$ {res['valor_final']:,.2f}"
            
            st.sidebar.markdown(f"**{lbl} ({config['code']})**")
            st.sidebar.markdown(f"<h1 style='color: {config['color']}; margin:0;'>{val_fmt}</h1>", unsafe_allow_html=True)
            
            col_a, col_b = st.sidebar.columns(2)
            col_a.metric("Variação", f"{res['percentual']:.2f}%")
            col_b.metric("Fator", f"{res['fator']:.4f}")

def main():
    # 1. Sidebar e Carga Inicial
    tipo_indice_sel, config_sel = render_sidebar()
    
    with st.spinner(f"Conectando às fontes de dados ({config_sel['source']})..."):
        if config_sel['source'] == 'sidra':
            df_raw = get_sidra_data(config_sel['table'], config_sel['variable'])
        else:
            df_raw = get_bcb_data(config_sel['bcb_code'])
            
        df_principal = processar_dataframe_padrao(df_raw)
        
    if df_principal.empty:
        st.error("⛔ Falha crítica: Não foi possível carregar os dados do índice selecionado. Verifique a conexão com as APIs governamentais.")
        st.stop()
        
    st.sidebar.success(f"Dados atualizados: {df_principal.iloc[0]['data_fmt']}")
    render_calculadora(df_principal, config_sel)

    # 2. Área Principal - Expander Focus/Câmbio
    with st.expander("🔭 Expectativas (Focus) & Câmbio", expanded=False):
        c1, c2 = st.columns([2, 1])
        
        # Focus
        df_focus = get_focus_data()
        ano_atual = date.today().year
        
        with c1:
            if not df_focus.empty:
                st.markdown("#### 🎯 Meta e Expectativas")
                df_curr = df_focus[df_focus['ano_referencia'] == ano_atual]
                if not df_curr.empty:
                    piv = df_curr.pivot_table(index='Indicador', values='previsao', aggfunc='first')
                    # Helper seguro
                    get_f = lambda k: piv.loc[k, 'previsao'] if k in piv.index else 0
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("IPCA", f"{get_f('IPCA'):.2f}%")
                    m2.metric("Selic", f"{get_f('Selic'):.2f}%")
                    m3.metric("PIB", f"{get_f('PIB Total'):.2f}%")
                    m4.metric("Dólar", f"R$ {get_f('Câmbio'):.2f}")
                
                # Tabela Projeções
                st.markdown("###### Projeções (3 Anos)")
                anos_proj = [ano_atual + i for i in range(3)]
                df_proj = df_focus[df_focus['ano_referencia'].isin(anos_proj)]
                if not df_proj.empty:
                    tab_proj = df_proj.pivot_table(index='Indicador', columns='ano_referencia', values='previsao')
                    st.dataframe(tab_proj, use_container_width=True)
            else:
                st.warning("⚠️ Boletim Focus indisponível ou erro na API.")

        # Câmbio
        with c2:
            st.markdown("#### 💵 Câmbio Agora")
            df_curr = get_currency_realtime()
            if not df_curr.empty:
                usd = df_curr.loc['USDBRL']
                eur = df_curr.loc['EURBRL']
                
                def show_curr(lbl, d):
                    st.metric(lbl, f"R$ {d['bid']:.2f}", f"{d['pctChange']:.2f}%")
                
                cc1, cc2 = st.columns(2)
                with cc1: show_curr("Dólar", usd)
                with cc2: show_curr("Euro", eur)
            else:
                st.info("Cotações em tempo real indisponíveis.")

    # 3. Expander Macro
    with st.expander("🧩 Conjuntura Macroeconômica", expanded=False):
        macro = get_macro_real()
        if macro:
            st.markdown("##### Atividade & Fiscal")
            k1, k2, k3, k4 = st.columns(4)
            # Uso seguro com .get() para evitar key errors se um falhar
            k1.metric("PIB", f"R$ {macro.get('PIB (R$ Bi)', 0):.2f} Tri")
            k2.metric("Dív. Líquida", f"{macro.get('Dívida Líq. (% PIB)', 0):.1f}% PIB")
            k3.metric("Res. Primário", f"{macro.get('Res. Primário (% PIB)', 0):.2f}% PIB")
            k4.metric("Res. Nominal", f"{macro.get('Res. Nominal (% PIB)', 0):.2f}% PIB")
            
            st.divider()
            st.markdown("##### Setor Externo")
            k5, k6, k7 = st.columns(3)
            k5.metric("Balança", f"US$ {macro.get('Balança Com. (US$ Mi)', 0):.1f} Bi")
            k6.metric("Trans. Correntes", f"US$ {macro.get('Trans. Correntes (US$ Mi)', 0):.1f} Bi")
            k7.metric("IDP", f"US$ {macro.get('IDP (US$ Mi)', 0):.1f} Bi")
        else:
            st.warning("⚠️ Dados macroeconômicos não carregados (Erro na API do BCB).")

    # 4. Expander Histórico Câmbio
    with st.expander("💸 Histórico de Câmbio (Longo Prazo)", expanded=False):
        df_hist = get_cambio_historico()
        if not df_hist.empty:
            t1, t2, t3 = st.tabs(["Gráfico", "Matriz", "Tabela"])
            with t1:
                fig = px.line(df_hist, x=df_hist.index, y=['Dólar', 'Euro'], 
                              color_discrete_map={"Dólar": "#00FF7F", "Euro": "#00BFFF"})
                fig.update_layout(template="plotly_white", hovermode="x unified", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with t2:
                # Matriz de Retornos
                moeda_sel = st.radio("Moeda:", ["Dólar", "Euro"], horizontal=True)
                # Resample seguro
                try:
                    df_m = df_hist[[moeda_sel]].resample('ME').last()
                    df_ret = df_m.pct_change() * 100
                    df_ret['ano'] = df_ret.index.year
                    df_ret['mes'] = df_ret.index.month
                    
                    # Pivot
                    matriz = df_ret.pivot(index='ano', columns='mes', values=moeda_sel)
                    # Renomear colunas
                    mapa_rev = {i: m for i, m in enumerate(['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'], 1)}
                    matriz = matriz.rename(columns=mapa_rev).sort_index(ascending=False)
                    
                    st.dataframe(matriz.style.background_gradient(
                        cmap=Config.CMAP_CUSTOM, vmin=-5, vmax=5).format("{:.2f}%"), 
                        use_container_width=True, height=450
                    )
                except Exception as e:
                    st.error(f"Erro ao gerar matriz de câmbio: {e}")
            
            with t3:
                # Tabela Simples
                df_tab = df_hist.sort_index(ascending=False).reset_index()
                df_tab['Date'] = df_tab['Date'].dt.strftime('%d/%m/%Y')
                st.dataframe(df_tab, use_container_width=True)

    # 5. Painel Principal (Deep Dive do Índice)
    st.title(f"📊 {config_sel['code']} - Análise Profunda")
    st.caption(f"Dados oficiais atualizados até: {df_principal.iloc[0]['data_fmt']}")

    # Diagnóstico Inteligente (NOVA FEATURE)
    analise_txt, analise_class = gerar_diagnostico_economico(df_principal, config_sel['code'])
    st.markdown(f"""
    <div class='insight-box {analise_class}'>
        {analise_txt}
    </div>
    """, unsafe_allow_html=True)

    # KPIs Principais
    pk1, pk2, pk3, pk4 = st.columns(4)
    pk1.metric("Mensal", f"{df_principal.iloc[0]['valor']:.2f}%")
    pk2.metric("Acum. 12 Meses", f"{df_principal.iloc[0]['acum_12m']:.2f}%")
    pk3.metric("Acum. Ano (YTD)", f"{df_principal.iloc[0]['acum_ano']:.2f}%")
    pk4.metric("Início Série", df_principal['ano'].min())

    # Tabs Visuais
    pt1, pt2, pt3 = st.tabs(["📈 Gráfico Interativo", "🗓️ Matriz de Sazonalidade", "📋 Base de Dados"])

    with pt1:
        # Slider de filtro de ano
        anos_disp = sorted(df_principal['ano'].astype(int).unique())
        if anos_disp:
            min_a, max_a = min(anos_disp), max(anos_disp)
            default_a = 2018 if 2018 >= min_a else min_a
            
            sel_ano = st.slider("Filtrar a partir de:", min_a, max_a, default_a)
            df_chart = df_principal[df_principal['ano'].astype(int) >= sel_ano].sort_values('data_date')
        else:
            df_chart = df_principal.sort_values('data_date')

        fig = px.area(df_chart, x='data_date', y='acum_12m', 
                      title=f"Acumulado 12 Meses - {config_sel['code']}")
        
        # Correção de cor segura
        fill_color = hex_to_rgba(config_sel['color'], 0.2)
        fig.update_traces(line_color=config_sel['color'], fillcolor=fill_color)
        fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    with pt2:
        try:
            mat = df_principal.pivot(index='ano', columns='mes_nome', values='valor')
            ordem = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
            mat = mat[[c for c in ordem if c in mat.columns]].sort_index(ascending=False)
            
            st.dataframe(mat.style.background_gradient(
                cmap=Config.CMAP_CUSTOM, vmin=-1.5, vmax=1.5).format("{:.2f}%"), 
                use_container_width=True, height=500
            )
        except Exception:
            st.warning("Matriz indisponível para este indicador.")

    with pt3:
        df_ex = df_principal[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].copy()
        csv = df_ex.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, f"{config_sel['code']}.csv", "text/csv")
        st.dataframe(df_ex, use_container_width=True)

    # Rodapé
    st.divider()
    st.markdown("<div style='text-align: center; color: #666;'>VPL Consultoria • Dados Oficiais: IBGE & Banco Central do Brasil</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
