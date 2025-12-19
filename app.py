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
import urllib3
import warnings

# --- CONFIGURAÇÃO INICIAL E SUPRESSÃO DE AVISOS ---
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO (NOVO LAYOUT) ---
st.markdown("""
<style>
    /* Estilo do Cartão */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Título acima do cartão (Focus) */
    .card-label-top {
        font-size: 0.85rem;
        font-weight: 600;
        color: #555;
        text-align: center;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Valor dentro do cartão */
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 5px;
    }

    /* Ajuste das Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(31, 119, 180, 0.1);
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# --- CORES E UTILITÁRIOS ---
CORES_INDICES = {
    'IPCA': '#00BFFF',   # Azul Claro
    'INPC': '#00FF7F',   # Verde Primavera
    'IGP-M': '#FF6347',  # Tomate
    'SELIC': '#FFD700',  # Ouro
    'CDI': '#9370DB'     # Roxo Médio
}

# Paleta suave para matriz de calor
cores_leves = ["#FFB3B3", "#FFFFFF", "#B3FFB3"]
cmap_custom = LinearSegmentedColormap.from_list("custom_rdylgn", cores_leves)

def hex_to_rgba(hex_color, opacity=0.2):
    """Converte HEX para RGBA para evitar erros no Plotly"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {opacity})"
    return hex_color

def formatar_moeda(valor):
    if pd.isna(valor): return "-"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def retry_request(func):
    def wrapper(*args, **kwargs):
        for _ in range(3):
            try:
                return func(*args, **kwargs)
            except Exception:
                time.sleep(1)
        return pd.DataFrame() 
    return wrapper

def processar_dataframe_comum(df):
    if df.empty: return df
    df = df.sort_values('data_date', ascending=True)
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12, min_periods=12).apply(np.prod, raw=True) - 1) * 100
    return df.sort_values('data_date', ascending=False)

# --- CARGA DE DADOS ---

@st.cache_data(ttl=3600)
def get_sidra_data(table_code, variable_code):
    try:
        dados_raw = sidrapy.get_table(
            table_code=table_code, territorial_level="1", ibge_territorial_code="all", 
            variable=variable_code, period="last 360"
        )
        if dados_raw.empty: return pd.DataFrame()
        df = dados_raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['D2C'].str.slice(0, 4)
        return processar_dataframe_comum(df)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_bcb_data(codigo_serie):
    @retry_request
    def fetch():
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        return response.json()
    try:
        data = fetch()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        return processar_dataframe_comum(df)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_focus_data():
    try:
        # Puxa top 5000 para garantir histórico recente de todos indicadores
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=5000&$orderby=Data%20desc&$format=json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, verify=False, timeout=15)
        data_json = response.json()
        df = pd.DataFrame(data_json['value'])
        if df.empty: return pd.DataFrame()

        indicadores = [
            'IPCA', 'PIB Total', 'Selic', 'Câmbio', 'IGP-M',
            'IPCA Administrados', 'Conta corrente', 'Balança comercial',
            'Investimento direto no país', 'Dívida líquida do setor público',
            'Resultado primário', 'Resultado nominal'
        ]
        
        df = df[df['Indicador'].isin(indicadores)]
        df = df.rename(columns={'Data': 'data_relatorio', 'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'})
        df['ano_referencia'] = pd.to_numeric(df['ano_referencia'], errors='coerce')
        df['previsao'] = pd.to_numeric(df['previsao'], errors='coerce')
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        
        # Ordenação rigorosa e remoção de duplicatas para pegar a ÚLTIMA previsão (Mediana)
        df = df.sort_values('data_relatorio', ascending=False)
        df = df.drop_duplicates(subset=['Indicador', 'ano_referencia'], keep='first')
        
        return df
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_currency_realtime():
    try:
        tickers = ["USDBRL=X", "EURBRL=X"]
        dados = {}
        for t in tickers:
            try:
                ticker_obj = yf.Ticker(t)
                try:
                    price = ticker_obj.fast_info['last_price']
                    prev = ticker_obj.fast_info['previous_close']
                except:
                    hist = ticker_obj.history(period='2d')
                    if hist.empty: continue
                    price = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[0]
                
                var = ((price - prev) / prev) * 100
                key = "USDBRL" if "USD" in t else "EURBRL"
                dados[key] = {'bid': price, 'pctChange': var}
            except: continue
        return pd.DataFrame.from_dict(dados, orient='index')
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_macro_real():
    series = {
        'PIB': 4382,
        'Dívida Líq.': 4513,
        'Res. Primário': 5362,
        'Res. Nominal': 5360,
        'Balança Com.': 22707,
        'Trans. Correntes': 22724,
        'IDP': 22885
    }
    kpis = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    for nome, codigo in series.items():
        try:
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/13?formato=json"
            resp = requests.get(url, headers=headers, verify=False, timeout=8)
            if resp.status_code != 200: continue
            df = pd.DataFrame(resp.json())
            if df.empty: continue
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            
            if nome == 'PIB':
                val_kpi = df['valor'].iloc[-1] / 1_000_000 
                fmt = 'tri'
            elif nome in ['Balança Com.', 'Trans. Correntes', 'IDP']:
                val_kpi = df['valor'].sum() / 1_000
                fmt = 'bi_usd'
            elif 'Primário' in nome or 'Nominal' in nome:
                val_kpi = df['valor'].iloc[-1] * -1
                fmt = 'pct_pib'
            else:
                val_kpi = df['valor'].iloc[-1]
                fmt = 'pct_pib'
            kpis[nome] = {'valor': val_kpi, 'data': df['data'].iloc[-1], 'fmt': fmt}
        except Exception: continue
    return kpis

@st.cache_data(ttl=86400)
def get_cambio_historico():
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="2000-01-01", progress=False)
        if df.empty: return pd.DataFrame()
        df = df['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo').tz_localize(None)
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        return df.ffill()
    except Exception: return pd.DataFrame()

# --- CÁLCULO DA CALCULADORA ---
def calcular_correcao(df, valor, data_ini_code, data_fim_code):
    try:
        is_reverso = int(data_ini_code) > int(data_fim_code)
        periodo_inicio = str(min(int(data_ini_code), int(data_fim_code)))
        periodo_fim = str(max(int(data_ini_code), int(data_fim_code)))
        mask = (df['D2C'] >= periodo_inicio) & (df['D2C'] <= periodo_fim)
        df_periodo = df.loc[mask].copy()
        if df_periodo.empty: return None, "Período sem dados suficientes."
        fator_acumulado = df_periodo['fator'].prod()
        if is_reverso:
            valor_final = valor / fator_acumulado
        else:
            valor_final = valor * fator_acumulado
        pct_total = (fator_acumulado - 1) * 100
        return {'valor_final': valor_final, 'percentual': pct_total, 'fator': fator_acumulado, 'is_reverso': is_reverso}, None
    except Exception as e: return None, f"Erro no cálculo: {str(e)}"

# ==============================================================================
# SIDEBAR (Configuração e Calculadora)
# ==============================================================================
st.sidebar.markdown("### 📊 VPL Consultoria")
st.sidebar.markdown("---")

INDICES_CONFIG = {
    "IPCA (Inflação Oficial)": {"tipo": "IPCA", "sidra": ("1737", "63"), "bcb": None},
    "INPC (Salários)": {"tipo": "INPC", "sidra": ("1736", "44"), "bcb": None},
    "IGP-M (Aluguéis)": {"tipo": "IGP-M", "sidra": None, "bcb": "189"},
    "SELIC (Taxa Básica)": {"tipo": "SELIC", "sidra": None, "bcb": "4390"},
    "CDI (Investimentos)": {"tipo": "CDI", "sidra": None, "bcb": "4391"}
}

opcao_indice = st.sidebar.selectbox("Selecione o Indicador", list(INDICES_CONFIG.keys()))
config_selecionada = INDICES_CONFIG[opcao_indice]
nome_curto = config_selecionada['tipo']
cor_tema = CORES_INDICES[nome_curto]

with st.spinner(f"Carregando {nome_curto}..."):
    if config_selecionada['sidra']:
        df_indice = get_sidra_data(*config_selecionada['sidra'])
    else:
        df_indice = get_bcb_data(config_selecionada['bcb'])

if df_indice.empty:
    st.error(f"Erro ao carregar {nome_curto}. Verifique conexão.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Calculadora de Correção")
valor_input = st.sidebar.number_input("Valor Inicial (R$)", value=1000.00, step=100.00)
lista_anos = sorted(df_indice['ano'].unique(), reverse=True)
meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
mapa_meses = {m: f"{i:02d}" for i, m in enumerate(meses_nome, 1)}

c1, c2 = st.sidebar.columns(2)
mes_ini = c1.selectbox("Mês Inicial", meses_nome, index=0)
ano_ini = c2.selectbox("Ano Inicial", lista_anos, index=min(1, len(lista_anos)-1) if len(lista_anos) > 1 else 0)
c3, c4 = st.sidebar.columns(2)
mes_fim = c3.selectbox("Mês Final", meses_nome, index=min(11, len(meses_nome)-1))
ano_fim = c4.selectbox("Ano Final", lista_anos, index=0)

if st.sidebar.button("Calcular Correção", type="primary", use_container_width=True):
    code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
    code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
    res, erro = calcular_correcao(df_indice, valor_input, code_ini, code_fim)
    if erro:
        st.sidebar.error(erro)
    else:
        st.sidebar.markdown("---")
        label_op = "🔻 Descapitalização" if res['is_reverso'] else "📈 Valor Corrigido"
        st.sidebar.markdown(f"""
        <div style="background-color: {cor_tema}20; padding: 15px; border-radius: 10px; border-left: 5px solid {cor_tema};">
            <small>{label_op}</small>
            <h2 style="color: {cor_tema}; margin: 0;">{formatar_moeda(res['valor_final'])}</h2>
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <span>Total: <b>{res['percentual']:.2f}%</b></span>
                <span>Fator: <b>{res['fator']:.4f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PAINEL PRINCIPAL
# ==============================================================================

# --- FOCUS (LAYOUT NOVO: RÓTULO EM CIMA, VALOR DENTRO) ---
with st.expander("🔭 **Expectativas de Mercado (Focus)**", expanded=False):
    df_focus = get_focus_data()
    ano_atual = date.today().year
    
    if not df_focus.empty:
        data_ref = df_focus['data_relatorio'].max().strftime('%d/%m/%Y')
        st.markdown(f"**Referência: {data_ref}**")
        
        df_atual = df_focus[df_focus['ano_referencia'] == ano_atual]
        pivot = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='first')
        
        def get_focus_val(idx):
            try: return pivot.loc[idx, 'previsao']
            except: return 0.0

        c1, c2, c3, c4 = st.columns(4)
        
        def card_focus(col, titulo, valor, prefixo="", sufixo=""):
            col.markdown(f"<div class='card-label-top'>{titulo}</div>", unsafe_allow_html=True)
            col.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{prefixo}{valor}{sufixo}</div>
                </div>
            """, unsafe_allow_html=True)

        card_focus(c1, f"IPCA {ano_atual}", f"{get_focus_val('IPCA'):.2f}", sufixo="%")
        card_focus(c2, f"Selic {ano_atual}", f"{get_focus_val('Selic'):.2f}", sufixo="%")
        card_focus(c3, f"PIB {ano_atual}", f"{get_focus_val('PIB Total'):.2f}", sufixo="%")
        card_focus(c4, f"Dólar {ano_atual}", f"{get_focus_val('Câmbio'):.2f}", prefixo="R$ ")
        
        st.markdown("###### Projeções 2025 - 2027")
        anos_exibir = [ano_atual, ano_atual + 1, ano_atual + 2]
        df_proj = df_focus[df_focus['ano_referencia'].isin(anos_exibir)].copy()
        
        if not df_proj.empty:
            pivot_proj = df_proj.pivot_table(index='Indicador', columns='ano_referencia', values='previsao', aggfunc='first')
            df_display = pivot_proj.copy()
            for col in df_display.columns:
                df_display[col] = df_display.apply(lambda x: 
                    f"R$ {x[col]:.2f}" if 'Câmbio' in x.name else
                    (f"US$ {x[col]:.1f} Bi" if any(k in x.name for k in ['comercial', 'Conta', 'Investimento']) else f"{x[col]:.2f}%")
                , axis=1)
            st.dataframe(df_display, use_container_width=True)
    else:
        st.warning("Dados do Focus indisponíveis no momento.")

# --- MACROECONOMIA ---
with st.expander("🧩 **Conjuntura Macroeconômica (Dados Realizados)**", expanded=False):
    kpis_macro = get_macro_real()
    if kpis_macro:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        def render_macro_card(col, titulo, key, kpis):
            if key in kpis:
                dado = kpis[key]
                val = dado['valor']
                fmt = dado['fmt']
                if fmt == 'tri': txt = f"R$ {val:.2f} Tri"
                elif fmt == 'bi_usd': txt = f"US$ {val:.1f} Bi"
                elif fmt == 'pct_pib': txt = f"{val:.2f}% PIB"
                else: txt = f"{val:.1f}"
                col.metric(titulo, txt)
        render_macro_card(col_m1, "PIB (Nominal)", 'PIB', kpis_macro)
        render_macro_card(col_m2, "Dívida Líquida", 'Dívida Líq.', kpis_macro)
        render_macro_card(col_m3, "Superávit Primário", 'Res. Primário', kpis_macro)
        render_macro_card(col_m4, "Balança Comercial (12m)", 'Balança Com.', kpis_macro)
    else:
        st.warning("Dados macroeconômicos não carregados (Erro BCB).")

# --- ANÁLISE DO ÍNDICE SELECIONADO ---
st.title(f"📊 Análise: {nome_curto}")
st.caption(f"Última atualização: {df_indice.iloc[0]['data_fmt']}")

def card_kpi(col, label, value, sublabel):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {cor_tema}">{value}</div>
        <div class="metric-delta">{sublabel}</div>
    </div>
    """, unsafe_allow_html=True)

kp1, kp2, kp3, kp4 = st.columns(4)
card_kpi(kp1, "Mensal", f"{df_indice.iloc[0]['valor']:.2f}%", f"Mês: {df_indice.iloc[0]['mes_nome']}")
card_kpi(kp2, "Acumulado 12 Meses", f"{df_indice.iloc[0]['acum_12m']:.2f}%", "Últimos 12 meses")
card_kpi(kp3, "Acumulado Ano (YTD)", f"{df_indice.iloc[0]['acum_ano']:.2f}%", f"Em {df_indice.iloc[0]['ano']}")
card_kpi(kp4, "Início da Série", df_indice['ano'].min(), "Ano base")

tab1, tab2, tab3 = st.tabs(["📈 Gráfico Interativo", "🗓️ Matriz de Calor", "📋 Dados Detalhados"])

with tab1:
    anos_disp = sorted(df_indice['ano'].astype(int).unique())
    ano_start = st.slider("Filtrar a partir do ano:", min(anos_disp), max(anos_disp), 2010 if 2010 in anos_disp else min(anos_disp), key='slider_indice')
    df_chart = df_indice[df_indice['ano'].astype(int) >= ano_start].sort_values('data_date')
    fig = px.area(df_chart, x='data_date', y='acum_12m', title=f"{nome_curto} - Acumulado 12 Meses (%)")
    fill_color = hex_to_rgba(cor_tema, 0.2)
    fig.update_traces(line_color=cor_tema, fillcolor=fill_color)
    fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=50, b=0), height=400, yaxis_title="Percentual (%)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("##### Variação Mensal (%)")
    try:
        matrix = df_indice.pivot(index='ano', columns='mes_nome', values='valor')
        ordem_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        matrix = matrix[ordem_meses].sort_index(ascending=False)
        st.dataframe(matrix.style.background_gradient(cmap=cmap_custom, axis=None, vmin=-1.5, vmax=1.5).format("{:.2f}"), use_container_width=True, height=500)
    except: st.error("Erro ao gerar matriz.")

with tab3:
    csv = df_indice[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, f"{nome_curto}.csv", "text/csv")
    st.dataframe(df_indice[['data_fmt', 'valor', 'acum_ano', 'acum_12m']], use_container_width=True, hide_index=True)

st.divider()

# --- ANÁLISE CÂMBIO (NOVA SEÇÃO COMPLETA) ---
st.title("💱 Análise: Câmbio")
df_moedas = get_currency_realtime()
df_hist_cambio = get_cambio_historico()

# Preparação dos KPIs de Câmbio
dolar_bid = 0
dolar_var = 0
euro_bid = 0
euro_var = 0
dolar_ytd = 0
euro_ytd = 0

if not df_moedas.empty and 'USDBRL' in df_moedas.index:
    dolar_bid = df_moedas.loc['USDBRL']['bid']
    dolar_var = df_moedas.loc['USDBRL']['pctChange']
if not df_moedas.empty and 'EURBRL' in df_moedas.index:
    euro_bid = df_moedas.loc['EURBRL']['bid']
    euro_var = df_moedas.loc['EURBRL']['pctChange']

# Cálculo YTD Histórico
if not df_hist_cambio.empty:
    ano_atual = datetime.now().year
    df_ytd = df_hist_cambio[df_hist_cambio.index.year == ano_atual]
    if not df_ytd.empty:
        first_val_d = df_ytd['Dólar'].iloc[0]
        last_val_d = df_ytd['Dólar'].iloc[-1]
        dolar_ytd = ((last_val_d - first_val_d) / first_val_d) * 100
        
        first_val_e = df_ytd['Euro'].iloc[0]
        last_val_e = df_ytd['Euro'].iloc[-1]
        euro_ytd = ((last_val_e - first_val_e) / first_val_e) * 100

def card_cambio(col, moeda, valor, var_dia, var_ytd):
    cor_dia = "green" if var_dia >= 0 else "red"
    cor_ytd = "green" if var_ytd >= 0 else "red"
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{moeda}</div>
        <div class="metric-value" style="color: #333">R$ {valor:.2f}</div>
        <div class="metric-delta">
            Dia: <span style="color: {cor_dia}">{var_dia:+.2f}%</span> | 
            Ano: <span style="color: {cor_ytd}">{var_ytd:+.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kc1, kc2 = st.columns(2)
card_cambio(kc1, "Dólar (Comercial)", dolar_bid, dolar_var, dolar_ytd)
card_cambio(kc2, "Euro (Comercial)", euro_bid, euro_var, euro_ytd)

ctab1, ctab2, ctab3 = st.tabs(["📈 Histórico", "🗓️ Matriz de Retornos", "📋 Tabela"])

with ctab1:
    if not df_hist_cambio.empty:
        fig_cambio = px.line(df_hist_cambio, x=df_hist_cambio.index, y=['Dólar', 'Euro'], 
                             color_discrete_map={"Dólar": "#00FF7F", "Euro": "#00BFFF"})
        fig_cambio.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), height=400,
                                 legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_cambio, use_container_width=True)
    else: st.info("Histórico indisponível.")

with ctab2:
    if not df_hist_cambio.empty:
        moeda_matriz = st.radio("Selecione a Moeda:", ["Dólar", "Euro"], horizontal=True)
        df_mensal = df_hist_cambio[[moeda_matriz]].resample('ME').last()
        df_retorno = df_mensal.pct_change() * 100
        df_retorno['ano'] = df_retorno.index.year
        df_retorno['mes_num'] = df_retorno.index.month
        df_retorno['mes'] = df_retorno['mes_num'].map({1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                                                       7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'})
        
        try:
            matrix_c = df_retorno.pivot(index='ano', columns='mes', values=moeda_matriz)
            colunas_ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            matrix_c = matrix_c[colunas_ordem].sort_index(ascending=False)
            st.dataframe(matrix_c.style.background_gradient(cmap=cmap_custom, axis=None, vmin=-5, vmax=5).format("{:.2f}%"), use_container_width=True)
        except: st.error("Dados insuficientes para matriz.")

with ctab3:
    if not df_hist_cambio.empty:
        df_exibir = df_hist_cambio.sort_index(ascending=False).reset_index()
        df_exibir.columns = ['Data', 'Dólar', 'Euro']
        st.dataframe(df_exibir, use_container_width=True, hide_index=True)

st.divider()
st.markdown("<div style='text-align: center; color: #888;'>Desenvolvido por VPL Consultoria • Dados: IBGE, BCB e Yahoo Finance</div>", unsafe_allow_html=True)
