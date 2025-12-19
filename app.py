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

# --- CSS (Baseado na Versão 1 - Visual Otimizado) ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f2f6 100%);
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-delta {
        font-size: 0.9rem;
    }
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        color: #003366;
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
    """Tenta a requisição 3 vezes antes de falhar (para APIs instáveis do governo)"""
    def wrapper(*args, **kwargs):
        for _ in range(3):
            try:
                return func(*args, **kwargs)
            except Exception:
                time.sleep(1)
        return pd.DataFrame() 
    return wrapper

def processar_dataframe_comum(df):
    """Lógica unificada de tratamento de dados"""
    if df.empty: return df
    
    df = df.sort_values('data_date', ascending=True)
    
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    
    # Cálculos Financeiros
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12, min_periods=12).apply(np.prod, raw=True) - 1) * 100
    
    return df.sort_values('data_date', ascending=False)

# --- CARGA DE DADOS (APIs) ---

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
    except Exception:
        return pd.DataFrame()

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
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_focus_data():
    """Lógica robusta do codigo_funcionando para evitar duplicatas"""
    try:
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
        
        df = df.sort_values('data_relatorio', ascending=False)
        df = df.drop_duplicates(subset=['Indicador', 'ano_referencia'], keep='first')
        
        return df
    except Exception:
        return pd.DataFrame()

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
            except:
                continue
        return pd.DataFrame.from_dict(dados, orient='index')
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_macro_real():
    """Busca dados macroeconômicos do BCB - CÓDIGOS CORRIGIDOS DO CODIGO_FUNCIONANDO"""
    series = {
        'PIB': 4382,
        'Dívida Líq.': 4513,
        'Res. Primário': 5362,  # Código do arquivo funcionando
        'Res. Nominal': 5360,   # Código do arquivo funcionando
        'Balança Com.': 22707,
        'Trans. Correntes': 22724,
        'IDP': 22885
    }
    
    kpis = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for nome, codigo in series.items():
        try:
            # Tenta pegar ultimos 13 meses para garantir cálculo de acumulado
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/13?formato=json"
            resp = requests.get(url, headers=headers, verify=False, timeout=8)
            
            if resp.status_code != 200: continue
            
            df = pd.DataFrame(resp.json())
            if df.empty: continue

            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            
            # Lógica de cálculo igual ao código funcionando
            if nome == 'PIB':
                val_kpi = df['valor'].iloc[-1] / 1_000_000 # Tri
                fmt = 'tri'
            elif nome in ['Balança Com.', 'Trans. Correntes', 'IDP']:
                val_kpi = df['valor'].sum() / 1_000 # Soma 12m (aproximado pelo endpoint ultimos/13) em Bi
                fmt = 'bi_usd'
            elif 'Primário' in nome or 'Nominal' in nome:
                # Nestes códigos (5362/5360), o BCB geralmente traz % PIB mensal acumulado ou similar
                # O código funcionando multiplicava por -1
                val_kpi = df['valor'].iloc[-1] * -1
                fmt = 'pct_pib'
            else:
                val_kpi = df['valor'].iloc[-1]
                fmt = 'pct_pib'

            kpis[nome] = {'valor': val_kpi, 'data': df['data'].iloc[-1], 'fmt': fmt}
            
        except Exception:
            continue
            
    return kpis

@st.cache_data(ttl=86400)
def get_cambio_historico():
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="2000-01-01", progress=False)
        if df.empty: return pd.DataFrame()
        
        df = df['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ajuste de timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo').tz_localize(None)
        
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        return df.ffill()
    except Exception:
        return pd.DataFrame()

# --- LÓGICA DE CÁLCULO ---
def calcular_correcao(df, valor, data_ini_code, data_fim_code):
    try:
        is_reverso = int(data_ini_code) > int(data_fim_code)
        periodo_inicio = str(min(int(data_ini_code), int(data_fim_code)))
        periodo_fim = str(max(int(data_ini_code), int(data_fim_code)))
        
        mask = (df['D2C'] >= periodo_inicio) & (df['D2C'] <= periodo_fim)
        df_periodo = df.loc[mask].copy()
        
        if df_periodo.empty:
            return None, "Período sem dados suficientes."
        
        fator_acumulado = df_periodo['fator'].prod()
        
        if is_reverso:
            valor_final = valor / fator_acumulado
        else:
            valor_final = valor * fator_acumulado
            
        pct_total = (fator_acumulado - 1) * 100
        
        return {
            'valor_final': valor_final, 
            'percentual': pct_total, 
            'fator': fator_acumulado, 
            'is_reverso': is_reverso
        }, None
    except Exception as e:
        return None, f"Erro no cálculo: {str(e)}"

# ==============================================================================
# INTERFACE (SIDEBAR)
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
    st.error(f"Não foi possível carregar os dados do {nome_curto}. Verifique a conexão.")
    st.stop()

# CALCULADORA
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
# ÁREA PRINCIPAL
# ==============================================================================

# --- EXPANDER 1: FOCUS E CÂMBIO ---
with st.expander("🔭 **Expectativas de Mercado (Focus) & Câmbio**", expanded=False):
    col_focus, col_cambio = st.columns([2, 1])
    
    with col_focus:
        df_focus = get_focus_data()
        ano_atual = date.today().year
        
        if not df_focus.empty:
            data_ref = df_focus['data_relatorio'].max().strftime('%d/%m/%Y')
            st.markdown(f"##### Boletim Focus ({data_ref})")
            
            df_atual = df_focus[df_focus['ano_referencia'] == ano_atual]
            pivot = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='first')
            
            def get_focus_val(idx):
                try: return pivot.loc[idx, 'previsao']
                except: return 0.0

            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric(f"IPCA {ano_atual}", f"{get_focus_val('IPCA'):.2f}%")
            fc2.metric(f"Selic {ano_atual}", f"{get_focus_val('Selic'):.2f}%")
            fc3.metric(f"PIB {ano_atual}", f"{get_focus_val('PIB Total'):.2f}%")
            fc4.metric(f"Dólar {ano_atual}", f"R$ {get_focus_val('Câmbio'):.2f}")
            
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

    with col_cambio:
        st.markdown("##### Câmbio (Agora)")
        df_moedas = get_currency_realtime()
        
        mc1, mc2 = st.columns(2)
        if not df_moedas.empty and 'USDBRL' in df_moedas.index:
            usd = df_moedas.loc['USDBRL']
            eur = df_moedas.loc['EURBRL'] if 'EURBRL' in df_moedas.index else None
            
            with mc1:
                st.markdown("""<div class="metric-card"><div class="metric-label">Dólar</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-value">R$ {usd['bid']:.2f}</div>""", unsafe_allow_html=True)
                cor_delta = "green" if usd['pctChange'] >= 0 else "red"
                st.markdown(f"""<div class="metric-delta" style="color: {cor_delta}">{usd['pctChange']:+.2f}%</div></div>""", unsafe_allow_html=True)
            
            with mc2:
                if eur is not None:
                    st.markdown("""<div class="metric-card"><div class="metric-label">Euro</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-value">R$ {eur['bid']:.2f}</div>""", unsafe_allow_html=True)
                    cor_delta = "green" if eur['pctChange'] >= 0 else "red"
                    st.markdown(f"""<div class="metric-delta" style="color: {cor_delta}">{eur['pctChange']:+.2f}%</div></div>""", unsafe_allow_html=True)
        else:
            st.info("Cotação indisponível")

# --- EXPANDER 2: MACROECONOMIA ---
with st.expander("🧩 **Conjuntura Macroeconômica (Dados Realizados)**", expanded=False):
    kpis_macro = get_macro_real()
    
    if kpis_macro:
        st.markdown("Monitoramento dos principais indicadores (Fonte: BCB)")
        
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
        render_macro_card(col_m4, "Balança Comercial", 'Balança Com.', kpis_macro)
    else:
        st.warning("Dados macroeconômicos não carregados (Erro BCB).")

# --- EXPANDER 3: GRÁFICO CÂMBIO ---
with st.expander("💸 **Histórico de Câmbio (Desde 2000)**", expanded=False):
    df_cambio = get_cambio_historico()
    if not df_cambio.empty:
        fig_cambio = px.line(df_cambio, x=df_cambio.index, y=['Dólar', 'Euro'], 
                             color_discrete_map={"Dólar": "#00FF7F", "Euro": "#00BFFF"})
        fig_cambio.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), height=350,
                                 legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_cambio, use_container_width=True)
    else:
        st.info("Histórico de câmbio indisponível.")

# --- PAINEL PRINCIPAL DO ÍNDICE ---
st.title(f"📊 Análise: {nome_curto}")
st.caption(f"Última atualização dos dados: {df_indice.iloc[0]['data_fmt']}")

kp1, kp2, kp3, kp4 = st.columns(4)

def card_html(label, value, sublabel):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {cor_tema}">{value}</div>
        <div class="metric-delta">{sublabel}</div>
    </div>
    """

kp1.markdown(card_html("Mensal", f"{df_indice.iloc[0]['valor']:.2f}%", f"Mês: {df_indice.iloc[0]['mes_nome']}"), unsafe_allow_html=True)
kp2.markdown(card_html("Acumulado 12 Meses", f"{df_indice.iloc[0]['acum_12m']:.2f}%", "Últimos 12 meses"), unsafe_allow_html=True)
kp3.markdown(card_html("Acumulado Ano (YTD)", f"{df_indice.iloc[0]['acum_ano']:.2f}%", f"Em {df_indice.iloc[0]['ano']}"), unsafe_allow_html=True)
kp4.markdown(card_html("Início da Série", df_indice['ano'].min(), "Ano base"), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Gráfico Interativo", "🗓️ Matriz de Calor", "📋 Dados Detalhados"])

with tab1:
    anos_disp = sorted(df_indice['ano'].astype(int).unique())
    ano_start = st.slider("Filtrar a partir do ano:", min(anos_disp), max(anos_disp), 2010 if 2010 in anos_disp else min(anos_disp))
    
    df_chart = df_indice[df_indice['ano'].astype(int) >= ano_start].sort_values('data_date')
    
    fig = px.area(df_chart, x='data_date', y='acum_12m', title=f"{nome_curto} - Acumulado 12 Meses (%)")
    
    # CORREÇÃO DO ERRO PLOTLY: Usando hex_to_rgba
    fill_color = hex_to_rgba(cor_tema, 0.2)
    fig.update_traces(line_color=cor_tema, fillcolor=fill_color)
    
    fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=50, b=0), height=400,
                      yaxis_title="Percentual (%)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("##### Variação Mensal (%)")
    try:
        matrix = df_indice.pivot(index='ano', columns='mes_nome', values='valor')
        ordem_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        matrix = matrix[ordem_meses].sort_index(ascending=False)
        
        # Limite visual para melhor leitura
        st.dataframe(
            matrix.style.background_gradient(
                cmap=cmap_custom, axis=None, vmin=-1.5, vmax=1.5
            ).format("{:.2f}"),
            use_container_width=True,
            height=500
        )
    except Exception as e:
        st.error(f"Erro ao gerar matriz: {e}")

with tab3:
    col_download, _ = st.columns([1, 4])
    csv = df_indice[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].to_csv(index=False).encode('utf-8')
    col_download.download_button(
        "📥 Download CSV",
        csv,
        f"{nome_curto}_historico.csv",
        "text/csv",
        key='download-csv'
    )
    
    st.dataframe(
        df_indice[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].rename(columns={
            'data_fmt': 'Data', 'valor': 'Mensal (%)', 'acum_ano': 'Acum. Ano (%)', 'acum_12m': 'Acum. 12M (%)'
        }),
        use_container_width=True,
        hide_index=True
    )

st.divider()
st.markdown("<div style='text-align: center; color: #888;'>Desenvolvido por VPL Consultoria • Dados: IBGE, BCB e Yahoo Finance</div>", unsafe_allow_html=True)
