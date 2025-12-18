import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
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
    }
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        font-weight: 600;
        color: #003366;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

cores_leves = ["#FFB3B3", "#FFFFFF", "#B3FFB3"] # Vermelho Suave, Branco, Verde Suave
cmap_leves = LinearSegmentedColormap.from_list("pastel_rdylgn", cores_leves)

# --- FUNÇÕES DE CARGA DE DADOS ---

# 1. IBGE (Sidra)
@st.cache_data
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
    except Exception as e:
        return pd.DataFrame()

# 2. Banco Central (SGS - Índices Mensais)
@st.cache_data
def get_bcb_data(codigo_serie):
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
        
        # Headers e Verify=False para evitar erros de conexão com o Governo
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json())
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        return processar_dataframe_comum(df)
    except Exception as e:
        return pd.DataFrame()

# 3. Boletim Focus (CORRIGIDO)
@st.cache_data(ttl=3600)
def get_focus_data():
    try:
        # Aumentamos o TOP para garantir que pegamos o histórico recente de todos os indicadores
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=5000&$orderby=Data%20desc&$format=json"
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        data_json = response.json()
        df = pd.DataFrame(data_json['value'])
        
        indicadores = [
            'IPCA', 'PIB Total', 'Selic', 'Câmbio', 'IGP-M',
            'IPCA Administrados', 'Conta corrente', 'Balança comercial',
            'Investimento direto no país', 'Dívida líquida do setor público',
            'Resultado primário', 'Resultado nominal'
        ]
        
        df = df[df['Indicador'].isin(indicadores)]
        df = df.rename(columns={'Data': 'data_relatorio', 'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'})
        
        # Tipagem correta
        df['ano_referencia'] = pd.to_numeric(df['ano_referencia'], errors='coerce')
        df['previsao'] = pd.to_numeric(df['previsao'], errors='coerce')
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        
        # --- A CORREÇÃO PRINCIPAL ---
        # Ordena da data mais recente para a mais antiga
        df = df.sort_values('data_relatorio', ascending=False)
        # Remove duplicatas mantendo apenas a PRIMEIRA ocorrência (a mais recente)
        # Isso evita misturar a previsão de hoje com a de ontem
        df = df.drop_duplicates(subset=['Indicador', 'ano_referencia'], keep='first')
        
        return df
    except Exception as e:
        return pd.DataFrame()

# 4. Cotação de Moedas (Tempo Real)
@st.cache_data(ttl=300)
def get_currency_realtime():
    try:
        tickers = ["USDBRL=X", "EURBRL=X"]
        dados = {}
        for t in tickers:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.fast_info
            preco_atual = info['last_price']
            fechamento_anterior = info['previous_close']
            variacao = ((preco_atual - fechamento_anterior) / fechamento_anterior) * 100
            key = t.replace("=X", "") 
            dados[key] = {'bid': preco_atual, 'pctChange': variacao}
        df = pd.DataFrame.from_dict(dados, orient='index')
        return df
    except Exception as e:
        return pd.DataFrame()

# 5. Histórico de Câmbio
@st.cache_data(ttl=86400)
def get_cambio_historico():
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="1994-07-01", progress=False)
        if df.empty: return pd.DataFrame()
        
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
        return pd.DataFrame()

# 6. Processamento Comum
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

# 7. Dados Macroeconômicos Reais (SGS)
@st.cache_data(ttl=3600)
def get_macro_real():
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
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/13?formato=json"
            resp = requests.get(url, headers=headers, verify=False, timeout=5)
            df = pd.DataFrame(resp.json())
            df['valor'] = pd.to_numeric(df['valor'])
            
            if df.empty: continue
            
            if nome == 'PIB (R$ Bi)':
                valor_final = df['valor'].iloc[-1] / 1_000_000 
            elif 'Balança' in nome or 'Trans.' in nome or 'IDP' in nome:
                valor_final = df['valor'].iloc[-12:].sum() / 1_000
            elif 'Primário' in nome or 'Nominal' in nome:
                valor_final = df['valor'].iloc[-1] * -1
            else:
                valor_final = df['valor'].iloc[-1]
            
            resultados[nome] = valor_final
            
        return resultados
    except Exception as e:
        return {}

# --- CÁLCULO ---
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
        'valor_final': valor_final, 'percentual': pct_total, 'fator': fator_acumulado, 'is_reverso': is_reverso
    }, None

# ==============================================================================
# LAYOUT - SIDEBAR
# ==============================================================================
try:
    st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
except:
    st.sidebar.write("VPL CONSULTORIA")

st.sidebar.header("Configurações")

tipo_indice = st.sidebar.selectbox(
    "Selecione o Indicador",
    ["IPCA (Inflação Oficial)", "INPC (Salários)", "IGP-M (Aluguéis)", "SELIC (Taxa Básica)", "CDI (Investimentos)"]
)

with st.spinner(f"Carregando dados..."):
    if "IPCA" in tipo_indice:
        df = get_sidra_data("1737", "63")
        cor_tema = "#00BFFF" 
    elif "INPC" in tipo_indice:
        df = get_sidra_data("1736", "44")
        cor_tema = "#00FF7F" 
    elif "IGP-M" in tipo_indice:
        df = get_bcb_data("189")
        cor_tema = "#FF6347" 
    elif "SELIC" in tipo_indice:
        df = get_bcb_data("4390")
        cor_tema = "#FFD700" 
    elif "CDI" in tipo_indice:
        df = get_bcb_data("4391")
        cor_tema = "#FFFFFF" 

if df.empty:
    st.error("Erro ao carregar dados. Verifique a conexão com o BCB/IBGE.")
    st.stop()

# --- CALCULADORA ---
st.sidebar.divider()
st.sidebar.subheader("Calculadora")
valor_input = st.sidebar.number_input("Valor (R$)", value=1000.00, step=100.00, format="%.2f")

lista_anos = sorted(df['ano'].unique(), reverse=True)
lista_meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
mapa_meses = {'Jan': '01', 'Fev': '02', 'Mar': '03', 'Abr': '04', 'Mai': '05', 'Jun': '06',
              'Jul': '07', 'Ago': '08', 'Set': '09', 'Out': '10', 'Nov': '11', 'Dez': '12'}

st.sidebar.markdown("**Data Referência**")
c1, c2 = st.sidebar.columns(2)
mes_ini = c1.selectbox("Mes Ini", lista_meses_nome, index=0, label_visibility="collapsed")
ano_ini = c2.selectbox("Ano Ini", lista_anos, index=1 if len(lista_anos)>1 else 0, label_visibility="collapsed")

st.sidebar.markdown("**Data Alvo**")
c3, c4 = st.sidebar.columns(2)
mes_fim = c3.selectbox("Mes Fim", lista_meses_nome, index=9, label_visibility="collapsed")
ano_fim = c4.selectbox("Ano Fim", lista_anos, index=0, label_visibility="collapsed")

if st.sidebar.button("Calcular", type="primary"):
    code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
    code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
    res, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
    
    if erro:
        st.error(erro)
    else:
        st.sidebar.divider()
        nome_indice = tipo_indice.split()[0]
        tipo_op = "Rendimento" if nome_indice in ["SELIC", "CDI"] else "Correção"
        texto_op = "Descapitalização" if res['is_reverso'] else f"{tipo_op} ({nome_indice})"
        st.sidebar.markdown(f"<small>{texto_op}</small>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<h2 style='color: {cor_tema}; margin:0;'>R$ {res['valor_final']:,.2f}</h2>", unsafe_allow_html=True)
        st.sidebar.markdown(f"Total Período: **{res['percentual']:.2f}%**")
        st.sidebar.markdown(f"Fator de Correção: **{res['fator']:.6f}**")

# ==============================================================================
# PAINEL PRINCIPAL
# ==============================================================================

# FOCUS
with st.expander("🔭 Clique para ver: Expectativas de Mercado (Focus) & Câmbio Hoje", expanded=False):
    col_top1, col_top2 = st.columns([2, 1])
    
    # FOCUS
    df_focus = get_focus_data()
    ano_atual = date.today().year
    
    with col_top1:
        if not df_focus.empty:
            # Pega a data mais recente disponível no dataframe filtrado
            ultima_data = df_focus['data_relatorio'].max()
            data_str = pd.to_datetime(ultima_data).strftime('%d/%m/%Y')
            st.markdown(f"**Boletim Focus ({data_str})**")
            
            # --- DESTAQUES ---
            df_atual = df_focus[df_focus['ano_referencia'] == ano_atual]
            pivot_atual = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='first')
            
            fc1, fc2, fc3, fc4 = st.columns(4)
            def get_val(idx): 
                try: return pivot_atual.loc[idx, 'previsao']
                except: return 0
            
            fc1.metric(f"IPCA {ano_atual}", f"{get_val('IPCA'):.2f}%")
            fc2.metric(f"Selic {ano_atual}", f"{get_val('Selic'):.2f}%")
            fc3.metric(f"PIB {ano_atual}", f"{get_val('PIB Total'):.2f}%")
            fc4.metric(f"Dólar {ano_atual}", f"R$ {get_val('Câmbio'):.2f}")
            
            st.divider()
            st.markdown("###### 📅 Projeções Macroeconômicas (2025 - 2027)")
            
            anos_exibir = [ano_atual, ano_atual + 1, ano_atual + 2]
            df_table = df_focus[df_focus['ano_referencia'].isin(anos_exibir)].copy()
            df_pivot_multi = df_table.pivot_table(index='Indicador', columns='ano_referencia', values='previsao', aggfunc='first')
            
            ordem = [
                'IPCA', 'IGP-M', 'IPCA Administrados', 'Selic', 'Câmbio', 'PIB Total', 
                'Dívida líquida do setor público', 'Resultado primário', 'Resultado nominal', 
                'Balança comercial', 'Conta corrente', 'Investimento direto no país' 
            ]
            ordem_final = [x for x in ordem if x in df_pivot_multi.index]
            df_pivot_multi = df_pivot_multi.reindex(ordem_final)
            
            # Formatação
            df_display = df_pivot_multi.copy()
            for col in df_display.columns:
                def formatador_inteligente(row):
                    val = row[col]
                    nome = row.name
                    if pd.isna(val): return "-"
                    if 'Câmbio' in nome: return f"R$ {val:.2f}"
                    elif any(x in nome for x in ['comercial', 'Conta corrente', 'Investimento']): return f"US$ {val:.2f} B"
                    else: return f"{val:.2f}%"
                df_display[col] = df_display.apply(formatador_inteligente, axis=1)
            
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("Focus indisponível (Erro na API ou Filtro).")

    # MOEDAS
    df_moedas = get_currency_realtime()
    with col_top2:
        st.markdown("**Câmbio (Agora)**")
        mc1, mc2 = st.columns(2)
        if not df_moedas.empty:
            try:
                usd = df_moedas.loc['USDBRL']
                eur = df_moedas.loc['EURBRL']
                mc1.metric("Dólar", f"R$ {float(usd['bid']):.2f}", f"{float(usd['pctChange']):.2f}%")
                mc2.metric("Euro", f"R$ {float(eur['bid']):.2f}", f"{float(eur['pctChange']):.2f}%")
            except: st.info("Erro moedas")
        else: st.info("API indisponível")

# CONJUNTURA MACRO
with st.expander("🧩 Conjuntura Macroeconômica (Dados Oficiais Realizados)", expanded=False):
    macro_dados = get_macro_real()
    if macro_dados:
        st.markdown("##### 🏛️ Atividade & Fiscal (Acum. 12 Meses)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PIB (Acum. 12m)", f"R$ {macro_dados.get('PIB (R$ Bi)', 0):.2f} Tri")
        c2.metric("Dív. Líquida Setor Púb.", f"{macro_dados.get('Dívida Líq. (% PIB)', 0):.1f}% PIB")
        c3.metric("Res. Primário", f"{macro_dados.get('Res. Primário (% PIB)', 0):.2f}% PIB")
        c4.metric("Res. Nominal", f"{macro_dados.get('Res. Nominal (% PIB)', 0):.2f}% PIB")
        
        st.divider()
        st.markdown("##### 🚢 Setor Externo (Acum. 12 Meses)")
        c5, c6, c7 = st.columns(3)
        c5.metric("Balança Comercial", f"US$ {macro_dados.get('Balança Com. (US$ Mi)', 0):.1f} Bi")
        c6.metric("Transações Correntes", f"US$ {macro_dados.get('Trans. Correntes (US$ Mi)', 0):.1f} Bi")
        c7.metric("Investimento Direto (IDP)", f"US$ {macro_dados.get('IDP (US$ Mi)', 0):.1f} Bi")
    else:
        st.warning("Não foi possível carregar os dados macroeconômicos do BCB.")

# CÂMBIO HISTÓRICO
with st.expander("💸 Histórico de Câmbio (Dólar e Euro desde 1994)", expanded=False):
    df_cambio = get_cambio_historico()
    if not df_cambio.empty:
        st.markdown(f"**Fechamento: {df_cambio.index[-1].strftime('%d/%m/%Y')}**")
        tab_graf, tab_matriz, tab_tabela = st.tabs(["📈 Gráfico", "📅 Matriz de Retornos", "📋 Tabela Diária"])
        
        with tab_graf:
            fig_cambio = px.line(df_cambio, x=df_cambio.index, y=['Dólar', 'Euro'], 
                                 color_discrete_map={"Dólar": "#00FF7F", "Euro": "#00BFFF"})
            fig_cambio.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font=dict(color="#E0E0E0"), hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_cambio, use_container_width=True)
            
        with tab_matriz:
            moeda_matriz = st.radio("Selecione a Moeda:", ["Dólar", "Euro"], horizontal=True)
            df_mensal = df_cambio[[moeda_matriz]].resample('ME').last()
            df_retorno = df_mensal.pct_change() * 100
            df_retorno['ano'] = df_retorno.index.year
            df_retorno['mes'] = df_retorno.index.month_name().str.slice(0, 3)
            mapa_meses_en_pt = {'Jan': 'Jan', 'Feb': 'Fev', 'Mar': 'Mar', 'Apr': 'Abr', 'May': 'Mai', 'Jun': 'Jun',
                                'Jul': 'Jul', 'Aug': 'Ago', 'Sep': 'Set', 'Oct': 'Out', 'Nov': 'Nov', 'Dec': 'Dez'}
            df_retorno['mes'] = df_retorno['mes'].map(mapa_meses_en_pt)
            try:
                matrix_cambio = df_retorno.pivot(index='ano', columns='mes', values=moeda_matriz)
                colunas_ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                matrix_cambio = matrix_cambio[colunas_ordem].sort_index(ascending=False)
                st.dataframe(matrix_cambio.style.background_gradient(cmap=cmap_leves, vmin=-5, vmax=5).format("{:.2f}%"), use_container_width=True, height=500)
            except: st.info("Matriz incompleta.")
            
        with tab_tabela:
            df_view = df_cambio.sort_index(ascending=False).reset_index()
            df_view['Date'] = df_view['Date'].dt.strftime('%d/%m/%Y')
            st.dataframe(df_view, use_container_width=True, hide_index=True)
    else:
        st.warning("Yahoo Finance indisponível.")

# PAINEL DO ÍNDICE PRINCIPAL
st.title(f"📊 Painel: {tipo_indice.split()[0]}")
if not df.empty:
    st.markdown(f"**Dados históricos atualizados até:** {df.iloc[0]['data_fmt']}")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Taxa do Mês", f"{df.iloc[0]['valor']:.2f}%")
    kpi2.metric("Acumulado 12 Meses", f"{df.iloc[0]['acum_12m']:.2f}%")
    kpi3.metric("Acumulado Ano (YTD)", f"{df.iloc[0]['acum_ano']:.2f}%")
    kpi4.metric("Início da Série", df['ano'].min())

    tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Matriz de Calor", "📋 Tabela Detalhada"])

    with tab1:
        df_chart = df.dropna(subset=['acum_12m']).sort_values('data_date')
        if any(idx in tipo_indice for idx in ["IGP-M", "SELIC", "CDI"]):
            df_chart = df_chart[df_chart['ano'].astype(int) >= 2000]
        fig = px.line(df_chart, x='data_date', y='acum_12m', title=f"Histórico 12 Meses - {tipo_indice.split()[0]}")
        fig.update_traces(line_color=cor_tema, line_width=3)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color="#E0E0E0"), hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        try:
            matrix = df.pivot(index='ano', columns='mes_nome', values='valor')
            ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            matrix = matrix[ordem].sort_index(ascending=False)
            st.dataframe(matrix.style.background_gradient(cmap=cmap_leves, axis=None, vmin=-1.5, vmax=1.5).format("{:.2f}"), use_container_width=True, height=500)
        except: st.warning("Matriz indisponível.")

    with tab3:
        csv_principal = df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar CSV", csv_principal, f"{tipo_indice.split()[0]}_historico.csv", "text/csv")
        st.dataframe(df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']], use_container_width=True, hide_index=True)
