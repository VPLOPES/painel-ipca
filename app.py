import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Calculadora de Inflação Pro",
    page_icon="💰",
    layout="wide"
)

# Estilo CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #003366;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE CARGA DE DADOS ---

# 1. Dados do IBGE (Sidra) - Serve para IPCA e INPC
@st.cache_data
def get_sidra_data(table_code, variable_code):
    try:
        # Busca últimos 30 anos
        dados_raw = sidrapy.get_table(
            table_code=table_code, 
            territorial_level="1", 
            ibge_territorial_code="all", 
            variable=variable_code, 
            period="last 360"
        )
        
        df = dados_raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'])
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m") # Auxiliar para ordenação
        df['ano'] = df['D2C'].str.slice(0, 4)
        
        return processar_dataframe_comum(df)
    except Exception as e:
        st.error(f"Erro ao buscar dados do IBGE: {e}")
        return pd.DataFrame()

# 2. Dados do Banco Central (SGS) - Serve para IGP-M
@st.cache_data
def get_bcb_data(codigo_serie):
    try:
        # URL oficial da API do Banco Central
        url = f"http://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
        df = pd.read_json(url)
        
        # O BCB retorna: 'data' (dd/mm/aaaa) e 'valor'
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        
        # Criar a coluna D2C (YYYYMM) para compatibilidade com o resto do sistema
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        df['mes_ano'] = df['data_date'].dt.strftime('%B %Y') # Nome provisório
        
        return processar_dataframe_comum(df)
    except Exception as e:
        st.error(f"Erro ao buscar dados do BCB: {e}")
        return pd.DataFrame()

# 3. Processamento Comum (Padroniza tudo para o App)
def processar_dataframe_comum(df):
    # Garante ordenação cronológica para cálculos
    df = df.sort_values('data_date', ascending=True)
    
    # Mapeamento de meses (PT-BR)
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    
    # Cálculos Financeiros
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12).apply(np.prod, raw=True) - 1) * 100
    
    return df.sort_values('data_date', ascending=False) # Retorna do mais recente pro antigo

# --- FUNÇÃO DE CÁLCULO (A mesma de antes) ---
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
        'is_reverso': is_reverso
    }, None

# --- SIDEBAR E SELEÇÃO DE ÍNDICE ---
st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
st.sidebar.header("Configurações")

# ** SELETOR DE ÍNDICE **
tipo_indice = st.sidebar.selectbox(
    "Selecione o Índice Econômico",
    ["IPCA (Inflação Oficial)", "IGP-M (Aluguéis)", "INPC (Salários)"]
)

# Lógica para carregar o dado certo
with st.spinner(f"Carregando dados do {tipo_indice}..."):
    if "IPCA" in tipo_indice:
        df = get_sidra_data("1737", "63") # Tabela 1737, Var 63 (IPCA)
        cor_tema = "#003366" # Azul
    elif "INPC" in tipo_indice:
        df = get_sidra_data("1736", "44") # Tabela 1737, Var 44 (INPC)
        cor_tema = "#2E8B57" # Verde SeaGreen
    elif "IGP-M" in tipo_indice:
        df = get_bcb_data("189") # Série 189 do BCB (IGP-M Mensal)
        cor_tema = "#8B0000" # Vermelho Escuro

if df.empty:
    st.stop()

# --- INPUTS DA CALCULADORA ---
st.sidebar.divider()
st.sidebar.subheader("Calculadora")
valor_input = st.sidebar.number_input("Valor (R$)", value=1000.00, step=100.00, format="%.2f")

# Listas para Data
lista_anos = sorted(df['ano'].unique(), reverse=True)
lista_meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
mapa_meses = {'Jan': '01', 'Fev': '02', 'Mar': '03', 'Abr': '04', 'Mai': '05', 'Jun': '06',
              'Jul': '07', 'Ago': '08', 'Set': '09', 'Out': '10', 'Nov': '11', 'Dez': '12'}

# Seletores
st.sidebar.markdown("**Data Referência**")
c1, c2 = st.sidebar.columns(2)
mes_ini = c1.selectbox("Mes Ini", lista_meses_nome, index=0, label_visibility="collapsed")
ano_ini = c2.selectbox("Ano Ini", lista_anos, index=1 if len(lista_anos)>1 else 0, label_visibility="collapsed")

st.sidebar.markdown("**Data Alvo**")
c3, c4 = st.sidebar.columns(2)
mes_fim = c3.selectbox("Mes Fim", lista_meses_nome, index=9, label_visibility="collapsed")
ano_fim = c4.selectbox("Ano Fim", lista_anos, index=0, label_visibility="collapsed")

if st.sidebar.button("Calcular Correção", type="primary"):
    code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
    code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
    
    res, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
    
    if erro:
        st.error(erro)
    else:
        st.sidebar.divider()
        texto_op = "Deflação (Reverso)" if res['is_reverso'] else f"Correção ({tipo_indice.split()[0]})"
        st.sidebar.markdown(f"<small>{texto_op}</small>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<h2 style='color: {cor_tema}; margin:0;'>R$ {res['valor_final']:,.2f}</h2>", unsafe_allow_html=True)
        st.sidebar.markdown(f"Acumulado: **{res['percentual']:.2f}%**")

# --- ÁREA PRINCIPAL ---
st.title(f"📊 Painel Econômico: {tipo_indice.split()[0]}")
st.markdown(f"**Dados atualizados até:** {df.iloc[0]['data_fmt']}")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Índice do Mês", f"{df.iloc[0]['valor']:.2f}%")
c2.metric("Acumulado 12 Meses", f"{df.iloc[0]['acum_12m']:.2f}%")
c3.metric("Acumulado Ano (YTD)", f"{df.iloc[0]['acum_ano']:.2f}%")
c4.metric("Início da Série", df['ano'].min())

st.divider()

# Abas
tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Matriz", "📋 Tabela"])

with tab1:
    # 1. Prepara os dados base (remove vazios e ordena)
    df_chart = df.dropna(subset=['acum_12m']).sort_values('data_date')
    
    # --- FILTRO DE DATA (A NOVIDADE) ---
    # Se for IGP-M, filtramos para mostrar apenas de 2000 para frente.
    # Isso remove a distorção da hiperinflação dos anos 90.
    if "IGP-M" in tipo_indice:
        df_chart = df_chart[df_chart['ano'].astype(int) >= 2000]
    
    # 2. Cria o gráfico com os dados filtrados
    fig = px.line(df_chart, x='data_date', y='acum_12m', title=f"Evolução {tipo_indice} (12 meses)")
    fig.update_traces(line_color=cor_tema, line_width=3)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    try:
        matrix = df.pivot(index='ano', columns='mes_nome', values='valor')
        ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        matrix = matrix[ordem].sort_index(ascending=False)
        st.dataframe(matrix.style.background_gradient(cmap='RdYlGn_r', axis=None, vmin=-0.5, vmax=1.5).format("{:.2f}"), use_container_width=True, height=500)
    except:
        st.warning("Dados insuficientes para gerar a matriz completa.")

with tab3:
    st.dataframe(df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']], use_container_width=True, hide_index=True)
    