import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="VPL Consultoria - Calculadora Pro",
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
    /* Ajuste para tabela do Focus */
    .stDataFrame { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE CARGA DE DADOS ---

# 1. IBGE (Sidra)
@st.cache_data
def get_sidra_data(table_code, variable_code):
    try:
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
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m")
        df['ano'] = df['D2C'].str.slice(0, 4)
        return processar_dataframe_comum(df)
    except Exception as e:
        st.error(f"Erro ao buscar dados do IBGE: {e}")
        return pd.DataFrame()

# 2. Banco Central (SGS - Séries Históricas)
@st.cache_data
def get_bcb_data(codigo_serie):
    try:
        url = f"http://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
        df = pd.read_json(url)
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        return processar_dataframe_comum(df)
    except Exception as e:
        st.error(f"Erro ao buscar dados do BCB: {e}")
        return pd.DataFrame()

# 3. NOVO: Banco Central (API Olinda - Boletim Focus)
@st.cache_data
def get_focus_data():
    try:
        # Busca as últimas 500 entradas da expectativa de mercado anual
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=1000&$orderby=Data desc&$format=json"
        
        # Lendo o JSON
        data_json = pd.read_json(url)
        df = pd.DataFrame(data_json['value'].tolist())
        
        # Filtros básicos (queremos apenas os indicadores principais)
        indicadores = ['IPCA', 'PIB Total', 'Selic', 'Câmbio']
        df = df[df['Indicador'].isin(indicadores)]
        
        # Renomear para facilitar
        df = df.rename(columns={
            'Data': 'data_relatorio',
            'DataReferencia': 'ano_referencia',
            'Mediana': 'previsao'
        })
        
        # Converter data
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        
        return df
    except Exception as e:
        st.error(f"Erro ao buscar Focus: {e}")
        return pd.DataFrame()

# 4. Processamento Comum
def processar_dataframe_comum(df):
    df = df.sort_values('data_date', ascending=True)
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    df['fator'] = 1 + (df['valor'] / 100)
    df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
    df['acum_12m'] = (df['fator'].rolling(window=12).apply(np.prod, raw=True) - 1) * 100
    return df.sort_values('data_date', ascending=False)

# --- CÁLCULO DA CALCULADORA ---
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

# --- SIDEBAR ---
st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
st.sidebar.header("Configurações")

tipo_indice = st.sidebar.selectbox(
    "Selecione o Indicador",
    ["IPCA (Inflação Oficial)", "INPC (Salários)", "IGP-M (Aluguéis)", "SELIC (Taxa Básica)", "CDI (Investimentos)"]
)

with st.spinner(f"Carregando dados..."):
    if "IPCA" in tipo_indice:
        df = get_sidra_data("1737", "63")
        cor_tema = "#003366"
    elif "INPC" in tipo_indice:
        df = get_sidra_data("1736", "44")
        cor_tema = "#2E8B57"
    elif "IGP-M" in tipo_indice:
        df = get_bcb_data("189")
        cor_tema = "#8B0000"
    elif "SELIC" in tipo_indice:
        df = get_bcb_data("4390")
        cor_tema = "#DAA520"
    elif "CDI" in tipo_indice:
        df = get_bcb_data("4391")
        cor_tema = "#333333"

if df.empty:
    st.stop()

# --- INPUTS CALCULADORA ---
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

# --- MAIN ---
st.title(f"📊 Painel Financeiro: {tipo_indice.split()[0]}")
st.markdown(f"**Dados até:** {df.iloc[0]['data_fmt']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Taxa do Mês", f"{df.iloc[0]['valor']:.2f}%")
c2.metric("Acumulado 12 Meses", f"{df.iloc[0]['acum_12m']:.2f}%")
c3.metric("Acumulado Ano (YTD)", f"{df.iloc[0]['acum_ano']:.2f}%")
c4.metric("Início da Série", df['ano'].min())

st.divider()

# --- ABAS (AGORA COM FOCUS) ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Gráfico", "📅 Matriz de Calor", "📋 Histórico", "🔮 Boletim Focus"])

with tab1:
    df_chart = df.dropna(subset=['acum_12m']).sort_values('data_date')
    indices_volateis = ["IGP-M", "SELIC", "CDI"]
    eh_volatil = any(idx in tipo_indice for idx in indices_volateis)
    
    if eh_volatil:
        df_chart = df_chart[df_chart['ano'].astype(int) >= 2000]
    
    fig = px.line(df_chart, x='data_date', y='acum_12m', title=f"Histórico 12 Meses")
    fig.update_traces(line_color=cor_tema, line_width=3)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    try:
        matrix = df.pivot(index='ano', columns='mes_nome', values='valor')
        ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        matrix = matrix[ordem].sort_index(ascending=False)
        st.dataframe(matrix.style.background_gradient(cmap='RdYlGn_r', axis=None, vmin=-0.1, vmax=1.5).format("{:.2f}"), use_container_width=True, height=500)
    except:
        st.warning("Matriz indisponível.")

with tab3:
    st.dataframe(df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']], use_container_width=True, hide_index=True)

# --- ABA 4: BOLETIM FOCUS (NOVA) ---
with tab4:
    st.subheader("Expectativas de Mercado (Boletim Focus - BCB)")
    st.markdown("Projeções medianas do mercado para o final de cada ano.")
    
    df_focus = get_focus_data()
    
    if not df_focus.empty:
        # Pega a data mais recente disponível no relatório
        ultima_data = df_focus['data_relatorio'].max()
        df_last = df_focus[df_focus['data_relatorio'] == ultima_data]
        
        data_formatada = pd.to_datetime(ultima_data).strftime('%d/%m/%Y')
        st.info(f"📅 Relatório referente a: **{data_formatada}**")

        # Anos de interesse: Ano Atual e Próximo Ano
        ano_atual = date.today().year
        ano_prox = ano_atual + 1
        
        # Filtrar apenas indicadores para este ano e o próximo
        df_view = df_last[df_last['ano_referencia'].isin([ano_atual, ano_prox])].copy()
        
        # Pivotar para ficar bonito (Indicadores nas linhas, Anos nas colunas)
        # pivot_table lida melhor com duplicatas se houver
        df_pivot = df_view.pivot_table(index='Indicador', columns='ano_referencia', values='previsao', aggfunc='mean')
        
        # Exibir como métricas
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"### 🗓️ Previsão {ano_atual}")
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            
            # Tenta buscar os valores com segurança (get)
            ipca_now = df_pivot.get(ano_atual, {}).get('IPCA', 0)
            selic_now = df_pivot.get(ano_atual, {}).get('Selic', 0)
            pib_now = df_pivot.get(ano_atual, {}).get('PIB Total', 0)
            dolar_now = df_pivot.get(ano_atual, {}).get('Câmbio', 0)
            
            c1.metric("IPCA", f"{ipca_now:.2f}%")
            c2.metric("Selic (Fim)", f"{selic_now:.2f}%")
            c3.metric("PIB", f"{pib_now:.2f}%")
            c4.metric("Dólar", f"R$ {dolar_now:.2f}")

        with col_b:
            st.markdown(f"### 🔭 Previsão {ano_prox}")
            c5, c6 = st.columns(2)
            c7, c8 = st.columns(2)
            
            ipca_next = df_pivot.get(ano_prox, {}).get('IPCA', 0)
            selic_next = df_pivot.get(ano_prox, {}).get('Selic', 0)
            pib_next = df_pivot.get(ano_prox, {}).get('PIB Total', 0)
            dolar_next = df_pivot.get(ano_prox, {}).get('Câmbio', 0)

            c5.metric("IPCA", f"{ipca_next:.2f}%", delta=f"{ipca_next - ipca_now:.2f}%")
            c6.metric("Selic", f"{selic_next:.2f}%", delta=f"{selic_next - selic_now:.2f}%")
            c7.metric("PIB", f"{pib_next:.2f}%", delta=f"{pib_next - pib_now:.2f}%")
            c8.metric("Dólar", f"R$ {dolar_next:.2f}", delta=f"{dolar_next - dolar_now:.2f}")
            
        st.divider()
        st.markdown("**Tabela Completa (Último Relatório)**")
        st.dataframe(df_last[['Indicador', 'ano_referencia', 'previsao']].sort_values(['ano_referencia', 'Indicador']), use_container_width=True, hide_index=True)
        
    else:
        st.warning("Não foi possível carregar os dados do Focus no momento.")