import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Calculadora IPCA Pro",
    page_icon="💰",
    layout="wide"
)

# Estilo CSS personalizado para ficar com cara de "Sistema"
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #003366;
        padding: 20px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #003366;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÃO DE CARGA DE DADOS (COM CACHE) ---
# O @st.cache_data impede que o app baixe os dados do IBGE a cada clique.
# Ele baixa uma vez e guarda na memória.
@st.cache_data
def get_ipca_data():
    try:
        # Busca últimos 30 anos (aprox 360 meses)
        ipca_raw = sidrapy.get_table(
            table_code="1737", 
            territorial_level="1", 
            ibge_territorial_code="all", 
            variable="63", 
            period="last 360"
        )
        
        # Limpeza
        df = ipca_raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'])
        df['ano'] = df['D2C'].str.slice(0, 4)
        df['mes_num'] = df['D2C'].str.slice(4, 6).astype(int)
        
        # Mapeamento de meses
        meses_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                     7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
        df['mes_nome'] = df['mes_num'].map(meses_map)
        df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
        
        # Cálculos Auxiliares (Fator, Acumulados)
        df = df.sort_values('D2C', ascending=True) # Cronológico
        df['fator'] = 1 + (df['valor'] / 100)
        df['acum_ano'] = (df.groupby('ano')['fator'].cumprod() - 1) * 100
        df['acum_12m'] = (df['fator'].rolling(window=12).apply(np.prod, raw=True) - 1) * 100
        
        return df.sort_values('D2C', ascending=False) # Retorna do mais recente pro antigo
    except Exception as e:
        st.error(f"Erro ao buscar dados do IBGE: {e}")
        return pd.DataFrame()

# --- FUNÇÃO DE CÁLCULO INTELIGENTE (IDA E VOLTA) ---
def calcular_correcao(df, valor, data_ini_code, data_fim_code):
    # Detecta se é cálculo reverso (Deflação/Volta no tempo)
    is_reverso = data_ini_code > data_fim_code
    
    # Define o intervalo cronológico correto para filtrar o DataFrame
    if is_reverso:
        periodo_inicio = data_fim_code
        periodo_fim = data_ini_code
    else:
        periodo_inicio = data_ini_code
        periodo_fim = data_fim_code
        
    # Filtra os dados
    mask = (df['D2C'] >= periodo_inicio) & (df['D2C'] <= periodo_fim)
    df_periodo = df.loc[mask].copy()
    
    if df_periodo.empty:
        return None, "Período sem dados suficientes."
        
    # Calcula o Fator Acumulado do período
    fator_acumulado = df_periodo['fator'].prod()
    
    # Aplica a matemática financeira
    if is_reverso:
        # Trazendo valor futuro para presente (Divisão)
        valor_final = valor / fator_acumulado
    else:
        # Levando valor presente para futuro (Multiplicação)
        valor_final = valor * fator_acumulado
        
    pct_total = (fator_acumulado - 1) * 100
    
    return {
        'valor_final': valor_final,
        'percentual': pct_total,
        'fator': fator_acumulado,
        'is_reverso': is_reverso,
        'meses': len(df_periodo)
    }, None

# --- CARREGANDO DADOS ---
with st.spinner("Conectando ao IBGE..."):
    df = get_ipca_data()

if df.empty:
    st.stop()

# --- INTERFACE (SIDEBAR) ---
st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True) 

st.sidebar.header("Parâmetros")

valor_input = st.sidebar.number_input("Valor (R$)", value=1000.00, step=100.00, format="%.2f")

st.sidebar.subheader("Período")

# Preparar listas para os dropdowns
lista_anos = sorted(df['ano'].unique(), reverse=True)
lista_meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
mapa_meses = {'Jan': '01', 'Fev': '02', 'Mar': '03', 'Abr': '04', 'Mai': '05', 'Jun': '06',
              'Jul': '07', 'Ago': '08', 'Set': '09', 'Out': '10', 'Nov': '11', 'Dez': '12'}

# --- DATA INICIAL ---
st.sidebar.markdown("**Data de Referência**")
c1, c2 = st.sidebar.columns(2)
with c1:
    mes_ini = st.selectbox("Mês Ini", lista_meses_nome, index=0, label_visibility="collapsed")
with c2:
    # Tenta selecionar 1 ano atrás como padrão
    idx_ano_ini = 1 if len(lista_anos) > 1 else 0
    ano_ini = st.selectbox("Ano Ini", lista_anos, index=idx_ano_ini, label_visibility="collapsed")

# --- DATA FINAL ---
st.sidebar.markdown("**Data Alvo**")
c3, c4 = st.sidebar.columns(2)
with c3:
    mes_fim = st.selectbox("Mês Fim", lista_meses_nome, index=9, label_visibility="collapsed") # Ex: Outubro
with c4:
    ano_fim = st.selectbox("Ano Fim", lista_anos, index=0, label_visibility="collapsed")

if st.sidebar.button("Calcular Correção", type="primary"):
    # Reconstruir o código D2C (YYYYMM)
    code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
    code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
    
    # Chama a função de cálculo
    res, erro = calcular_correcao(df, valor_input, code_ini, code_fim)
    
    if erro:
        st.error(erro)
    else:
        # (O resto do código de exibição do resultado continua igual...)
        st.sidebar.divider()
        st.sidebar.markdown("### Resultado")
        cor_valor = "#d32f2f" if res['is_reverso'] else "#2e7d32"
        texto_op = "Deflação (Reverso)" if res['is_reverso'] else "Correção (IPCA)"
        
        st.sidebar.markdown(f"<p style='font-size: 12px; margin:0;'>Operação: <b>{texto_op}</b></p>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<h2 style='color: {cor_valor}; margin:0;'>R$ {res['valor_final']:,.2f}</h2>", unsafe_allow_html=True)
        st.sidebar.markdown(f"Inflação Período: **{res['percentual']:.2f}%**")
        st.sidebar.markdown(f"Fator: **{res['fator']:.6f}**")

        
# --- INTERFACE (MAIN) ---
st.title("📊 Painel IPCA")
st.markdown(f"**Dados atualizados até:** {df.iloc[0]['data_fmt']} | Fonte: IBGE (SIDRA)")

# KPIs Topo
col1, col2, col3, col4 = st.columns(4)
col1.metric("IPCA do Último Mês", f"{df.iloc[0]['valor']:.2f}%")
col2.metric("Acumulado 12 Meses", f"{df.iloc[0]['acum_12m']:.2f}%")
col3.metric("Acumulado Ano (YTD)", f"{df.iloc[0]['acum_ano']:.2f}%")
col4.metric("Início da Série", df['ano'].min())

st.divider()

# Abas
tab1, tab2, tab3 = st.tabs(["📈 Evolução Gráfica", "📅 Matriz de Calor", "📋 Tabela Detalhada"])

with tab1:
    st.subheader("IPCA Acumulado (12 Meses)")
    df_chart = df.dropna(subset=['acum_12m']).sort_values('D2C')
    
    fig = px.line(df_chart, x='mes_ano', y='acum_12m', markers=False)
    fig.update_traces(line_color='#003366', line_width=3)
    fig.update_layout(yaxis_title="%", xaxis_title=None, height=400)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Mapa de Calor da Inflação Mensal")
    # Pivotando para Matriz
    matrix = df.pivot(index='ano', columns='mes_nome', values='valor')
    ordem_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    matrix = matrix[ordem_meses].sort_index(ascending=False)
    
    # Usando o dataframe com gradiente de cor nativo do Streamlit
    st.dataframe(
        matrix.style.background_gradient(cmap='RdYlGn_r', axis=None, vmin=-0.5, vmax=1.5).format("{:.2f}"),
        use_container_width=True,
        height=600
    )

with tab3:
    st.subheader("Dados Históricos")
    
    df_show = df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].copy()
    df_show.columns = ['Mês/Ano', 'IPCA Mês (%)', 'Acum. Ano (%)', 'Acum. 12m (%)']
    
    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "IPCA Mês (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "Acum. Ano (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "Acum. 12m (%)": st.column_config.NumberColumn(format="%.2f %%"),
        }
    )