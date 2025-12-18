import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
import requests
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap

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
# -------------------------------------------------------

# --- FUNÇÕES DE CARGA DE DADOS ---

# 1. IBGE (Sidra)
@st.cache_data
def get_sidra_data(table_code, variable_code):
    try:
        dados_raw = sidrapy.get_table(
            table_code=table_code, territorial_level="1", ibge_territorial_code="all", 
            variable=variable_code, period="last 360"
        )
        df = dados_raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'])
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m")
        df['ano'] = df['D2C'].str.slice(0, 4)
        return processar_dataframe_comum(df)
    except Exception as e:
        return pd.DataFrame()

# 2. Banco Central (SGS - Índices Mensais)
@st.cache_data
def get_bcb_data(codigo_serie):
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
        
        # --- A CORREÇÃO MÁGICA ---
        # Fingimos ser um navegador e ignoramos o erro de certificado SSL do governo
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status() # Garante que não foi erro 404 ou 500
        
        # Cria o DataFrame a partir do JSON da resposta
        df = pd.DataFrame(response.json())
        # -------------------------

        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        return processar_dataframe_comum(df)
    except Exception as e:
        # Dica: Se quiser debugar, descomente o print abaixo
        # print(f"Erro BCB {codigo_serie}: {e}")
        return pd.DataFrame()

# 3. Boletim Focus (Expandido com Fiscal e Externo)
@st.cache_data(ttl=3600)
def get_focus_data():
    try:
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=1000&$orderby=Data%20desc&$format=json"
        response = requests.get(url)
        data_json = response.json()
        df = pd.DataFrame(data_json['value'])
        
        # Lista expandida com os novos indicadores
        indicadores = [
            'IPCA', 'PIB Total', 'Selic', 'Câmbio', 'IGP-M',
            'IPCA Administrados', 'Conta corrente', 'Balança comercial',
            'Investimento direto no país', 'Dívida líquida do setor público',
            'Resultado primário', 'Resultado nominal'
        ]
        
        df = df[df['Indicador'].isin(indicadores)]
        df = df.rename(columns={'Data': 'data_relatorio', 'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'})
        df['ano_referencia'] = df['ano_referencia'].astype(int)
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        return df
    except:
        return pd.DataFrame()

# 4. Cotação de Moedas (Tempo Real - Via Yahoo Finance)
@st.cache_data(ttl=300)
def get_currency_realtime():
    try:
        tickers = ["USDBRL=X", "EURBRL=X"]
        dados = {}
        for t in tickers:
            ticker_obj = yf.Ticker(t)
            preco_atual = ticker_obj.fast_info['last_price']
            fechamento_anterior = ticker_obj.fast_info['previous_close']
            variacao = ((preco_atual - fechamento_anterior) / fechamento_anterior) * 100
            key = t.replace("=X", "") 
            dados[key] = {'bid': preco_atual, 'pctChange': variacao}
        df = pd.DataFrame.from_dict(dados, orient='index')
        return df
    except Exception as e:
        return pd.DataFrame()

# 5. Histórico de Câmbio (Via Yahoo Finance - Com Correção de Fuso)
@st.cache_data(ttl=86400)
def get_cambio_historico():
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="1994-07-01", progress=False)
        df = df['Close']
        
        # --- CORREÇÃO DE DATA/FUSO ---
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo')
        df.index = df.index.tz_localize(None)
        
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        # -----------------------------

        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'})
        df = df.ffill()
        return df
    except Exception as e:
        return pd.DataFrame()

# 6. Processamento Comum
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

# 7. Dados Macroeconômicos Reais (Com Datas)
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
    
    resultados = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Mapa de meses para garantir português
    mapa_meses = {'01':'jan', '02':'fev', '03':'mar', '04':'abr', '05':'mai', '06':'jun',
                  '07':'jul', '08':'ago', '09':'set', '10':'out', '11':'nov', '12':'dez'}
    
    try:
        for nome, codigo in series.items():
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/13?formato=json"
            resp = requests.get(url, headers=headers, verify=False, timeout=5)
            df = pd.DataFrame(resp.json())
            df['valor'] = pd.to_numeric(df['valor'])
            
            # --- CAPTURA E FORMATAÇÃO DA DATA ---
            # Pega a data do último registro
            data_raw = df['data'].iloc[-1] # Vem como dd/mm/aaaa
            dia, mes, ano = data_raw.split('/')
            data_curta = f"{mapa_meses[mes]}/{ano[2:]}" # Ex: out/25
            # ------------------------------------

            if nome == 'PIB':
                valor = df['valor'].iloc[-1] / 1_000_000 # Trilhões
            elif nome in ['Balança Com.', 'Trans. Correntes', 'IDP']:
                valor = df['valor'].iloc[-12:].sum() / 1_000 # Bilhões (Soma 12m)
            elif 'Primário' in nome or 'Nominal' in nome:
                valor = df['valor'].iloc[-1] * -1 # Inverte sinal
            else:
                valor = df['valor'].iloc[-1] # Dívida
                
            resultados[nome] = {'valor': valor, 'data': data_curta}
            
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
# Tenta carregar logo, se não existir, segue sem
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
        cor_tema = "#00BFFF" # Azul Neon (Ajustado para Dark)
    elif "INPC" in tipo_indice:
        df = get_sidra_data("1736", "44")
        cor_tema = "#00FF7F" # Verde Neon
    elif "IGP-M" in tipo_indice:
        df = get_bcb_data("189")
        cor_tema = "#FF6347" # Vermelho Tomate (Mais visível no escuro)
    elif "SELIC" in tipo_indice:
        df = get_bcb_data("4390")
        cor_tema = "#FFD700" # Dourado
    elif "CDI" in tipo_indice:
        df = get_bcb_data("4391")
        cor_tema = "#FFFFFF" # Branco

if df.empty:
    st.error("Erro ao carregar dados.")
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
        # MELHORIA: EXIBIR FATOR
        st.sidebar.markdown(f"Fator de Correção: **{res['fator']:.6f}**")

# ==============================================================================
# ÁREA SUPERIOR: EXPANDERS
# ==============================================================================

# 1. BOLETIM FOCUS (COM NOVOS INDICADORES MACRO)
with st.expander("🔭 Clique para ver: Expectativas de Mercado (Focus) & Câmbio Hoje", expanded=False):
    col_top1, col_top2 = st.columns([2, 1])
    
    # FOCUS
    df_focus = get_focus_data()
    ano_atual = date.today().year
    
    with col_top1:
        if not df_focus.empty:
            ultima_data = df_focus['data_relatorio'].max()
            df_last = df_focus[df_focus['data_relatorio'] == ultima_data]
            
            data_str = pd.to_datetime(ultima_data).strftime('%d/%m/%Y')
            st.markdown(f"**Boletim Focus ({data_str})**")
            
            # --- DESTAQUES (Mantivemos os 4 principais no topo) ---
            df_atual = df_last[df_last['ano_referencia'] == ano_atual]
            pivot_atual = df_atual.pivot_table(index='Indicador', values='previsao', aggfunc='mean')
            
            fc1, fc2, fc3, fc4 = st.columns(4)
            # Função auxiliar segura para pegar valor
            def get_val(idx): return pivot_atual.loc[idx, 'previsao'] if idx in pivot_atual.index else 0
            
            fc1.metric(f"IPCA {ano_atual}", f"{get_val('IPCA'):.2f}%")
            fc2.metric(f"Selic {ano_atual}", f"{get_val('Selic'):.2f}%")
            fc3.metric(f"PIB {ano_atual}", f"{get_val('PIB Total'):.2f}%")
            fc4.metric(f"Dólar {ano_atual}", f"R$ {get_val('Câmbio'):.2f}")
            
            # --- TABELA COMPLETA (Setor Externo e Fiscal) ---
            st.divider()
            st.markdown("###### 📅 Projeções Macroeconômicas (2025 - 2027)")
            
            anos_exibir = [ano_atual, ano_atual + 1, ano_atual + 2]
            df_table = df_last[df_last['ano_referencia'].isin(anos_exibir)].copy()
            df_pivot_multi = df_table.pivot_table(index='Indicador', columns='ano_referencia', values='previsao')
            
            # Ordem lógica de exibição
            ordem = [
                'IPCA', 'IGP-M', 'IPCA Administrados', 'Selic', 'Câmbio', 'PIB Total', # Atividade/Inflação
                'Dívida líquida do setor público', 'Resultado primário', 'Resultado nominal', # Fiscal
                'Balança comercial', 'Conta corrente', 'Investimento direto no país' # Externo
            ]
            # Filtra apenas os que existem na resposta para evitar erro
            ordem_final = [x for x in ordem if x in df_pivot_multi.index]
            df_pivot_multi = df_pivot_multi.reindex(ordem_final)
            
            # --- FORMATAÇÃO INTELIGENTE (US$, R$ e %) ---
            df_display = df_pivot_multi.copy()
            
            for col in df_display.columns:
                def formatador_inteligente(row):
                    val = row[col]
                    nome = row.name
                    if pd.isna(val): return "-"
                    
                    if 'Câmbio' in nome:
                        return f"R$ {val:.2f}"
                    elif any(x in nome for x in ['comercial', 'Conta corrente', 'Investimento']):
                        return f"US$ {val:.2f} B" # Bilhões de Dólares
                    else:
                        return f"{val:.2f}%" # O resto é tudo % (PIB, Dívida, Inflação)

                df_display[col] = df_display.apply(formatador_inteligente, axis=1)
            
            st.dataframe(df_display, use_container_width=True)

        else:
            st.warning("Focus indisponível.")

    # MOEDAS (Mantém igual)
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
            except:
                st.info("Erro moedas")
        else:
            st.info("API indisponível")

# ==============================================================================
# NOVO BLOCO: CONJUNTURA MACROECONÔMICA (DADOS REAIS)
# ==============================================================================
# ==============================================================================
# BLOCO ATUALIZADO: CONJUNTURA MACROECONÔMICA (COM DATAS)
# ==============================================================================
with st.expander("🧩 Conjuntura Macroeconômica (Dados Oficiais Realizados)", expanded=False):
    st.markdown("Principais indicadores da economia brasileira (Dados mais recentes do Banco Central).")
    
    macro = get_macro_real()
    
    if macro:
        # Função auxiliar para pegar Valor e Data com segurança
        def get_dado(chave):
            item = macro.get(chave, {'valor': 0, 'data': '--'})
            return item['valor'], item['data']

        # --- LINHA 1: ATIVIDADE E FISCAL ---
        st.markdown("##### 🏛️ Atividade & Fiscal (Acum. 12 Meses)")
        c1, c2, c3, c4 = st.columns(4)
        
        v_pib, d_pib = get_dado('PIB')
        v_div, d_div = get_dado('Dívida Líq.')
        v_pri, d_pri = get_dado('Res. Primário')
        v_nom, d_nom = get_dado('Res. Nominal')
        
        c1.metric(f"PIB ({d_pib})", f"R$ {v_pib:.2f} Tri")
        c2.metric(f"Dív. Líquida ({d_div})", f"{v_div:.1f}% PIB")
        c3.metric(f"Res. Primário ({d_pri})", f"{v_pri:.2f}% PIB")
        c4.metric(f"Res. Nominal ({d_nom})", f"{v_nom:.2f}% PIB")
        
        st.divider()
        
        # --- LINHA 2: SETOR EXTERNO ---
        st.markdown("##### 🚢 Setor Externo (Acum. 12 Meses)")
        c5, c6, c7 = st.columns(3)
        
        v_bal, d_bal = get_dado('Balança Com.')
        v_tra, d_tra = get_dado('Trans. Correntes')
        v_idp, d_idp = get_dado('IDP')
        
        c5.metric(f"Balança Com. ({d_bal})", f"US$ {v_bal:.1f} Bi")
        c6.metric(f"Trans. Correntes ({d_tra})", f"US$ {v_tra:.1f} Bi")
        c7.metric(f"Inv. Direto - IDP ({d_idp})", f"US$ {v_idp:.1f} Bi")
        
    else:
        st.warning("Não foi possível carregar os dados macroeconômicos do BCB.")

# 2. HISTÓRICO DE CÂMBIO (COMPLETO)
with st.expander("💸 Histórico de Câmbio (Dólar e Euro desde 1994)", expanded=False):
    st.markdown("Evolução das moedas frente ao Real (R$) desde o início do Plano Real.")
    
    # Carrega dados diários
    df_cambio = get_cambio_historico()
    
    if not df_cambio.empty:
        # --- 1. RESUMO DO TOPO ---
        ultimo_dado = df_cambio.iloc[-1]
        penultimo_dado = df_cambio.iloc[-2]
        data_atual = df_cambio.index[-1].strftime('%d/%m/%Y')
        
        st.markdown(f"**Fechamento: {data_atual}**")
        col_res1, col_res2, col_res3 = st.columns([1,1,2])
        
        usd_val = ultimo_dado['Dólar']
        usd_var = ((usd_val - penultimo_dado['Dólar']) / penultimo_dado['Dólar']) * 100
        
        eur_val = ultimo_dado['Euro']
        eur_var = ((eur_val - penultimo_dado['Euro']) / penultimo_dado['Euro']) * 100
        
        col_res1.metric("Dólar", f"R$ {usd_val:.2f}", f"{usd_var:.2f}%")
        col_res2.metric("Euro", f"R$ {eur_val:.2f}", f"{eur_var:.2f}%")
        
        st.divider()

        # --- 2. ABAS ---
        tab_graf, tab_matriz, tab_tabela = st.tabs(["📈 Gráfico", "📅 Matriz de Retornos", "📋 Tabela Diária"])
        
        # ABA GRÁFICO
        with tab_graf:
            cores_map = {"Dólar": "#00FF7F", "Euro": "#00BFFF"}
            fig_cambio = px.line(df_cambio, x=df_cambio.index, y=['Dólar', 'Euro'], 
                                 labels={'value': 'Cotação (R$)', 'variable': 'Moeda', 'data': ''},
                                 color_discrete_map=cores_map)
            fig_cambio.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E0E0E0"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig_cambio.update_xaxes(showgrid=False, rangeslider_visible=False)
            fig_cambio.update_yaxes(showgrid=True, gridcolor='#333333', tickprefix="R$ ")
            st.plotly_chart(fig_cambio, use_container_width=True)

        # ABA MATRIZ
        with tab_matriz:
            st.caption("A matriz mostra a variação percentual (%) mês a mês.")
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
                st.dataframe(
                    # Usamos 'cmap_leves' em vez de 'RdYlGn'
                    # Mantemos vmin/vmax simétricos para que o 0 fique branco
                    matrix_cambio.style.background_gradient(cmap=cmap_leves, vmin=-5, vmax=5).format("{:.2f}%"), 
                    use_container_width=True, 
                    height=500
                )
            except Exception as e:
                st.info(f"Dados insuficientes para gerar a matriz completa: {e}")

        # ABA TABELA (COM DOWNLOAD)
        with tab_tabela:
            df_view = df_cambio.sort_index(ascending=False).copy()
            df_view.index.name = "Data"
            df_view = df_view.reset_index()
            df_view['Data'] = df_view['Data'].dt.strftime('%d/%m/%Y')
            
            # Botão Download
            csv_cambio = df_view.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar dados em CSV", csv_cambio, "cambio_historico.csv", "text/csv")
            
            st.dataframe(df_view, use_container_width=True, hide_index=True,
                         column_config={"Dólar": st.column_config.NumberColumn(format="R$ %.4f"), "Euro": st.column_config.NumberColumn(format="R$ %.4f")})

    else:
        st.warning("Não foi possível carregar o histórico do Yahoo Finance.")

# ==============================================================================
# ÁREA PRINCIPAL: DETALHES DO ÍNDICE
# ==============================================================================

st.title(f"📊 Painel: {tipo_indice.split()[0]}")
st.markdown(f"**Dados históricos atualizados até:** {df.iloc[0]['data_fmt']}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Taxa do Mês", f"{df.iloc[0]['valor']:.2f}%")
kpi2.metric("Acumulado 12 Meses", f"{df.iloc[0]['acum_12m']:.2f}%")
kpi3.metric("Acumulado Ano (YTD)", f"{df.iloc[0]['acum_ano']:.2f}%")
kpi4.metric("Início da Série", df['ano'].min())

tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Matriz de Calor", "📋 Tabela Detalhada"])

# GRÁFICO PRINCIPAL (ESTILIZADO)
with tab1:
    df_chart = df.dropna(subset=['acum_12m']).sort_values('data_date')
    indices_volateis = ["IGP-M", "SELIC", "CDI"]
    eh_volatil = any(idx in tipo_indice for idx in indices_volateis)
    if eh_volatil:
        df_chart = df_chart[df_chart['ano'].astype(int) >= 2000]
    
    fig = px.line(df_chart, x='data_date', y='acum_12m', title=f"Histórico 12 Meses - {tipo_indice.split()[0]}")
    
    # Visual Dark
    fig.update_traces(line_color=cor_tema, line_width=3)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E0E0E0"),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_yaxes(showgrid=True, gridcolor='#333333', ticksuffix="%")
    fig.update_xaxes(showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)

# MATRIZ PRINCIPAL
with tab2:
    try:
        matrix = df.pivot(index='ano', columns='mes_nome', values='valor')
        ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        matrix = matrix[ordem].sort_index(ascending=False)
        st.dataframe(
            matrix.style.background_gradient(cmap=cmap_leves, axis=None, vmin=-1.5, vmax=1.5).format("{:.2f}"), 
            use_container_width=True, 
            height=500
        )
    except:
        st.warning("Matriz indisponível.")

# TABELA PRINCIPAL (COM DOWNLOAD)
with tab3:
    # Botão Download
    csv_principal = df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar dados em CSV", csv_principal, f"{tipo_indice.split()[0]}_historico.csv", "text/csv")

    st.dataframe(df[['data_fmt', 'valor', 'acum_ano', 'acum_12m']], use_container_width=True, hide_index=True)
