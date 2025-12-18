import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, date
import requests
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
import urllib3
import logging

# --- CONFIGURAÇÃO INICIAL ---
# Desabilita avisos de SSL (Necessário para APIs do Governo)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(
    page_title="VPL Consultoria - Inteligência Financeira",
    page_icon="📈",
    layout="wide"
)

# Estilo CSS Personalizado
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .stBadge {
        font-family: 'Source Sans Pro', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÕES GERAIS ---
class Config:
    """Configurações visuais e constantes"""
    CORES = {
        'auto': '#28a745',   # Verde Sucesso
        'manual': '#ffc107', # Amarelo Alerta
        'erro': '#dc3545',   # Vermelho Erro
        'tema_azul': '#00BFFF',
        'tema_verde': '#00FF7F',
        'tema_dourado': '#FFD700',
        'tema_branco': '#FFFFFF',
        'tema_vermelho': '#FF6347'
    }
    HEADERS = {'User-Agent': 'Mozilla/5.0'}

# Paleta para Matrizes (Verde/Branco/Vermelho Suave)
cores_leves = ["#FFB3B3", "#FFFFFF", "#B3FFB3"]
cmap_leves = LinearSegmentedColormap.from_list("pastel_rdylgn", cores_leves)

# --- SISTEMA DE STATUS (BADGES) ---
class StatusFonte:
    """Gerencia os badges de status da fonte (Singleton no Session State)"""
    if 'status_dict' not in st.session_state:
        st.session_state['status_dict'] = {}

    @staticmethod
    def set(chave, tipo, fonte):
        st.session_state['status_dict'][chave] = {'tipo': tipo, 'fonte': fonte}

    @staticmethod
    def get_badge(chave):
        status = st.session_state['status_dict'].get(chave, {'tipo': 'indefinido', 'fonte': '-'})
        cor = Config.CORES.get(status['tipo'], '#6c757d')
        icone = "🟢" if status['tipo'] == 'auto' else "🟠" if status['tipo'] == 'manual' else "🔴"
        texto = "Automático" if status['tipo'] == 'auto' else "Manual" if status['tipo'] == 'manual' else "Erro"
        
        # Badge HTML estilizado
        return f"""
        <div style="text-align: right; margin-bottom: 5px;">
            <span style="background-color: {cor}15; color: {cor}; 
                         border: 1px solid {cor}40; padding: 2px 10px; 
                         border-radius: 12px; font-size: 0.75em; font-weight: 600;">
                {icone} {texto} • {status['fonte']}
            </span>
        </div>
        """

# --- DADOS DE FALLBACK (SEGURANÇA) ---
def get_fallback_focus():
    """Dados manuais caso a API do Focus falhe"""
    dados = [
        {'Indicador': 'IPCA', 'ano_referencia': 2025, 'previsao': 3.80},
        {'Indicador': 'Selic', 'ano_referencia': 2025, 'previsao': 9.00},
        {'Indicador': 'PIB Total', 'ano_referencia': 2025, 'previsao': 2.00},
        {'Indicador': 'Câmbio', 'ano_referencia': 2025, 'previsao': 5.40},
        {'Indicador': 'IPCA', 'ano_referencia': 2026, 'previsao': 3.50},
        {'Indicador': 'Selic', 'ano_referencia': 2026, 'previsao': 8.50},
        {'Indicador': 'Câmbio', 'ano_referencia': 2026, 'previsao': 5.45},
        {'Indicador': 'PIB Total', 'ano_referencia': 2026, 'previsao': 2.00},
    ]
    return pd.DataFrame(dados)

def get_fallback_macro():
    """Dados manuais de macroeconomia para caso de erro"""
    return {
        'PIB (R$ Bi)': 11000.00, 'Dívida Líq. (% PIB)': 62.5,
        'Res. Primário (% PIB)': -0.6, 'Res. Nominal (% PIB)': -7.5,
        'Balança Com. (US$ Mi)': 80000, 'Trans. Correntes (US$ Mi)': -40000,
        'IDP (US$ Mi)': 65000
    }

# --- PROCESSAMENTO COMUM ---
def processar_dataframe_comum(df):
    if df.empty: return df
    df = df.sort_values('data_date', ascending=True)
    df['mes_num'] = df['data_date'].dt.month
    meses_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    df['mes_nome'] = df['mes_num'].map(meses_map)
    df['data_fmt'] = df['mes_nome'] + '/' + df['ano']
    
    # Cálculos financeiros
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
        'is_reverso': is_reverso
    }, None

# --- FUNÇÕES DE CARGA DE DADOS (COM ROBUSTEZ) ---

@st.cache_data(ttl=3600)
def get_focus_data_pro():
    """Busca Focus com correção de duplicatas, tratamento de erro e status"""
    try:
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$top=5000&$orderby=Data%20desc&$format=json"
        response = requests.get(url, headers=Config.HEADERS, verify=False, timeout=6)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json()['value'])
        
        indicadores = [
            'IPCA', 'PIB Total', 'Selic', 'Câmbio', 'IGP-M', 'IPCA Administrados', 
            'Conta corrente', 'Balança comercial', 'Investimento direto no país', 
            'Dívida líquida do setor público', 'Resultado primário', 'Resultado nominal'
        ]
        
        df = df[df['Indicador'].isin(indicadores)].copy()
        df = df.rename(columns={'Data': 'data_relatorio', 'DataReferencia': 'ano_referencia', 'Mediana': 'previsao'})
        
        df['ano_referencia'] = pd.to_numeric(df['ano_referencia'], errors='coerce')
        df['previsao'] = pd.to_numeric(df['previsao'], errors='coerce')
        df['data_relatorio'] = pd.to_datetime(df['data_relatorio'])
        
        # CORREÇÃO DE VALORES: Ordena por data e pega o primeiro (mais recente)
        df = df.sort_values('data_relatorio', ascending=False)
        df = df.drop_duplicates(subset=['Indicador', 'ano_referencia'], keep='first')
        
        StatusFonte.set('focus', 'auto', 'BCB/Olinda')
        return df
    except Exception:
        StatusFonte.set('focus', 'manual', 'Fallback (Erro API)')
        return get_fallback_focus()

@st.cache_data(ttl=3600)
def get_macro_real_pro():
    """Busca dados macro com Fallback e Status"""
    series = {
        'PIB (R$ Bi)': 4382, 'Dívida Líq. (% PIB)': 4513,
        'Res. Primário (% PIB)': 5362, 'Res. Nominal (% PIB)': 5360,
        'Balança Com. (US$ Mi)': 22707, 'Trans. Correntes (US$ Mi)': 22724,
        'IDP (US$ Mi)': 22885
    }
    resultados = {}
    
    try:
        for nome, codigo in series.items():
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/13?formato=json"
            resp = requests.get(url, headers=Config.HEADERS, verify=False, timeout=5)
            if resp.status_code == 200:
                df = pd.DataFrame(resp.json())
                valor = pd.to_numeric(df['valor']).iloc[-1]
                
                if 'PIB' in nome and 'R$' in nome: valor /= 1_000_000
                elif 'Balança' in nome or 'Trans.' in nome or 'IDP' in nome:
                    valor = pd.to_numeric(df['valor']).iloc[-12:].sum() / 1_000
                elif 'Primário' in nome or 'Nominal' in nome: valor *= -1
                
                resultados[nome] = valor
        
        if resultados:
            StatusFonte.set('macro', 'auto', 'BCB/SGS')
            return resultados
        raise Exception("Vazio")
    except:
        StatusFonte.set('macro', 'manual', 'Fallback Interno')
        return get_fallback_macro()

@st.cache_data(ttl=86400)
def get_cambio_historico_pro():
    """Busca histórico de câmbio com Status"""
    try:
        df = yf.download(["USDBRL=X", "EURBRL=X"], start="1994-07-01", progress=False)
        if df.empty: raise Exception("Vazio")
        
        df = df['Close']
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Sao_Paulo').tz_localize(None)
        
        hoje = pd.Timestamp.now().normalize()
        df = df[df.index <= hoje]
        df = df.rename(columns={'USDBRL=X': 'Dólar', 'EURBRL=X': 'Euro'}).ffill()
        
        StatusFonte.set('cambio_hist', 'auto', 'Yahoo Finance')
        return df
    except:
        StatusFonte.set('cambio_hist', 'erro', 'Indisponível')
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_cotacao_realtime():
    """Busca cotação tempo real com Status"""
    try:
        url = "https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        df = pd.DataFrame([
            {'moeda': 'Dólar', 'bid': float(data['USDBRL']['bid']), 'pct': float(data['USDBRL']['pctChange'])},
            {'moeda': 'Euro', 'bid': float(data['EURBRL']['bid']), 'pct': float(data['EURBRL']['pctChange'])}
        ]).set_index('moeda')
        return df
    except:
        return pd.DataFrame()

# Wrappers para Indicador Principal
@st.cache_data
def get_indicador_principal(tipo):
    try:
        if "IPCA" in tipo:
            raw = sidrapy.get_table(table_code="1737", territorial_level="1", ibge_territorial_code="all", variable="63", period="last 360")
            fonte = "IBGE/Sidra"
        elif "INPC" in tipo:
            raw = sidrapy.get_table(table_code="1736", territorial_level="1", ibge_territorial_code="all", variable="44", period="last 360")
            fonte = "IBGE/Sidra"
        elif "IGP-M" in tipo: return get_bcb_wrapper("189", "FGV/BCB")
        elif "SELIC" in tipo: return get_bcb_wrapper("4390", "BCB")
        elif "CDI" in tipo: return get_bcb_wrapper("4391", "CETIP/BCB")
        else: return pd.DataFrame()

        # Processamento SIDRA
        if raw.empty: raise ValueError()
        df = raw.iloc[1:].copy()
        df.rename(columns={'V': 'valor', 'D2N': 'mes_ano'}, inplace=True)
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_date'] = pd.to_datetime(df['D2C'], format="%Y%m", errors='coerce')
        df['ano'] = df['D2C'].str.slice(0, 4)
        df['D2C'] = df['data_date'].dt.strftime('%Y%m') # Garante formato para calculadora
        
        StatusFonte.set('principal', 'auto', fonte)
        return processar_dataframe_comum(df)
        
    except:
        StatusFonte.set('principal', 'erro', 'Falha na Carga')
        return pd.DataFrame()

def get_bcb_wrapper(codigo, nome_fonte):
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json"
        resp = requests.get(url, headers=Config.HEADERS, verify=False, timeout=10)
        df = pd.DataFrame(resp.json())
        df['data_date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        df['D2C'] = df['data_date'].dt.strftime('%Y%m')
        df['ano'] = df['data_date'].dt.strftime('%Y')
        
        StatusFonte.set('principal', 'auto', nome_fonte)
        return processar_dataframe_comum(df)
    except:
        StatusFonte.set('principal', 'erro', 'Falha na Carga')
        return pd.DataFrame()

# ==============================================================================
# INTERFACE PRINCIPAL (MAIN)
# ==============================================================================
def main():
    # --- SIDEBAR ---
    try:
        st.sidebar.image("Logo_VPL_Consultoria_Financeira.png", use_container_width=True)
    except:
        st.sidebar.title("VPL CONSULTORIA")
    
    st.sidebar.header("Parâmetros")
    tipo_indice = st.sidebar.selectbox(
        "Indicador Principal",
        ["IPCA (Inflação Oficial)", "INPC (Salários)", "IGP-M (Aluguéis)", "SELIC (Taxa Básica)", "CDI (Investimentos)"]
    )

    # Definição de Cores do Tema
    if "IPCA" in tipo_indice: cor_tema = Config.CORES['tema_azul']
    elif "INPC" in tipo_indice: cor_tema = Config.CORES['tema_verde']
    elif "IGP-M" in tipo_indice: cor_tema = Config.CORES['tema_vermelho']
    elif "SELIC" in tipo_indice: cor_tema = Config.CORES['tema_dourado']
    else: cor_tema = Config.CORES['tema_branco']

    # --- CARGA DE DADOS ---
    with st.spinner("Sincronizando dados com fontes oficiais..."):
        df_ind = get_indicador_principal(tipo_indice)
        df_focus = get_focus_data_pro()
        dados_macro = get_macro_real_pro()
        df_cambio_hist = get_cambio_historico_pro()
        df_cotacoes = get_cotacao_realtime()

    if df_ind.empty:
        st.error("Falha crítica ao carregar o indicador principal. Verifique sua conexão.")
        st.stop()

    # --- CALCULADORA NA SIDEBAR ---
    st.sidebar.divider()
    st.sidebar.subheader("🧮 Calculadora")
    valor_input = st.sidebar.number_input("Valor (R$)", value=1000.00, step=100.00, format="%.2f")
    
    lista_anos = sorted(df_ind['ano'].unique(), reverse=True)
    lista_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    mapa_meses = {m: f"{i+1:02d}" for i, m in enumerate(lista_meses)}
    
    c1, c2 = st.sidebar.columns(2)
    mes_ini = c1.selectbox("Início", lista_meses, index=0)
    ano_ini = c2.selectbox("Ano", lista_anos, index=1 if len(lista_anos)>1 else 0)
    
    c3, c4 = st.sidebar.columns(2)
    mes_fim = c3.selectbox("Fim", lista_meses, index=11)
    ano_fim = c4.selectbox("Ano ", lista_anos, index=0)
    
    if st.sidebar.button("Calcular Correção", type="primary", use_container_width=True):
        code_ini = f"{ano_ini}{mapa_meses[mes_ini]}"
        code_fim = f"{ano_fim}{mapa_meses[mes_fim]}"
        res, erro = calcular_correcao(df_ind, valor_input, code_ini, code_fim)
        
        if erro:
            st.sidebar.error(erro)
        else:
            st.sidebar.success("Cálculo Realizado!")
            st.sidebar.markdown(f"<h3 style='color: {cor_tema};'>R$ {res['valor_final']:,.2f}</h3>", unsafe_allow_html=True)
            st.sidebar.caption(f"Variação: {res['percentual']:.2f}% | Fator: {res['fator']:.6f}")

    # --- PAINEL 1: FOCUS & CÂMBIO ---
    with st.expander("🔭 Expectativas de Mercado (Focus) & Câmbio", expanded=True):
        col_tit, col_sts = st.columns([3, 1])
        col_sts.markdown(StatusFonte.get_badge('focus'), unsafe_allow_html=True)
        
        c_focus, c_cambio = st.columns([2, 1])
        
        with c_focus:
            if not df_focus.empty:
                ano = date.today().year
                df_atual = df_focus[df_focus['ano_referencia'] == ano]
                
                # KPIs Focus
                try:
                    pivot = df_atual.set_index('Indicador')['previsao']
                    k1, k2, k3 = st.columns(3)
                    k1.metric(f"IPCA {ano}", f"{pivot.get('IPCA',0):.2f}%")
                    k2.metric(f"Selic {ano}", f"{pivot.get('Selic',0):.2f}%")
                    k3.metric(f"PIB {ano}", f"{pivot.get('PIB Total',0):.2f}%")
                    
                    st.markdown("###### Projeções 2025-2027")
                    cols = [ano, ano+1, ano+2]
                    df_tab = df_focus[df_focus['ano_referencia'].isin(cols)]
                    df_pivot = df_tab.pivot(index='Indicador', columns='ano_referencia', values='previsao')
                    
                    # Formatação condicional
                    df_display = df_pivot.copy()
                    for col in df_display.columns:
                        def fmt(row):
                            val = row[col]
                            nome = row.name
                            if pd.isna(val): return "-"
                            if 'Câmbio' in nome: return f"R$ {val:.2f}"
                            elif any(x in nome for x in ['comercial', 'Conta corrente', 'Investimento']): return f"US$ {val:.2f} B"
                            else: return f"{val:.2f}%"
                        df_display[col] = df_display.apply(fmt, axis=1)

                    st.dataframe(df_display, use_container_width=True, height=300)
                except: st.error("Erro ao renderizar Focus")
        
        with c_cambio:
            st.markdown("###### Câmbio Agora")
            if not df_cotacoes.empty:
                usd = df_cotacoes.loc['Dólar']
                eur = df_cotacoes.loc['Euro']
                st.metric("Dólar", f"R$ {usd['bid']:.2f}", f"{usd['pct']:.2f}%")
                st.metric("Euro", f"R$ {eur['bid']:.2f}", f"{eur['pct']:.2f}%")
            else: st.info("Cotação Indisponível")

    # --- PAINEL 2: MACROECONOMIA ---
    with st.expander("🧩 Conjuntura Macroeconômica (Realizado)", expanded=False):
        col_tit, col_sts = st.columns([3, 1])
        col_sts.markdown(StatusFonte.get_badge('macro'), unsafe_allow_html=True)
        
        if dados_macro:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PIB (12m)", f"R$ {dados_macro.get('PIB (R$ Bi)',0):.2f} T")
            m2.metric("Dívida Líq.", f"{dados_macro.get('Dívida Líq. (% PIB)',0):.1f}%")
            m3.metric("Balança Com.", f"US$ {dados_macro.get('Balança Com. (US$ Mi)',0):.1f} B")
            m4.metric("IDP (12m)", f"US$ {dados_macro.get('IDP (US$ Mi)',0):.1f} B")

    # --- PAINEL 3: HISTÓRICO DE CÂMBIO ---
    with st.expander("💸 Histórico de Câmbio (Longo Prazo)", expanded=False):
        st.markdown(StatusFonte.get_badge('cambio_hist'), unsafe_allow_html=True)
        if not df_cambio_hist.empty:
            tab_g, tab_m, tab_t = st.tabs(["Gráfico", "Matriz Mensal", "Tabela"])
            with tab_g:
                fig_c = px.line(df_cambio_hist, x=df_cambio_hist.index, y=['Dólar', 'Euro'], color_discrete_map={"Dólar": "#00FF7F", "Euro": "#00BFFF"})
                fig_c.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig_c, use_container_width=True)
            with tab_m:
                st.caption("Variação % Mensal do Dólar")
                df_m = df_cambio_hist[['Dólar']].resample('ME').last().pct_change()*100
                df_m['ano'] = df_m.index.year
                df_m['mes'] = df_m.index.month_name().str.slice(0,3)
                try:
                    mat = df_m.pivot(index='ano', columns='mes', values='Dólar')
                    meses = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                    mat = mat[meses].sort_index(ascending=False)
                    st.dataframe(mat.style.background_gradient(cmap=cmap_leves, vmin=-5, vmax=5).format("{:.2f}"), use_container_width=True)
                except: st.info("Matriz indisponível")
            with tab_t:
                csv = df_cambio_hist.to_csv().encode('utf-8')
                st.download_button("📥 Baixar CSV Câmbio", csv, "cambio_full.csv", "text/csv")
                st.dataframe(df_cambio_hist.sort_index(ascending=False).head(100), use_container_width=True)

    # --- PAINEL 4: INDICADOR PRINCIPAL ---
    st.divider()
    col_ind_tit, col_ind_sts = st.columns([3, 1])
    col_ind_tit.title(f"📊 Painel: {tipo_indice.split()[0]}")
    col_ind_sts.markdown(StatusFonte.get_badge('principal'), unsafe_allow_html=True)

    if not df_ind.empty:
        ult = df_ind.iloc[0]
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Taxa Mensal", f"{ult['valor']:.2f}%", ult['data_fmt'])
        i2.metric("Acum. 12 Meses", f"{ult['acum_12m']:.2f}%")
        i3.metric("Acum. Ano", f"{ult['acum_ano']:.2f}%")
        i4.metric("Início da Série", df_ind['ano'].min())

        tab_main1, tab_main2, tab_main3 = st.tabs(["Gráfico de Tendência", "Matriz de Calor", "Dados Históricos"])
        
        with tab_main1:
            df_chart = df_ind.head(60) # Últimos 5 anos
            fig = px.line(df_chart, x='data_date', y='acum_12m', title=f"Acumulado 12M - {tipo_indice.split()[0]}")
            fig.update_traces(line_color=cor_tema, line_width=3)
            fig.update_layout(template="plotly_dark", height=400, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab_main2:
            try:
                matrix = df_ind.pivot(index='ano', columns='mes_nome', values='valor')
                ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                matrix = matrix[ordem].sort_index(ascending=False)
                st.dataframe(matrix.style.background_gradient(cmap=cmap_leves, axis=None, vmin=-1.5, vmax=1.5).format("{:.2f}"), use_container_width=True)
            except: st.warning("Matriz indisponível.")
            
        with tab_main3:
            csv_ind = df_ind.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar CSV Indicador", csv_ind, f"{tipo_indice.split()[0]}.csv", "text/csv")
            st.dataframe(df_ind[['data_fmt', 'valor', 'acum_ano', 'acum_12m']], use_container_width=True)

if __name__ == "__main__":
    main()
