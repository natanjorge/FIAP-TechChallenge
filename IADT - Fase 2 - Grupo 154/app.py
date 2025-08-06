import streamlit as st
import pandas as pd
import numpy as np
import pyodbc
import folium
from streamlit_folium import st_folium
from algoritmos import algoritmo_genetico, nearest_neighbor

colunas_descricao = {
    'ZZZ_CLIENT': 'Código do Cliente',
    'A1_NOME':    'Nome do Cliente',
    'A1_END':     'Endereço',
    'A1_MUN':     'Município',
    'A1_EST':     'Estado',
    'Latitude':   'Latitude',
    'Longitude':  'Longitude'
}

# CSS 
st.markdown("""
    <style>
        .titulo-principal {
            font-size: 2.8rem !important;
            font-weight: bold;
            color: #29b6f6;
            margin-bottom: 2rem;
        }
        .card-bloco {
            background: #181c25;
            border-radius: 15px;
            padding: 24px 18px 6px 18px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px #00000040;
            border: 1px solid #263040;
        }
        .campo-label {
            color: #90caf9;
            font-weight: 600;
            font-size: 1.0rem;
        }
        .campo-valor {
            color: #fff;
            font-weight: 400;
            font-size: 1.2rem;
        }
        .stButton button {
            background: linear-gradient(90deg,#1565c0 70%,#29b6f6 100%);
            color: white !important;
            font-weight: bold;
            border-radius: 10px;
            border: none;
            margin: 10px 0 10px 0;
            padding: 0.7em 2em;
        }
        .stButton button:hover {
            background: linear-gradient(90deg,#29b6f6 70%,#1565c0 100%);
            color: #fff !important;
            font-weight: bold;
            box-shadow: 0 2px 8px #29b6f650;
        }
        .stDataFrame {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="titulo-principal">Otimização de Rotas - ERP Protheus</div>', unsafe_allow_html=True)

def get_conn():
    return pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=EMPRESA1_TESTE;'
        'UID=sa;'
        'PWD=suasenha'
    )

def safe_float(value):
    try:
        if value is None or pd.isnull(value):
            return None
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        s = str(value).replace(",", ".").strip()
        s = s.replace("–", "-")
        if s == '' or s.lower() == 'none':
            return None
        return float(s)
    except Exception:
        return None

with st.spinner("Carregando roteiros..."):
    conn = get_conn()
    roteiros = pd.read_sql(
        "SELECT ZZY_COD, ZZY_MOTORI, ZZY_NOMMOT, ZZY_HORA, ZZY_DATA, ZZY_LATINI, ZZY_LONINI FROM dbo.ZZY990 WHERE D_E_L_E_T_ = ' ' ",
        conn
    )

#  ROTEIRO COM BUSCA 
st.markdown('<div class="card-bloco">', unsafe_allow_html=True)
st.subheader("Escolha um roteiro")

# Campo de busca diretamente acima do selectbox de roteiros
busca = st.text_input(
    "Buscar em Roteiros disponíveis (por código, nome ou motorista):", 
    placeholder="Digite para buscar..."
)

roteiros['opcao'] = roteiros['ZZY_COD'].astype(str) + " - Motorista: " + roteiros['ZZY_NOMMOT'].astype(str)

if busca:
    roteiros_filtrados = roteiros[roteiros['opcao'].str.lower().str.contains(busca.lower())]
else:
    roteiros_filtrados = roteiros

if roteiros_filtrados.empty:
    st.warning("Nenhum roteiro encontrado com esse filtro.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
else:
    roteiro_selecionado = st.selectbox(
        "Roteiros disponíveis:",
        options=roteiros_filtrados['opcao'],
        index=0
    )

st.markdown('</div>', unsafe_allow_html=True)

codigo_roteiro = roteiro_selecionado.split(" - ")[0]
cabecalho = roteiros[roteiros['ZZY_COD'] == codigo_roteiro].iloc[0]

st.markdown('<div class="card-bloco">', unsafe_allow_html=True)
st.subheader("Detalhes do Roteiro Selecionado")

campos_exibir = [
    ('ZZY_COD',    'Código da Viagem'),
    ('ZZY_MOTORI', 'Código Motorista'),
    ('ZZY_NOMMOT', 'Motorista'),
    ('ZZY_HORA',   'Hora da Partida'),
    ('ZZY_DATA',   'Data da Partida'),
    ('ZZY_LATINI', 'Lat Inicial'),
    ('ZZY_LONINI', 'Lon Inicial'),
]
n_colunas = 3
linhas = [campos_exibir[i:i + n_colunas] for i in range(0, len(campos_exibir), n_colunas)]

for linha in linhas:
    cols = st.columns(len(linha))
    for col, (campo, nome) in zip(cols, linha):
        valor = cabecalho.get(campo, '')
        col.markdown(f"<span class='campo-label'>{nome}:</span> <span class='campo-valor'>{valor}</span>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

latini = safe_float(cabecalho.get('ZZY_LATINI'))
lonini = safe_float(cabecalho.get('ZZY_LONINI'))
latfim = latini
lonfim = lonini 

with st.spinner("Buscando entregas e endereços dos clientes..."):
    itens = pd.read_sql(f"SELECT * FROM dbo.ZZZ990 WHERE ZZZ_COD = '{codigo_roteiro}' AND D_E_L_E_T_ = ' ' ", conn)
    entregas = []
    for idx, row in itens.iterrows():
        client = row['ZZZ_CLIENT']
        sa1 = pd.read_sql(
            f"SELECT TOP 1 A1_COD, A1_LOJA, A1_NOME, A1_END, A1_MUN, A1_EST, A1_CEP, A1_ZZLATIT, A1_ZZLONGI "
            f"FROM dbo.SA1990 WHERE A1_COD = '{client}' AND D_E_L_E_T_ = ' ' ",
            conn
        )
        if not sa1.empty:
            lat_raw = sa1['A1_ZZLATIT'].iloc[0]
            lon_raw = sa1['A1_ZZLONGI'].iloc[0]
            lat = safe_float(lat_raw)
            lon = safe_float(lon_raw)
            dados_cliente = sa1.iloc[0].to_dict()
            dados_cliente['A1_ZZLATIT'] = lat
            dados_cliente['A1_ZZLONGI'] = lon
            entrega = {**row, **dados_cliente}
            entregas.append(entrega)
    df_entregas = pd.DataFrame(entregas)

st.markdown('<div class="card-bloco">', unsafe_allow_html=True)
st.subheader("Entregas do roteiro selecionado")
st.info("Edite (se desejar) e clique para otimizar a rota!")
if not df_entregas.empty and 'A1_ZZLATIT' in df_entregas.columns and 'A1_ZZLONGI' in df_entregas.columns:
    df_exibir = df_entregas[
        ['ZZZ_CLIENT', 'A1_NOME', 'A1_END', 'A1_MUN', 'A1_EST', 'A1_ZZLATIT', 'A1_ZZLONGI']
    ].rename(columns={'A1_ZZLATIT': 'Latitude', 'A1_ZZLONGI': 'Longitude'})
    df_exibir = df_exibir.rename(columns=colunas_descricao)
    edited_df = st.data_editor(
        df_exibir.reset_index(drop=True),
        num_rows="dynamic",
        use_container_width=True
    )
else:
    edited_df = pd.DataFrame()
st.markdown('</div>', unsafe_allow_html=True)

def convert_latlon(df):
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df['Latitude'] = df['Latitude'].apply(safe_float)
        df['Longitude'] = df['Longitude'].apply(safe_float)
    return df

def gerar_link_google_maps(pontos):
    base = "https://www.google.com/maps/dir/"
    rotas = "/".join([f"{lat},{lon}" for lat, lon in pontos])
    return base + rotas

def montar_pontos_rota(base_ini, entregas_latlon, base_fim=None):
    pontos = []
    if base_ini:
        pontos.append(base_ini)
    pontos += entregas_latlon
    if base_fim:
        pontos.append(base_fim)
    return pontos

if "rota_completa" not in st.session_state:
    st.session_state["rota_completa"] = None
if "msg_sucesso" not in st.session_state:
    st.session_state["msg_sucesso"] = None
if "cor" not in st.session_state:
    st.session_state["cor"] = None
if "link_maps" not in st.session_state:
    st.session_state["link_maps"] = None
if "dist_ag" not in st.session_state:
    st.session_state["dist_ag"] = None
if "dist_greedy" not in st.session_state:
    st.session_state["dist_greedy"] = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Otimizar Rota (Algoritmo Genético)"):
        try:
            edited_df = convert_latlon(edited_df)
            entregas_latlon = edited_df[['Latitude', 'Longitude']].dropna().values.tolist()
            base_ini = [latini, lonini] if (latini is not None and lonini is not None) else None
            base_fim = [latfim, lonfim] if (latfim is not None and lonfim is not None) else base_ini

            if not entregas_latlon:
                st.error("Nenhuma entrega com latitude/longitude válida.")
            else:
                pontos_otimizacao = entregas_latlon
                ordem_otima, melhor_dist = algoritmo_genetico(np.array(pontos_otimizacao))
                rota_completa = montar_pontos_rota(base_ini, [entregas_latlon[i] for i in ordem_otima], base_fim)
                link_maps = gerar_link_google_maps(rota_completa)
                st.session_state["rota_completa"] = rota_completa
                st.session_state["msg_sucesso"] = f"Rota otimizada!"
                st.session_state["cor"] = "blue"
                st.session_state["link_maps"] = link_maps
                st.session_state["dist_ag"] = melhor_dist
        except Exception as e:
            st.error(f"Erro ao otimizar/plotar rota: {e}")
with col2:
    if st.button("Otimizar Rota (Greedy - Ponto Mais Próximo)"):
        try:
            edited_df = convert_latlon(edited_df)
            entregas_latlon = edited_df[['Latitude', 'Longitude']].dropna().values.tolist()
            base_ini = [latini, lonini] if (latini is not None and lonini is not None) else None
            base_fim = [latfim, lonfim] if (latfim is not None and lonfim is not None) else base_ini

            if not entregas_latlon:
                st.error("Nenhuma entrega com latitude/longitude válida.")
            else:
                pontos_otimizacao = entregas_latlon
                ordem_greedy = nearest_neighbor(np.array(pontos_otimizacao))
                # Calcule a distância da rota greedy
                dist_greedy = 0
                for i in range(1, len(ordem_greedy)):
                    a = np.array(pontos_otimizacao[ordem_greedy[i-1]])
                    b = np.array(pontos_otimizacao[ordem_greedy[i]])
                    dist_greedy += np.linalg.norm(a - b)
                rota_completa = montar_pontos_rota(base_ini, [entregas_latlon[i] for i in ordem_greedy], base_fim)
                link_maps = gerar_link_google_maps(rota_completa)
                st.session_state["rota_completa"] = rota_completa
                st.session_state["msg_sucesso"] = f"Rota greedy!"
                st.session_state["cor"] = "green"
                st.session_state["link_maps"] = link_maps
                st.session_state["dist_greedy"] = dist_greedy
        except Exception as e:
            st.error(f"Erro ao otimizar/plotar rota: {e}")
# Comparação automática sempre que ambos forem executados
dist_ag = st.session_state.get("dist_ag")
dist_greedy = st.session_state.get("dist_greedy")

if dist_ag is not None and dist_greedy is not None and dist_greedy > 0:
    ganho_percentual = 100 * (dist_greedy - dist_ag) / dist_greedy
    st.markdown(f"""
        <div style='background:#263040;padding:20px;border-radius:12px;margin-bottom:12px'>
            <span style='color:#90caf9;font-weight:600;font-size:1.1rem'>Comparativo de Eficiência</span><br>
            <span style='color:#fff;font-size:1.2rem'>
                Distância AG: <b>{dist_ag:.2f} km</b><br>
                Distância Greedy: <b>{dist_greedy:.2f} km</b><br>
                <span style='color:#29b6f6;font-weight:bold;'>O Algoritmo Genético otimizou a rota em {ganho_percentual:.2f}% comparado ao Greedy.</span>
            </span>
        </div>
    """, unsafe_allow_html=True)

def plotar_mapa(rota_completa, cor, label_sucesso, link_maps):
    if len(rota_completa) >= 2:
        m = folium.Map(location=np.mean(np.array(rota_completa), axis=0), zoom_start=12)
        folium.PolyLine(rota_completa, color=cor, weight=2.5, opacity=1).add_to(m)
        if len(rota_completa) >= 1:
            folium.Marker(rota_completa[0], popup="Base Inicial", icon=folium.Icon(color="red")).add_to(m)
        for idx, ponto in enumerate(rota_completa[1:-1], 1):
            folium.Marker(ponto, popup=f"Entrega {idx}").add_to(m)
        if len(rota_completa) > 2:
            folium.Marker(rota_completa[-1], popup="Base Final", icon=folium.Icon(color="green")).add_to(m)
        st.success(label_sucesso)
        st_folium(m, width=900, height=600)
        st.markdown(f"[🗺️ Abrir rota no Google Maps]({link_maps})")
    else:
        st.warning("Rota insuficiente para exibir o mapa (mínimo 2 pontos).")


# Exibe o mapa se existir rota salva
if st.session_state["rota_completa"] is not None:
    plotar_mapa(
        st.session_state["rota_completa"],
        st.session_state["cor"],
        st.session_state["msg_sucesso"],
        st.session_state["link_maps"]
    )

st.markdown("---")
st.caption("Desenvolvido para o Tech Challenge - Algoritmo Genético aplicado à Logística com ERP Protheus")
