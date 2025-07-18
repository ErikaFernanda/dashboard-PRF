import streamlit as st
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st


st.set_page_config(layout="wide")
folder_path = "./dataset/"
YEARS = [2017, 2018, 2019]
mapa_dias = {
    0: "segunda-feira",
    1: "terça-feira",
    2: "quarta-feira",
    3: "quinta-feira",
    4: "sexta-feira",
    5: "sábado",
    6: "domingo",
}


def generate_brasil_map(df_oficial, geojson):
    df = df_oficial.groupby(["uf", "ano"]).size().reset_index(name="qtd_acidentes")

    fig = px.choropleth(
        df,
        locations="uf",
        featureidkey="properties.sigla",
        geojson=geojson,
        animation_frame="ano",
        color="qtd_acidentes",
        scope="south america",
        range_color=(0, df["qtd_acidentes"].max()),
        color_continuous_scale="Reds",
        title="Quantidade de Acidentes por Estado",
        width=1000,
        height=700,
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def generate_heat_map(df_oficial):
    df = df_oficial.copy()
    df["horario_dt"] = pd.to_datetime(df["horario"], format="%H:%M:%S", errors="coerce")
    df["hora"] = df["horario_dt"].dt.hour

    heatmap_data = (
        df.groupby(["hora", "dia_semana", "ano"])
        .size()
        .reset_index(name="qtd_acidentes")
    )

    fig = px.density_heatmap(
        heatmap_data,
        x="dia_semana",
        y="hora",
        z="qtd_acidentes",
        category_orders={
            "dia_semana": list(mapa_dias.values()),
            "hora": list(range(24)),
        },
        animation_frame="ano",
        color_continuous_scale="Reds",
        title="Densidade de Acidentes por Hora e Dia da Semana",
        labels={
            "qtd_acidentes": "N° de Acidentes",
            "hora": "Hora do Dia",
            "dia_semana": "Dia da Semana",
        },
        width=1000,
        height=600,
    )

    fig.update_layout(yaxis=dict(dtick=1, autorange=True))
    return fig


def generate_bar_acidentes_por_gravidade(df_oficial, ano_escolhido):
    df_filtrado = df_oficial[df_oficial["ano"] == ano_escolhido]

    
    dados = (
        df_filtrado.groupby(["mes", "classificacao_acidente"])
        .size()
        .reset_index(name="qtd")
        .sort_values("mes")
    )

    fig = px.bar(
        dados,
        x="mes",
        y="qtd",
        color="classificacao_acidente",
        barmode="group",
        title=f"Acidentes por Gravidade mês a mês - {ano_escolhido}",
        labels={
            "mes": "Mês",
            "qtd": "Número de Acidentes",
            "classificacao_acidente": "Gravidade",
        },
        category_orders={"mes": list(range(1, 13))},
        color_discrete_sequence=[
            "#1f77b4",
            "#ff7f0e",
            "#d62728",
        ], 
        height=500,
        width=900,
    )
    fig.update_layout(xaxis=dict(tickmode="linear", tick0=1, dtick=1))
    return fig


def plot_grafico_acidentes_graves_por_clima(df):

    st.subheader("Acidentes Graves por Condição do Céu")

    df_graves = df[(df["mortos"] > 0) | (df["feridos_graves"] > 0)]

    contagem = (
        df_graves["condicao_metereologica"].value_counts().sort_values(ascending=False)
    )

    if contagem.empty:
        st.warning("Nenhum acidente grave encontrado nos dados.")
        return

    fig, ax = plt.subplots()
    contagem.plot(kind="bar", color="darkred", edgecolor="black", ax=ax)
    ax.set_title("Acidentes Graves por Condição Climática")
    ax.set_xlabel("Clima")
    ax.set_ylabel("Quantidade")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


def load_csvs(folder: str, years):
    frames = []
    for yr in years:
        path = os.path.join(folder, f"datatran{yr}.csv")
        df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)
        df["ano"] = yr

        df["data"] = pd.to_datetime(
            df["data_inversa"], format="%Y-%m-%d", dayfirst=True, errors="coerce"
        )
        df["mes"] = df["data"].dt.month
        df["dia_semana"] = df["data"].dt.weekday.map(mapa_dias)

        df["horario_dt"] = pd.to_datetime(
            df["horario"], format="%H:%M:%S", errors="coerce"
        )
        df["hora"] = df["horario_dt"].dt.hour

        if "latitude" in df.columns and "longitude" in df.columns:
            try:
                df["latitude"] = df["latitude"].str.replace(",", ".").astype(float)
                df["longitude"] = df["longitude"].str.replace(",", ".").astype(float)
            except Exception as e:
                st.warning(f"Erro ao converter latitude/longitude: {e}")

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


st.title("🚧 Dashboards de Acidentes de Trânsito")
with st.spinner("Carregando dados..."):
    url_geojson = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    geojson = requests.get(url_geojson).json()
    df_oficial = load_csvs(folder_path, YEARS)

st.subheader("📊 Mapa de Densidade de Acidentes por Hora e Dia da Semana")
fig1 = generate_heat_map(df_oficial)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🗺️ Mapa de Acidentes por Estado")
fig2 = generate_brasil_map(df_oficial, geojson)
st.plotly_chart(fig2, use_container_width=True)


anos_disponiveis = sorted(df_oficial["ano"].unique())
ano_selecionado = st.selectbox("Selecione o ano:", anos_disponiveis)

fig = generate_bar_acidentes_por_gravidade(df_oficial, ano_selecionado)
st.plotly_chart(fig)

plot_grafico_acidentes_graves_por_clima(df_oficial)

with open("modelo_gravidade_simplificado2.pkl", "rb") as f:
    modelo = pickle.load(f)

st.title("Predição de Gravidade de Acidente com IA")

hora = st.number_input("Hora do dia (0–23):", min_value=0, max_value=23, step=1)
dia_semana = st.selectbox(
    "Dia da semana:",
    [
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo",
    ],
)
clima = st.selectbox(
    "Condição Climática:",
    ["Céu claro", "Sol", "Chuva", "Nublado", "Nevoeiro", "Ignorado"],
)


if st.button("Prever Gravidade"):
    entrada = {
        "hour": [hora],
        "dia_semana": [dia_semana],
        "condicao_metereologica": [clima],
    }

    df_input = pd.DataFrame(entrada)
    pred = modelo.predict(df_input)[0]
    classe = "Grave" if pred == 1 else "Não Grave"

    st.markdown(f"### 🚨 Gravidade prevista: **{classe}**")
st.image(
    "./tree.png", caption="Primeira arvore da Random Forest", use_container_width=True
)
st.image("./info.png", caption="Informações do modelo", use_container_width=True)
