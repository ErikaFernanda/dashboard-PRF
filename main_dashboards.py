import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests

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
                print(f"Erro ao converter latitude/longitude: {e}")

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


url_geojson = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
geojson = requests.get(url_geojson).json()

df_oficial = load_csvs(folder_path, YEARS)
fig1 = generate_heat_map(df_oficial)
fig2 = generate_brasil_map(df_oficial, geojson)

html1 = pio.to_html(fig1, include_plotlyjs="cdn", full_html=False)
html2 = pio.to_html(fig2, include_plotlyjs=False, full_html=False)

form_section = """
<div class="predicao-container">
  <h2>🎯 Predição de Gravidade do Acidente</h2>
  <form id="predict-form" class="predicao-form">
    <div class="form-group">
      <label for="hora">Hora do dia (0–23):</label>
      <input type="number" id="hora" name="hora" min="0" max="23" required>
    </div>

    <div class="form-group">
      <label for="dia_semana">Dia da semana:</label>
      <select id="dia_semana" name="dia_semana" required>
        <option value="segunda-feira">Segunda-feira</option>
        <option value="terça-feira">Terça-feira</option>
        <option value="quarta-feira">Quarta-feira</option>
        <option value="quinta-feira">Quinta-feira</option>
        <option value="sexta-feira">Sexta-feira</option>
        <option value="sábado">Sábado</option>
        <option value="domingo">Domingo</option>
      </select>
    </div>

    <div class="form-group">
      <label for="clima">Condição Climática:</label>
      <select id="clima" name="clima" required>
        <option value="Céu claro">Céu claro</option>
        <option value="Chuva">Chuva</option>
        <option value="Nublado">Nublado</option>
        <option value="Nevoeiro">Nevoeiro</option>
        <option value="Ignorado">Ignorado</option>
      </select>
    </div>

    <button type="submit" class="submit-btn">Prever Gravidade</button>
  </form>

  <div id="resultado" class="resultado"></div>
</div>

<script>
document.getElementById("predict-form").addEventListener("submit", function(e) {
    e.preventDefault();

    const hora = document.getElementById("hora").value;
    const dia_semana = document.getElementById("dia_semana").value;
    const clima = document.getElementById("clima").value;

    fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hora, dia_semana, clima })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("resultado").innerText =
            `🔍 Gravidade prevista: ${data.predicao}`;
    })
    .catch(err => {
        document.getElementById("resultado").innerText =
            "⚠️ Erro na predição. Verifique o backend.";
        console.error(err);
    });
});
</script>
"""

html_final = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Dashboard de Acidentes</title>
   <style>
  body {{
    font-family: "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f4f6f8;
  }}

  .grafico, .predicao-container {{
    padding: 30px;
    margin: 20px;
    background-color: #fff;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  }}

  .predicao-container h2 {{
    margin-bottom: 20px;
    font-size: 1.5em;
    color: #333;
  }}

  .predicao-form {{
    display: flex;
    flex-direction: column;
    gap: 15px;
  }}

  .form-group {{
    display: flex;
    flex-direction: column;
  }}

  label {{
    font-weight: bold;
    margin-bottom: 5px;
    color: #444;
  }}

  input[type="number"],
  select {{
    padding: 8px 12px;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 1em;
  }}

  .submit-btn {{
    padding: 12px;
    background-color: #007BFF;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.3s;
  }}

  .submit-btn:hover {{
    background-color: #0056b3;
  }}

  .resultado {{
    margin-top: 20px;
    font-weight: bold;
    font-size: 1.2em;
    color: #006400;
  }}
</style>

</head>
<body>
    <div class="grafico">{html1}</div>
    <div class="grafico">{html2}</div>
    {form_section}
</body>
</html>
"""

with open("acidentes_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_final)

print("✅ Dashboard gerado com sucesso: acidentes_dashboard.html")
