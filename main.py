import os
import pandas as pd
import threading
import dash
from dash import dcc, html, Input, Output
import plotly.express as px


folder_path = "/dataset/"
YEARS = [2016, 2017, 2018, 2019]

MESES_PT = [
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
]
DIAS_PT = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]


def load_csvs(folder: str, years: list[int]) -> pd.DataFrame:
    frames = []
    for yr in years:
        path = os.path.join(folder, f"datatran{yr}.csv")
        if not os.path.exists(path):
            print(f"Arquivo não encontrado: {path}")
            continue
        df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)
        df["ano"] = yr

        df["data"] = pd.to_datetime(df["data_inversa"], dayfirst=True, errors="coerce")
        df["mes"] = df["data"].dt.month
        df["dia_semana"] = df["data"].dt.weekday

        df["horario_dt"] = pd.to_datetime(
            df["horario"], format="%H:%M:%S", errors="coerce"
        )
        df["hour"] = df["horario_dt"].dt.hour
        df["minute"] = df["horario_dt"].dt.minute
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "Nenhum CSV carregado. Verifique `folder_path` e `YEARS`."
        )
    return pd.concat(frames, ignore_index=True)


DF_RAW = load_csvs(folder_path, YEARS)


def prep_pivot(df: pd.DataFrame) -> pd.DataFrame:
    df["mes_nome"] = df["mes"].map(dict(zip(range(1, 13), MESES_PT)))
    df["dia_nome"] = df["dia_semana"].map(dict(zip(range(7), DIAS_PT)))
    tabela = (
        df.groupby(["mes_nome", "dia_nome"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=MESES_PT, columns=DIAS_PT)
    )
    return tabela


app = dash.Dash(__name__, title="Acidentes de Trânsito BR – 2016‑2019")
server = app.server

app.layout = html.Div(
    [
        html.H2("Dashboard de Acidentes de Trânsito", style={"textAlign": "center"}),
        html.H3("Erika Fernanda - Ciência de dados", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="year-dropdown",
            multi=True,
            placeholder="Selecione ano(s)",
            options=[{"label": "Todos", "value": "all"}]
            + [{"label": str(yr), "value": str(yr)} for yr in YEARS],
            value=["all"],
            style={"width": "50%", "margin": "0 auto"},
            clearable=False,
        ),
        dcc.Graph(id="heatmap"),
        dcc.Graph(id="bar-chart"),
        dcc.Graph(id="minute-line"),
        html.Div(
            [
                html.Label("Intervalo de Horas:"),
                dcc.RangeSlider(
                    id="hour-range-slider",
                    min=0,
                    max=23,
                    step=1,
                    value=[0, 23],
                    allowCross=False,
                    marks={h: f"{h:02d}h" for h in range(0, 24, 2)},
                ),
            ],
            style={"width": "60%", "margin": "1rem auto"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "padding": "1rem 2rem"},
)


@app.callback(
    Output("heatmap", "figure"),
    Output("bar-chart", "figure"),
    Output("minute-line", "figure"),
    Input("year-dropdown", "value"),
    Input("hour-range-slider", "value"),
)
def update_figs(selected_years, hour_range):

    if not selected_years or "all" in selected_years:
        df = DF_RAW.copy()
        title_suffix = "2016‑2019"
    else:
        yrs = [int(y) for y in selected_years]
        df = DF_RAW[DF_RAW["ano"].isin(yrs)]
        title_suffix = ", ".join(selected_years)

    pivot = prep_pivot(df)
    heat_fig = px.imshow(
        pivot,
        labels=dict(x="Dia da Semana", y="Mês", color="Nº de acidentes"),
        x=DIAS_PT,
        y=MESES_PT,
        aspect="auto",
        title=f"Acidentes por Dia da Semana e Mês ({title_suffix})",
        color_continuous_scale="Oranges",
    )
    heat_fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))

    bars = df.groupby("dia_semana", observed=True).size().reindex(range(7)).fillna(0)
    bar_fig = px.bar(
        x=[DIAS_PT[d] for d in bars.index],
        y=bars.values,
        labels={"x": "Dia da Semana", "y": "Nº de acidentes"},
        title=f"Total de Acidentes por Dia da Semana ({title_suffix})",
        text_auto=True,
    )

    df_natal = df[df["uf"].str.contains("RN", na=False, case=False)]
    df_natal["hora"] = df_natal["horario_dt"].dt.hour
    df_natal["hora_minuto"] = pd.to_datetime(
        df_natal["horario_dt"].dt.strftime("%H:%M"), format="%H:%M"
    )

    start_hour, end_hour = hour_range
    df_natal = df_natal[
        (df_natal["hora"] >= start_hour) & (df_natal["hora"] <= end_hour)
    ]

    minute_counts = (
        df_natal.groupby(["hora_minuto", "ano"])
        .size()
        .reset_index(name="count")
        .sort_values(["ano", "hora_minuto"])
    )

    line_fig = px.line(
        minute_counts,
        x="hora_minuto",
        y="count",
        color="ano",
        labels={
            "hora_minuto": "Hora (HH:MM)",
            "count": "Nº de acidentes",
            "ano": "Ano",
        },
        title=f"Acidentes por Hora e Minuto no RN ({title_suffix})",
    )
    line_fig.update_layout(xaxis=dict(tickformat="%H:%M", tickangle=45))

    return heat_fig, bar_fig, line_fig


def run_dash():
    app.run(debug=False, use_reloader=False)


thread = threading.Thread(target=run_dash)
thread.start()