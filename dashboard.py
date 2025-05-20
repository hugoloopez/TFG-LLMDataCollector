import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
import re
import json
from plotly.utils import PlotlyJSONEncoder
from plotly.io import from_json
from dash.exceptions import PreventUpdate

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOSTNAME"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USERNAME2"),
    password=os.getenv("DB_PASSWORD2"),
    dbname=os.getenv("DB_NAME")
)

cursor = conn.cursor()


def calcular_coverage_por_dni(dni):
    dni = dni.strip().upper()
    try:
        campos_totales = 17
        puntos_por_campo = 100 / campos_totales
        coverage = 0

        cursor.execute("""
            SELECT cliente_id, nombre, apellidos, dni, correo, telefono
            FROM clientes WHERE UPPER(TRIM(dni)) = %s;
        """, (dni,))
        cliente = cursor.fetchone()
        if not cliente:
            return 0
        cliente_id = cliente[0]
        campos_cliente = cliente[1:]
        coverage += sum([puntos_por_campo for val in campos_cliente if val not in (None, "", " ")])

        cursor.execute("""
            SELECT calle, numero, ciudad, pais, codigo_postal
            FROM direcciones WHERE cliente_id = %s LIMIT 1;
        """, (cliente_id,))
        direccion = cursor.fetchone()
        if direccion:
            coverage += sum([puntos_por_campo for val in direccion if val not in (None, "", " ")])

        cursor.execute("""
            SELECT numero_tarjeta, titular_tarjeta, fecha_expiracion, cvv
            FROM metodos_pago WHERE cliente_id = %s LIMIT 1;
        """, (cliente_id,))
        pago = cursor.fetchone()
        if pago:
            coverage += sum([puntos_por_campo for val in pago if val not in (None, "", " ")])

        cursor.execute("""
            SELECT plan_id, fecha_inicio, activo
            FROM clientes_planes WHERE cliente_id = %s LIMIT 1;
        """, (cliente_id,))
        plan = cursor.fetchone()
        if plan:
            coverage += sum([puntos_por_campo for val in plan if val not in (None, "", " ")])

        return round(coverage, 2)
    except Exception as e:
        print(f"Error al calcular coverage para {dni}: {e}")
        return 0


BASE_DIR = "log/2025-04-22"
metricas_data = []
logs_data = []

for modelo in os.listdir(BASE_DIR):
    modelo_path = os.path.join(BASE_DIR, modelo)
    if not os.path.isdir(modelo_path):
        continue

    for archivo in os.listdir(modelo_path):
        path_archivo = os.path.join(modelo_path, archivo)

        if archivo.endswith(".csv") and "_" in archivo:
            dni = archivo.split("_")[1].split(".")[0]
            df = pd.read_csv(path_archivo)
            df["modelo"] = modelo
            df["dni"] = dni
            metricas_data.append(df)

        elif archivo.endswith(".log"):
            dni = archivo.split(".")[0]
            with open(path_archivo, encoding='utf-8') as f:
                texto = f.read()

            cost = 0
            if "Coste de la conversación" in texto:
                try:
                    cost = float(texto.split("Coste de la conversación:")[1].split("USD")[0].strip())
                except:
                    pass

            tiempo_total = 0
            try:
                tiempos = re.findall(r"Tiempo total de conversación:\s*([0-9.]+)\s*segundos", texto)
                if tiempos:
                    tiempo_total = float(tiempos[-1])
            except Exception as e:
                print(f"Error al extraer tiempo total del log de {dni}: {e}")

            logs_data.append({
                "modelo": modelo,
                "dni": dni,
                "coste_usd": cost,
                "tiempo_total_s": tiempo_total
            })

df_metricas = pd.concat(metricas_data, ignore_index=True)
df_logs = pd.DataFrame(logs_data)
df_completo = pd.merge(df_metricas, df_logs, on=["modelo", "dni"])

df_completo["dni"] = df_completo["dni"].astype(str).str.strip().str.upper()
df_completo["coverage"] = df_completo["dni"].apply(calcular_coverage_por_dni)
df_completo["registrado_en_bbdd"] = df_completo["coverage"] == 100
usuarios_registrados = df_completo["coverage"] > 0
df_completo["registro_no_exitoso"] = usuarios_registrados & (df_completo["coverage"] < 100)


agrupado = df_completo.groupby("modelo")[["cosine", "bleu", "meteor", "rougeL", "tiempo_total_s", "coste_usd", "coverage"]].mean().reset_index()
total_intentos = df_completo.groupby("modelo")["dni"].nunique()
exitos = df_completo[df_completo["registrado_en_bbdd"] == True].groupby("modelo")["dni"].nunique()
no_exitosos = df_completo[df_completo["registro_no_exitoso"] == True].groupby("modelo")["dni"].nunique()
no_registrados = df_completo[df_completo["coverage"] == 0].groupby("modelo")["dni"].nunique()

exitos = exitos.reindex(total_intentos.index, fill_value=0)
no_exitosos = no_exitosos.reindex(total_intentos.index, fill_value=0)
no_registrados = no_registrados.reindex(total_intentos.index, fill_value=0)

porcentaje_exito = (exitos / total_intentos * 100).round(2)
porcentaje_no_exito = (no_exitosos / total_intentos * 100).round(2)
porcentaje_no_registrado = (no_registrados / total_intentos * 100).round(2)

tabla_resumen = agrupado.copy()
tabla_resumen = tabla_resumen.merge(
    porcentaje_exito.rename("% registros exitosos"), on="modelo", how="left"
)
tabla_resumen = tabla_resumen.merge(
    porcentaje_no_exito.rename("% registros no exitosos"), on="modelo", how="left"
)
tabla_resumen = tabla_resumen.merge(
    porcentaje_no_registrado.rename("% registros no registrados"), on="modelo", how="left"
)
tabla_resumen["% registros exitosos"] = tabla_resumen["% registros exitosos"].fillna(0)
tabla_resumen["% registros no exitosos"] = tabla_resumen["% registros no exitosos"].fillna(0)
tabla_resumen["% registros no registrados"] = tabla_resumen["% registros no registrados"].fillna(0)


def formatear_colores(df):
    estilos = []
    for i, row in df.iterrows():
        for col in df.columns:
            if col == "% registros exitosos":
                estilos.append({"if": {"row_index": i, "column_id": col}, "color": "green"})
            elif col == "% registros no exitosos":
                estilos.append({"if": {"row_index": i, "column_id": col}, "color": "orange"})
            elif col == "% registros no registrados":
                estilos.append({"if": {"row_index": i, "column_id": col}, "color": "red"})
    return estilos


colores_modelos = {
    "Claude-3_Opus": "#FFA15A",
    "Claude-3_Haiku": "#19D3F3",
    "Cohere_Command_R+": "#FF6692",
    "Claude-3.5": "#636EFA",
    "Claude_3.7_Sonnet_(20250219)": "#B6E880",
    "Gemini_2.0_Flash": "#FF97FF",
    "Gemini_2.0_Flash-Lite": "#FECB52",
    "Gemini_2.5_Pro": "#00B894",
    "OpenAI_GPT-3.5": "#EF553B",
    "OpenAI_GPT-4.1": "#AB63FA",
    "OpenAI_GPT-4o": "#00CC96",
    "OpenAI_GPT-4_Turbo": "#7057FF",
    "Deepseek-V3": "#FFB347",
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


metricas_disponibles = ["cosine", "bleu", "meteor", "rougeL", "tiempo_total_s", "coste_usd", "coverage", "% registros exitosos"]
titulos = {
    "cosine": "Similitud Coseno",
    "bleu": "BLEU Score",
    "meteor": "METEOR",
    "rougeL": "ROUGE-L",
    "tiempo_total_s": "Tiempo Total (s)",
    "coste_usd": "Coste (USD)",
    "coverage": "Coverage",
    "% registros exitosos": "% Registros Exitosos"
}

app.layout = dbc.Container([

    html.H2("Comparativa de Modelos", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Métrica:"),
            dcc.Dropdown(
                options=[{"label": titulos[m], "value": m} for m in metricas_disponibles],
                value="cosine",
                id="filtro_metrica"
            )
        ], md=4),
        dbc.Col([
            html.Label("Mostrar solo registrados"),
            dbc.Checkbox(id="filtro_registrados", value=False)
        ], md=4),
        dbc.Col([
            html.Label("Ordenar gráfico de barras"),
            dcc.RadioItems(
                options=[
                    {"label": "Original", "value": "original"},
                    {"label": "Ascendente", "value": "asc"},
                    {"label": "Descendente", "value": "desc"},
                ],
                value="original",
                id="orden_barras",
                inline=True
            )
        ], md=4),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="grafico_barras"),
            dcc.Store(id="store_barras"),
            html.Button("Descargar Gráfico de Barras", id="btn_download_barras", className="btn btn-primary my-2"),
            dcc.Download(id="download_barras")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="grafico_boxplot"),
            dcc.Store(id="store_boxplot"),
            html.Button("Descargar Boxplot", id="btn_download_boxplot", className="btn btn-primary my-2"),
            dcc.Download(id="download_boxplot")
        ], md=6),

        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id="modelo_donut",
                    options=[{"label": modelo, "value": modelo} for modelo in tabla_resumen["modelo"].unique()],
                    placeholder="Todos (general)",
                    value=None,
                    style={"marginBottom": "20px", "display": "none"}
                ),
                dcc.Graph(id="grafico_dispersion"),
                dcc.Store(id="store_dispersion"),
                html.Button("Descargar Dispersión", id="btn_download_dispersion", className="btn btn-primary my-2"),
                dcc.Download(id="download_dispersion")
            ], id="contenedor_grafico_dispersion")
        ], md=6),
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Tabs(id="tabs_radar", value="Claude", children=[]),
            dcc.Store(id="store_radar"),
            html.Button("Descargar Radar", id="btn_download_radar", className="btn btn-primary my-2"),
            dcc.Download(id="download_radar")
        ])
    ]),


    html.Hr(),

    html.H4("Tabla resumen de métricas por modelo"),
    dash_table.DataTable(
        data=tabla_resumen.round(2).to_dict("records"),
        columns=[{"name": i, "id": i} for i in tabla_resumen.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=formatear_colores(tabla_resumen)
    )

], fluid=True)


@app.callback(
    Output("grafico_dispersion", "style"),
    Input("filtro_metrica", "value")
)
def ocultar_dispersion_si_exito(metrica):
    if metrica == "% registros exitosos":
        return {"display": "none"}
    return {"display": "block"}


@app.callback(
    Output("modelo_donut", "style"),
    Input("filtro_metrica", "value")
)
def mostrar_dropdown(metrica):
    if metrica == "% registros exitosos":
        return {"marginBottom": "20px", "display": "block"}
    return {"display": "none"}


@app.callback(
    Output("grafico_barras", "figure"),
    Output("grafico_boxplot", "figure"),
    Output("grafico_dispersion", "figure"),
    Output("tabs_radar", "children"),
    Output("store_barras", "data"),
    Output("store_boxplot", "data"),
    Output("store_dispersion", "data"),
    Output("store_radar", "data"),
    Input("filtro_metrica", "value"),
    Input("filtro_registrados", "value"),
    Input("orden_barras", "value"),
    Input("modelo_donut", "value")
)
def actualizar_graficos(metrica, solo_registrados, orden_barras, modelo_seleccionado):
    df = df_completo.copy()
    resumen = tabla_resumen.copy()

    if solo_registrados:
        df = df[df["registrado_en_bbdd"] == True]

    if orden_barras in ["asc", "desc"] and metrica in resumen.columns:
        resumen = resumen.sort_values(by=metrica, ascending=(orden_barras == "asc"))

    if metrica == "coste_usd":
        resumen = resumen[resumen["coste_usd"] > 0]

    if metrica == "% registros exitosos":
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=resumen["modelo"],
            y=resumen[metrica],
            name="% Registros Exitosos",
            marker_color=[colores_modelos.get(m, "#888") for m in resumen["modelo"]],
            text=resumen[metrica],
            textposition="auto",
            yaxis="y1"
        ))
        fig_bar.add_trace(go.Scatter(
            x=resumen["modelo"],
            y=resumen["tiempo_total_s"],
            name="Tiempo Promedio (s)",
            mode="lines+markers+text",
            text=[f"{v:.1f}s" for v in resumen["tiempo_total_s"]],
            textposition="top center",
            yaxis="y2",
            line=dict(color="black", dash="dot")
        ))
        fig_bar.update_layout(
            title="Promedio de % Registros Exitosos y Tiempo Promedio por Modelo",
            yaxis=dict(title="% Registros Exitosos", side="left"),
            yaxis2=dict(title="Tiempo Promedio (s)", overlaying="y", side="right"),
            xaxis=dict(title="Modelo"),
            legend=dict(x=0.5, xanchor="center", orientation="h", y=1.1),
            margin=dict(t=60, b=60)
        )
    else:
        fig_bar = px.bar(
            resumen,
            x="modelo",
            y=metrica,
            title=f"Promedio de {titulos.get(metrica, metrica)}",
            text_auto=".2f",
            color="modelo",
            color_discrete_map=colores_modelos
        )
        fig_bar.update_traces(texttemplate="%{y:.2f}")
        fig_bar.update_layout(yaxis_tickformat=".2f")

    if metrica == "% registros exitosos":
        if modelo_seleccionado:
            valores = [
                resumen[resumen["modelo"] == modelo_seleccionado]["% registros exitosos"].values[0],
                resumen[resumen["modelo"] == modelo_seleccionado]["% registros no exitosos"].values[0],
                resumen[resumen["modelo"] == modelo_seleccionado]["% registros no registrados"].values[0]
            ]
        else:
            valores = [
                porcentaje_exito.sum(),
                porcentaje_no_exito.sum(),
                porcentaje_no_registrado.sum()
            ]

        fig_box = go.Figure(data=[go.Pie(
            labels=["Registro completo", "Registro parcial", "No Registrado"],
            values=valores,
            hole=0.5,
            textinfo="label+percent",
            marker_colors=["green", "orange", "red"]
        )])
        fig_disp = go.Figure()

    else:
        fig_box = px.box(
            df,
            x="modelo",
            y=metrica,
            color="modelo",
            title=f"Distribución de {titulos.get(metrica, metrica)}",
            color_discrete_map=colores_modelos
        )
        df_disp = df.dropna(subset=[metrica, "tiempo_total_s", "coste_usd"])
        fig_disp = px.scatter(
            df_disp,
            x="tiempo_total_s",
            y="coste_usd",
            color="modelo",
            size=metrica,
            title=f"Tiempo vs Coste por {titulos.get(metrica, metrica)}",
            color_discrete_map=colores_modelos
        )

    labels = ["cosine", "bleu", "meteor", "rougeL"]
    maximos_globales = resumen[labels].max().values.tolist()

    proveedores = {
        "Claude": ["Claude-3.5", "Claude-3_Haiku", "Claude-3_Opus", "Claude_3.7_Sonnet_(20250219)"],
        "OpenAI": ["OpenAI_GPT-3.5", "OpenAI_GPT-4.1", "OpenAI_GPT-4_Turbo", "OpenAI_GPT-4o"],
        "Gemini": ["Gemini_2.0_Flash", "Gemini_2.0_Flash-Lite", "Gemini_2.5_Pro"],
        "Otros": ["Cohere_Command_R+", "Deepseek-V3"]
    }

    radar_tabs = []
    radar_dict = {}

    for nombre_prov, modelos in proveedores.items():
        radar_fig = go.Figure()
        for _, row in resumen[resumen["modelo"].isin(modelos)].iterrows():
            valores = row[labels].tolist() + [row[labels[0]]]
            radar_fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=labels + [labels[0]],
                fill='toself',
                name=row["modelo"],
                line=dict(color=colores_modelos.get(row["modelo"], None))
            ))

        radar_fig.update_layout(
            title=f"Radar de Métricas - {nombre_prov}",
            polar=dict(radialaxis=dict(visible=True, range=[0, max(maximos_globales)])),
            showlegend=True,
            legend=dict(
                orientation="v",
                x=0.6,
                y=1,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="lightgray",
                borderwidth=1,
                font=dict(size=14),
                itemsizing="constant",
                itemwidth=30        
            ),
            margin=dict(t=60, b=40, l=40, r=100)
        )

        radar_tabs.append(dcc.Tab(label=nombre_prov, value=nombre_prov, children=[dcc.Graph(figure=radar_fig)]))
        radar_dict[nombre_prov] = radar_fig


    return (
        fig_bar, fig_box, fig_disp, radar_tabs,
        json.dumps(fig_bar, cls=PlotlyJSONEncoder),
        json.dumps(fig_box, cls=PlotlyJSONEncoder),
        json.dumps(fig_disp, cls=PlotlyJSONEncoder),
        json.dumps(radar_tabs, cls=PlotlyJSONEncoder)
    )


@app.callback(
    Output("download_csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True
)
def descargar_csv(n_clicks):
    return dcc.send_data_frame(tabla_resumen.to_csv, "tabla_resumen_2025-04-16.csv")


def exportar_figura_json(fig_json, filename="grafica.png"):
    if not fig_json:
        raise PreventUpdate
    try:
        fig = from_json(fig_json)
        img_bytes = fig.to_image(format="png", scale=2)
        return dcc.send_bytes(img_bytes, filename=filename)
    except Exception as e:
        print("Error al exportar imagen:", e)
        raise PreventUpdate


@app.callback(
    Output("download_barras", "data"),
    Input("btn_download_barras", "n_clicks"),
    State("store_barras", "data"),
    prevent_initial_call=True
)
def descargar_barras(n_clicks, fig_json):
    print("Contenido de fig_json:", fig_json)
    if not fig_json:
        raise dash.exceptions.PreventUpdate
    return exportar_figura_json(fig_json, "barras.png")


@app.callback(
    Output("download_boxplot", "data"),
    Input("btn_download_boxplot", "n_clicks"),
    State("store_boxplot", "data"),
    prevent_initial_call=True
)
def descargar_boxplot(n_clicks, fig_json):
    return exportar_figura_json(fig_json, "boxplot.png")


@app.callback(
    Output("download_dispersion", "data"),
    Input("btn_download_dispersion", "n_clicks"),
    State("store_dispersion", "data"),
    prevent_initial_call=True
)
def descargar_dispersion(n_clicks, fig_json):
    return exportar_figura_json(fig_json, "dispersion.png")


@app.callback(
    Output("download_radar", "data"),
    Input("btn_download_radar", "n_clicks"),
    State("store_radar", "data"),
    prevent_initial_call=True
)
def descargar_radar(n_clicks, fig_json):
    return exportar_figura_json(fig_json, "radar.png")


app.run(debug=False)
