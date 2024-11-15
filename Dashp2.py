import dash
import os
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from plotly.subplots import make_subplots

# Inicializar la aplicación con el tema Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define el servidor Flask para gunicorn
server = app.server

# Colores para cada sujeto
subject_colors = {
    "s1": "#6a0dad",
    "s2": "#1e90ff",
    "s3": "#9370db",
    "s4": "#87cefa",
    "s5": "#ba55d3"
}

# Ruta base de los datos
base_path = "./fisioterapia_dataset_regresion"

# Layout del Dashboard
app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Análisis Exploratorio de Datos - Fisioterapia", className="ms-2", style={"color": "white"})
        ]),
        color="#6a0dad", dark=True, className="mb-4"
    ),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="exercise-selector", options=[{"label": f"e{i}", "value": f"e{i}"} for i in range(1, 9)], value="e1", clearable=False), width=4),
        dbc.Col(dcc.Dropdown(id="unit-selector", options=[{"label": f"u{i}", "value": f"u{i}"} for i in range(1, 6)], value="u1", clearable=False), width=4),
        dbc.Col(dcc.Dropdown(id="variable-selector", options=[], value=None, clearable=False), width=4)
    ], justify="center"),
    html.Div([
        dbc.Card([dcc.Graph(id="decomposition-plot"), dbc.CardBody(html.Div(id="decomposition-analysis"))], className="mb-4"),
        dbc.Card([dcc.Graph(id="acf-plot"), dbc.CardBody(html.Div(id="acf-analysis"))], className="mb-4"),
        dbc.Card([dcc.Graph(id="pacf-plot"), dbc.CardBody(html.Div(id="pacf-analysis"))], className="mb-4")
    ])
], fluid=True)

# Callback para opciones dinámicas del dropdown de variables
@app.callback(
    [Output("variable-selector", "options"), Output("variable-selector", "value")],
    [Input("exercise-selector", "value"), Input("unit-selector", "value")]
)
def update_variable_options(selected_exercise, selected_unit):
    try:
        file_path = f"{base_path}/s1/{selected_exercise}/{selected_unit}/template_session.txt"
        data = pd.read_csv(file_path, delimiter=";", header=0)
        variables = [col for col in data.columns if col not in ["time index", "subject"]]
        options = [{"label": var, "value": var} for var in variables]
        return options, variables[0]
    except Exception as e:
        print(f"Error: {e}")
        return [], None

# Callback para actualizar las gráficas y análisis
@app.callback(
    [Output("decomposition-plot", "figure"), Output("decomposition-analysis", "children"),
     Output("acf-plot", "figure"), Output("acf-analysis", "children"),
     Output("pacf-plot", "figure"), Output("pacf-analysis", "children")],
    [Input("exercise-selector", "value"), Input("unit-selector", "value"), Input("variable-selector", "value")]
)
def update_graphs_and_analysis(selected_exercise, selected_unit, selected_variable):
    data_list = []
    for subject in ["s1", "s2", "s3", "s4", "s5"]:
        file_path = f"{base_path}/{subject}/{selected_exercise}/{selected_unit}/template_session.txt"
        try:
            data = pd.read_csv(file_path, delimiter=";", header=0)
            data["subject"] = subject
            data_list.append(data)
        except Exception as e:
            print(f"Error al cargar datos para {subject}: {e}")

    combined_data = pd.concat(data_list, ignore_index=True)

    # Descomposición de la Serie de Tiempo
    decomposition_fig = make_subplots(rows=5, cols=3, subplot_titles=["Tendencia", "Estacionalidad", "Residuos"] * 5)
    decomposition_analysis = ""
    row = 1
    for subject in ["s1", "s2", "s3", "s4", "s5"]:
        subject_data = combined_data[combined_data["subject"] == subject]
        result = seasonal_decompose(subject_data[selected_variable], model='additive', period=25)
        color = subject_colors[subject]

        decomposition_fig.add_trace(go.Scatter(x=subject_data["time index"], y=result.trend, mode="lines", line=dict(color=color), name=f"Tendencia {subject}"), row=row, col=1)
        decomposition_fig.add_trace(go.Scatter(x=subject_data["time index"], y=result.seasonal, mode="lines", line=dict(color=color), name=f"Estacionalidad {subject}"), row=row, col=2)
        decomposition_fig.add_trace(go.Scatter(x=subject_data["time index"], y=result.resid, mode="lines", line=dict(color=color), name=f"Residuos {subject}"), row=row, col=3)

        decomposition_analysis += f"Para {subject}: La tendencia muestra cambios a largo plazo, la estacionalidad destaca patrones cíclicos y los residuos representan la variabilidad aleatoria. "
        row += 1

    decomposition_fig.update_layout(title="Descomposición por Sujeto", height=1500)

    # ACF y PACF para cada sujeto
    acf_fig = go.Figure()
    pacf_fig = go.Figure()
    acf_analysis = ""
    pacf_analysis = ""
    for subject in ["s1", "s2", "s3", "s4", "s5"]:
        subject_data = combined_data[combined_data["subject"] == subject][selected_variable]
        acf_values = acf(subject_data, nlags=40)
        pacf_values = pacf(subject_data, nlags=40)

        acf_fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name=f"ACF {subject}", marker_color=subject_colors[subject]))
        pacf_fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name=f"PACF {subject}", marker_color=subject_colors[subject]))

        acf_analysis += f"Para {subject}: La gráfica ACF muestra la autocorrelación de la serie, indicando posibles patrones de periodicidad. "
        pacf_analysis += f"Para {subject}: La gráfica PACF muestra la correlación parcial, útil para identificar el orden del modelo ARIMA."

    acf_fig.update_layout(title="Autocorrelación (ACF) por Sujeto", height=400)
    pacf_fig.update_layout(title="Correlación Parcial (PACF) por Sujeto", height=400)

    return decomposition_fig, decomposition_analysis, acf_fig, acf_analysis, pacf_fig, pacf_analysis

# Ejecutar la aplicación
if __name__ == "__main__":
    # Configura el servidor para usar el puerto que Render asigna
    port = int(os.environ.get("PORT", 8050))
    
    # Asegúrate de que `server` sea la instancia del servidor Flask en tu aplicación Dash
    app.run_server(host="0.0.0.0", port=port)

