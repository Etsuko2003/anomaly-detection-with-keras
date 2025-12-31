import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model
from autoencoder_project.main import preprocess, timeseries_dataset 

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="", layout="wide")
st.title("Analyse détaillée des anomalies (Vue 4h)")
st.markdown(
    "Nous avons fixé la fenêtre sur **4 heures** pour maintenir une meilleure visibilité sur les variations."
)


# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    scaler = joblib.load("models/scaler.pkl")
    model = load_model("models/best_autoencoder.keras")
    return scaler, model


@st.cache_data
def get_all_data_and_preds(_model, _scaler):
    df_raw = pd.read_csv(
        "data/household_power_consumption_sample_2010.csv", parse_dates=["datetime"]
    )
    df_processed = preprocess(df_raw)
    data_scaled = _scaler.transform(df_processed)

    # Paramètres temporels
    timesteps = 240
    ds = timeseries_dataset(data_scaled, timesteps=timesteps, batch_size=512)

    # Inférence
    preds = _model.predict(ds, verbose=0)

    return df_processed, data_scaled, preds


scaler, model = load_resources()
df_proc, scaled_vals, all_preds = get_all_data_and_preds(model, scaler)

# --- ALIGNEMENT TEMPOREL ---
timesteps = 240
y_true_aligned = scaled_vals[timesteps - 1 : timesteps - 1 + len(all_preds)]
dates_aligned = df_proc.index[timesteps - 1 : timesteps - 1 + len(all_preds)]

# --- PARAMÈTRES (SIDEBAR) ---
st.sidebar.header("Réglages")
threshold = st.sidebar.slider(
    "Seuil d'anomalie (MSE)", 0.001, 0.05, 0.01, format="%.3f"
)

# --- CRÉATION DU GRAPHIQUE MULTI-COLONNES ---
features = df_proc.columns
num_features = len(features)

# Création des sous-graphiques : une ligne par feature
fig = make_subplots(
    rows=num_features,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,  # Espace minimal entre les graphiques
    subplot_titles=[f"Variable : {col}" for col in features],
)

for i, col_name in enumerate(features):
    # Reconstruction
    y_pred = all_preds[:, -1, i]
    y_actual = y_true_aligned[:, i]

    # Erreurs et Masque
    errors = (y_actual - y_pred) ** 2
    anomalies = errors > threshold

    # Trace de la donnée réelle
    fig.add_trace(
        go.Scatter(
            x=dates_aligned,
            y=y_actual,
            name=f"Réel {col_name}",
            mode="lines",
            line=dict(width=1.8),
            hovertemplate="Valeur: %{y:.3f}<extra></extra>",
        ),
        row=i + 1,
        col=1,
    )

    # Trace des anomalies (points rouges)
    if np.any(anomalies):
        fig.add_trace(
            go.Scatter(
                x=dates_aligned[anomalies],
                y=y_actual[anomalies],
                mode="markers",
                marker=dict(color="#EF553B", size=5, symbol="circle"),
                name="Anomalie",
                legendgroup="anomalie",
                showlegend=(i == 0),
            ),
            row=i + 1,
            col=1,
        )

# --- 5. CONFIGURATION DU DÉROULEMENT ET DU RANGE SLIDER ---
# On fixe la vue sur les 4 premières heures (240 min)
start_view = dates_aligned[0]
end_view = start_view + pd.Timedelta(hours=4)

fig.update_layout(
    height=280 * num_features,  # Hauteur confortable pour chaque ligne
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=60, r=40, t=100, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# Application de la fenêtre de 4h et du Range Slider unique tout en bas
fig.update_xaxes(
    range=[start_view, end_view],  # Zoom initial 4h
    rangeslider=dict(visible=True, thickness=0.05),  # Barre de défilement
    row=num_features,
    col=1,
)

# On cache les sliders intermédiaires pour ne garder que celui du bas (contrôle global)
for r in range(1, num_features):
    fig.update_xaxes(rangeslider=dict(visible=False), row=r, col=1)

# Affichage final
st.plotly_chart(fig, use_container_width=True)

# --- 6. RÉSUMÉ PAR COLONNE ---
st.subheader("Nombre d'anomalies détectées")
metrics_cols = st.columns(num_features)
for i, col_name in enumerate(features):
    y_pred = all_preds[:, -1, i]
    y_actual = y_true_aligned[:, i]
    nb_anom = np.sum(((y_actual - y_pred) ** 2) > threshold)
    metrics_cols[i].metric(label=col_name, value=nb_anom)
