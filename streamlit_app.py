import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Scoring Marketing", page_icon="🎯", layout="centered")

# ── Chargement modele ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_scoring.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("colonnes.json", "r") as f:
        colonnes = json.load(f)
    return model, scaler, colonnes

model, scaler, colonnes = load_model()

# ── Interface ─────────────────────────────────────────────────
st.title("🎯 Scoring Marketing — Probabilité d'achat")
st.markdown("**Projet MLOps — Bilguissou Diallo | Big Data 2026**")
st.markdown("Remplis les informations du client pour obtenir une prédiction.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    age      = st.slider("Âge", 18, 95, 35)
    balance  = st.number_input("Balance (€)", -5000, 50000, 1500)
    duration = st.number_input("Durée dernier contact (sec)", 0, 3000, 200)
    day      = st.slider("Jour du mois", 1, 31, 15)
with col2:
    campaign = st.number_input("Nombre de contacts campagne", 1, 50, 2)
    pdays    = st.number_input("Jours depuis dernier contact (-1=jamais)", -1, 999, -1)
    previous = st.number_input("Contacts précédents", 0, 50, 0)

if st.button("🔍 Prédire la probabilité d'achat", use_container_width=True):
    # Construire le dataframe
    df = pd.DataFrame([np.zeros(len(colonnes))], columns=colonnes)
    for col in ["age","balance","duration","campaign","pdays","previous","day"]:
        if col in df.columns:
            df[col] = [age,balance,duration,campaign,pdays,previous,day][["age","balance","duration","campaign","pdays","previous","day"].index(col)]
    if "balance_par_age" in df.columns:
        df["balance_par_age"] = balance / (age + 1)
    if "contact_intensif" in df.columns:
        df["contact_intensif"] = int(campaign > 3)
    if "deja_contacte" in df.columns:
        df["deja_contacte"] = int(previous > 0)

    df_sc = scaler.transform(df)
    proba  = model.predict_proba(df_sc)[0][1]
    decision = proba >= 0.3

    st.divider()
    st.subheader("📊 Résultat de la prédiction")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        title={"text": "Probabilité d'achat (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#1D9E75" if decision else "#D85A30"},
            "steps": [
                {"range": [0, 30],  "color": "#FDECEA"},
                {"range": [30, 60], "color": "#FFF8E1"},
                {"range": [60, 100],"color": "#E8F5E9"}
            ],
            "threshold": {"line": {"color": "#333", "width": 3}, "thickness": 0.75, "value": 30}
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    if decision:
        st.success(f"✅ OUI — Client susceptible d'acheter")
        st.info(f"💡 Ce client a **{proba*100:.1f}%** de chances de souscrire.")
    else:
        st.error(f"❌ NON — Client peu susceptible d'acheter")
        st.info(f"💡 Ce client a seulement **{proba*100:.1f}%** de chances de souscrire.")

st.divider()
st.caption("Projet MLOps — Scoring Marketing | Modèle : Logistic Regression | Score métier : 0.940")
