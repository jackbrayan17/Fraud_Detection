import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Page config : light mode et wide layout
st.set_page_config(page_title="Détection de Fraude", layout="wide")

# Load du modèle (vérifie que le fichier est bien dans ton repo Streamlit Cloud)
model = joblib.load("xgb_model.pkl")

# Custom CSS pour un look clean
st.markdown("""
    <style>
        .card-hover {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
            transition: 0.3s ease-in-out;
        }
        .card-hover:hover {
            box-shadow: 0px 6px 20px rgba(13,110,253,0.3);
            transform: translateY(-5px);
        }
    </style>
""", unsafe_allow_html=True)

# --------- FRAUDE DETECTION DASHBOARD ---------
st.title("💳 Détection de Fraude")
st.subheader("Plateforme de prédiction de fraude bancaire")

st.markdown("Entrez les informations ci-dessous pour évaluer le risque de fraude d'une transaction bancaire.")

with st.form("fraude_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Âge", 18, 100, 35)
        salaire = st.number_input("Salaire (€)", 0.0, value=3000.0)
        score_credit = st.slider("Score Crédit", 300, 900, 650)
        montant_transaction = st.number_input("Montant de la transaction (€)", 0.0, value=150.0)
    with col2:
        anciennete = st.slider("Ancienneté du compte (années)", 0, 100, 5)
        type_carte = st.selectbox("Type de carte", ["Master_card", "Visa_card"])
        region = st.selectbox("Région", ["Houston", "Orlando", "Miami"])
        genre = st.selectbox("Genre", ["Homme", "Femme"])

    submit = st.form_submit_button("Prédire la Fraude 🚀")

# Mapping pour encodage
type_map = {"Master_card": 0, "Visa_card": 1}
genre_map = {"Homme": 0, "Femme": 1}
region_map = {"Houston": 0.395, "Orlando": 0.316, "Miami": 0.289}

# Encodage des inputs
type_encoded = type_map[type_carte]
genre_encoded = genre_map[genre]
region_encoded = region_map[region]

# Préparation DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "salaire": [salaire],
    "score_credit": [score_credit],
    "montant_transaction": [montant_transaction],
    "anciennete_compte": [anciennete],
    "type_carte": [type_encoded],
    "region": [region_encoded],
    "genre": [genre_encoded]
})

# Prédiction
if submit:
    D_input = xgb.DMatrix(input_data)
    prediction = model.predict(D_input)[0]
    classe = "✅ Transaction Légitime" if prediction == 0 else "🚨 Fraude Suspectée"

    # Affichage résultat
    st.markdown(f"""
        <div style="margin-top:30px; padding:20px; background-color:#e9ecef;
        border-left:5px solid {'#198754' if prediction == 0 else '#dc3545'}; border-radius:8px;">
            <h3>Résultat :</h3>
            <h2 style="color:{'green' if prediction == 0 else 'red'};">{classe}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Affichage des données
    st.markdown("### 📊 Données saisies")
    cols = st.columns(2)
    for idx, col_name in enumerate(input_data.columns):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
                <div class="card-hover">
                    <h5>{col_name.replace("_", " ").capitalize()}</h5>
                    <p style="font-size: 22px; font-weight: bold; color: #dc3545;">
                        {input_data[col_name].values[0]}
                    </p>
                </div>
            """, unsafe_allow_html=True)
