import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import xgboost as xgb



# Load model
model = joblib.load("xgb_model.pkl")

# Page config
st.set_page_config(page_title="Détection de Fraude", layout="wide")

# Sidebar navigation
st.sidebar.image('image1.jpg', use_container_width=True)
st.sidebar.title("🧭 Navigation")

st.markdown("""
            <style>
                /* Style pour tous les boutons dans la sidebar */
                section[data-testid="stSidebar"] button {
                    background-color: #087ff0;
                    color: white;
                    font-size: 15px;
                    border-radius: 8px;
                    padding: 8px 18px;
                    transition: 0.3s;
                }
                section[data-testid="stSidebar"] button:hover {
                    transform: scale(1.05);
                    color: #ffffff;
                    border: 1px solid #087ff0;
                }
            </style>
            """, unsafe_allow_html=True)


Acceuil = st.sidebar.button("Accueil", key="home",use_container_width=True)
Detection = st.sidebar.button("Détection de Fraude", key="fraud_detection",use_container_width=True)

if 'page' not in st.session_state:
    st.session_state.page = 'Accueil'

if Acceuil:
    st.session_state.page = "Accueil"
if Detection:
    st.session_state.page = "Fraude_Detection"


# --------- PAGE ACCUEIL ---------
if st.session_state.page == "Accueil":
    st.title("💳 Application de Détection de Fraude")
    st.image('image2.jpg', use_container_width=True)
    st.markdown("""
        ### À propos de l'application
        Cette application permet de **détecter automatiquement les transactions frauduleuses** en se basant sur un modèle de machine learning pré-entraîné (**XGBoost**).

        🔍 **Fonctionnalités :**
        - Interface utilisateur intuitive
        - Prédiction en temps réel d'une transaction
        - Affichage clair des données saisies

        🧠 **But :** Aider les institutions financières à détecter plus efficacement les fraudes et à réduire les pertes.
    """)

# --------- PAGE PREDICTION ---------
if st.session_state.page == "Fraude_Detection":
    st.title("🕵️ Détection de Fraude")
    
    st.markdown("""
        <style>
        
        .upload-section {
            background: linear-gradient(135deg, #062a99 0%, #067399 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            color: white;
            text-align: center;
        }
        
        </style>       
                
    """,unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-section">
        <h3>📝Remplissez les informations </h3>
        <p>Obtenez une prediction </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Données utilisateur à saisir ---
    st.subheader("📝 Entrée des données")

    with st.form("fraude_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Âge", min_value=18, max_value=100, value=35)
            salaire = st.number_input("Salaire", min_value=0.0, value=3000.0)
            score_credit = st.slider("Score Crédit", min_value=300, max_value=900, value=650)
            montant_transaction = st.number_input("Montant de la transaction", min_value=0.0, value=150.0)
        with col2:
            anciennete = st.slider("Ancienneté du compte (en années)", min_value=0, max_value=100, value=5)
            type_carte = st.selectbox("Type de carte", ["Master_card", "Visa_card"])
            region = st.selectbox("Région", ["Houston", "Orlando", "Miami"])
            genre = st.selectbox("Genre", ["Homme", "Femme"])

        submit = st.form_submit_button("Predict")

    # === ENCODAGE des variables catégorielles ===
    # Type de carte: Master_card = 0, Visa_card = 1
    type_map = {"Master_card": 0, "Visa_card": 1}
    type_encoded = type_map[type_carte]

    # Genre: Homme = 0, Femme = 1
    genre_map = {"Homme": 0, "Femme": 1}
    genre_encoded = genre_map[genre]

    # Région (fréquence encoding):
    region_map = {"Houston": 0.39509804, "Orlando": 0.31568627, "Miami": 0.28921569}
    region_encoded = region_map[region]

    # === Création du DataFrame ===
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
    
    # --- Prédiction ---
    if submit:
        D_input = xgb.DMatrix(input_data)
        prediction = model.predict(D_input)[0]
        classe = "✅ Transaction Légitime" if prediction == 0 else "🚨 Fraude Suspectée"

        st.markdown("""
            <div style='margin-top:30px; padding:20px; background-color:#e9ecef;border-left:5px solid #0d6efd; border-radius:8px;'>
                <h3>Résultat :</h3>
                <h2 style='color:{};'>{}</h2>
            </div>
        """.format("green" if prediction == 0 else "red", classe), unsafe_allow_html=True)


 # --- Affichage des valeurs sous forme de belles cartes avec effet de survol ---
    st.markdown("""
        <style>
        .card-hover {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease-in-out;
        }
        .card-hover:hover {
            box-shadow: 0px 6px 20px rgba(13, 110, 253, 0.3);
            transform: translateY(-5px);
            background-color:#cdedf9 ;
        }
        
        </style>
    """, unsafe_allow_html=True)
    
    
    
     # --- Affichage des valeurs sous forme de belles cartes ---
    st.markdown("### 🧾 Données saisies")
    cols = st.columns(2)
    for idx, col_name in enumerate(input_data.columns):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
                <div class='card-hover'>
                    <h5 style='margin-bottom: 10px; color: black;'>{col_name}</h5>
                    <p style='font-size: 22px; font-weight: bold; color: red;'>{input_data[col_name].values[0]}</p>
                </div>
            """, unsafe_allow_html=True)

    