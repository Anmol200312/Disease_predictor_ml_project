import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI
from fpdf import FPDF
import datetime
import time

# --- Constants for File Paths and Image Mapping ---
MODEL_PATH = "disease_model.pkl"
DATA_PATH = "Training.csv" 

# --- UPDATED SYMPTOM IMAGES MAPPED TO BODY PARTS/SYSTEMS ---
SYMPTOM_IMAGES = {
    # --- Head/Nervous System ---
    "headache": "https://cdn-icons-png.flaticon.com/128/4843/4843993.png",     
    "dizziness": "https://cdn-icons-png.flaticon.com/128/3997/3997779.png", 
    "fatigue": "https://cdn-icons-png.flaticon.com/128/13441/13441821.png",      
    "chills": "https://cdn-icons-png.flaticon.com/128/3782/3782103.png",       
    "high_fever": "https://cdn-icons-png.flaticon.com/128/3781/3781981.png",   
    "fever": "https://cdn-icons-png.flaticon.com/128/3781/3781981.png",        
    "altered_sensorium": "https://cdn-icons-png.flaticon.com/128/8657/8657731.png", 
    "loss_of_balance": "https://cdn-icons-png.flaticon.com/128/4676/4676579.png", 
    "unsteadiness": "https://cdn-icons-png.flaticon.com/128/4676/4676579.png",
    
    # --- Eye/Ear/Nose/Throat (EENT) ---
    "runny_nose": "https://cdn-icons-png.flaticon.com/128/2853/2853869.png",     
    "congestion": "https://cdn-icons-png.flaticon.com/128/10447/10447974.png",    
    "sinus_pressure": "https://cdn-icons-png.flaticon.com/128/15694/15694871.png",
    "cough": "https://cdn-icons-png.flaticon.com/128/2811/2811503.png",       
    "throat_irritation": "https://cdn-icons-png.flaticon.com/128/3954/3954169.png", 
    "sore_throat": "https://cdn-icons-png.flaticon.com/128/3954/3954169.png",
    "red_spots_over_body": "https://cdn-icons-png.flaticon.com/128/836/836916.png", 
    "puffy_eyes": "https://cdn-icons-png.flaticon.com/128/158/158826.png", 
    "sunken_eyes": "https://cdn-icons-png.flaticon.com/128/158/158814.png",
    
    # --- Respiratory ---
    "breathlessness": "https://cdn-icons-png.flaticon.com/128/13462/13462406.png", 
    "chest_pain": "https://cdn-icons-png.flaticon.com/128/3782/3782076.png",
    "phlegm": "https://cdn-icons-png.flaticon.com/128/2811/2811503.png",
    "mucoid_sputum": "https://cdn-icons-png.flaticon.com/128/2811/2811503.png",
    
    # --- Digestive/Abdominal ---
    "abdominal_pain": "https://cdn-icons-png.flaticon.com/128/5730/5730077.png", 
    "vomiting": "https://cdn-icons-png.flaticon.com/128/5730/5730148.png", 
    "diarrhoea": "https://cdn-icons-png.flaticon.com/128/3954/3954106.png", 
    "nausea": "https://cdn-icons-png.flaticon.com/128/5641/5641608.png", 
    "loss_of_appetite": "https://cdn-icons-png.flaticon.com/128/1847/1847250.png",
    "constipation": "https://cdn-icons-png.flaticon.com/128/1000/1000538.png",
    "abdominal_distension": "https://cdn-icons-png.flaticon.com/128/5730/5730077.png",
    "acidity": "https://cdn-icons-png.flaticon.com/128/2405/2405423.png", 
    "indigestion": "https://cdn-icons-png.flaticon.com/128/2405/2405423.png",
    "stomach_bleeding": "https://cdn-icons-png.flaticon.com/128/850/850960.png", 
    "passage_of_gases": "https://cdn-icons-png.flaticon.com/128/598/598063.png",
    "blackheads": "https://cdn-icons-png.flaticon.com/128/1076/1076779.png", 
    
    # --- Pain/Musculoskeletal/Joints ---
    "joint_pain": "https://cdn-icons-png.flaticon.com/128/9961/9961835.png", 
    "muscle_pain": "https://cdn-icons-png.flaticon.com/128/5506/5506780.png",
    "pain_in_anal_region": "https://cdn-icons-png.flaticon.com/128/870/870634.png",
    "back_pain": "https://cdn-icons-png.flaticon.com/128/480/480112.png",
    "neck_pain": "https://cdn-icons-png.flaticon.com/128/3782/3782068.png",
    "swelling_joints": "https://cdn-icons-png.flaticon.com/128/9961/9961835.png",
    "movement_stiffness": "https://cdn-icons-png.flaticon.com/128/7391/7391672.png",
    
    # --- Skin/Hair/Nails/Appearance ---
    "skin_rash": "https://cdn-icons-png.flaticon.com/128/3954/3954175.png",
    "itching": "https://cdn-icons-png.flaticon.com/128/3954/3954175.png",
    "skin_peeling": "https://cdn-icons-png.flaticon.com/128/3954/3954175.png",
    "patches_in_throat": "https://cdn-icons-png.flaticon.com/128/3468/3468641.png",
    "dischromic_patches": "https://cdn-icons-png.flaticon.com/128/836/836916.png",
    "internal_itching": "https://cdn-icons-png.flaticon.com/128/3954/3954175.png",
    "silver_like_dusting": "https://cdn-icons-png.flaticon.com/128/836/836916.png",
    "small_dents_in_nails": "https://cdn-icons-png.flaticon.com/128/836/836916.png",
    "brittle_nails": "https://cdn-icons-png.flaticon.com/128/3063/3063558.png", 
    
    # --- Urinary/Genital ---
    "bladder_discomfort": "https://cdn-icons-png.flaticon.com/128/3774/3774574.png", 
    "foul_smell_of_urine": "https://cdn-icons-png.flaticon.com/128/3774/3774574.png",
    "burning_micturition": "https://cdn-icons-png.flaticon.com/128/3774/3774574.png",
    "polyuria": "https://cdn-icons-png.flaticon.com/128/3774/3774574.png",
    "spotting_ urination": "https://cdn-icons-png.flaticon.com/128/3774/3774574.png",
    
    # --- Weight/General Appearance ---
    "weight_loss": "https://cdn-icons-png.flaticon.com/128/2150/2150824.png", 
    "weight_gain": "https://cdn-icons-png.flaticon.com/128/2150/2150824.png",
    "swelling_of_stomach": "https://cdn-icons-png.flaticon.com/128/5730/5730077.png",
    "distention_of_abdomen": "https://cdn-icons-png.flaticon.com/128/5730/5730077.png",
    "enlarged_thyroid": "https://cdn-icons-png.flaticon.com/128/1324/1324294.png", 
    "swollen_legs": "https://cdn-icons-png.flaticon.com/128/9961/9961835.png", 
    
    # --- Cardiovascular/Blood ---
    "palpitations": "https://cdn-icons-png.flaticon.com/128/771/771438.png", 
    "cold_hands_and_feets": "https://cdn-icons-png.flaticon.com/128/16779/16779865.png", 
    "blood_in_sputum": "https://cdn-icons-png.flaticon.com/128/2968/2968939.png", 
    
    # --- Other/Mental/Behavioral ---
    "anxiety": "https://cdn-icons-png.flaticon.com/128/7145/7145123.png", 
    "mood_swings": "https://cdn-icons-png.flaticon.com/128/12370/12370029.png", 
    "irritability": "https://cdn-icons-png.flaticon.com/128/3590/3590305.png", 
    "restlessness": "https://cdn-icons-png.flaticon.com/128/10371/10371324.png", 
}
DEFAULT_IMAGE = "https://cdn-icons-png.flaticon.com/512/4320/4320372.png" 


# ---------- Setup OpenAI API ----------
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("⚠️ OpenAI API Key not found.")
    client = None

# ---------- Load Model & Data ----------
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        symptoms = [col for col in df.columns if col.lower() != "prognosis"]
        return model, df, symptoms
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Check if '{MODEL_PATH}' and '{DATA_PATH}' exist in the same directory.")
        return None, None, []
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        return None, None, []

model, df, symptoms = load_resources()

# --- Streamlit Page Config ---
st.set_page_config(page_title="🩺 AI Symptom Checker", page_icon="🤖", layout="wide")

# --- Custom CSS (Updated for modern look and card styling) ---
st.markdown("""
<style>
/* General Aesthetics */
h1, h2, h3, h4 { color: #1a237e; }
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: nowrap;
    border-radius: 4px 4px 0 0;
    background-color: #f0f2f6;
    color: #1a237e;
    font-weight: 600;
    padding: 10px 15px;
    margin-right: 5px;
}
.stTabs [aria-selected="true"] {
    background-color: #2196f3;
    color: white;
    border-bottom: 3px solid #2196f3;
}
/* Button Styling */
.stButton>button {
    color: white;
    background-color: #2196f3;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #1e88e5;
    transform: translateY(-1px);
}
/* Symptom Card Styling */
.symptom-card { 
    text-align: center; 
    padding: 10px; 
    margin: 5px 0; 
    border: 1px solid #e0e0e0; 
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.symptom-card-selected {
    border: 3px solid #2196f3;
    background-color: #e3f2fd;
    box-shadow: 0 4px 8px rgba(33, 150, 243, 0.2);
}
.symptom-card img { 
    width: 50px; 
    height: 50px; 
    border-radius: 50%; 
    margin-bottom: 5px; 
}
.symptom-name { 
    font-size: 14px; 
    color: #1a237e; 
    font-weight: 500; 
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are a friendly and empathetic AI doctor. All advice is informational. Encourage consulting a professional."}]
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

# --- Main App Header ---
st.title("🤖 AI-Enabled Disease Symptom Checker")
st.markdown("A preliminary tool to check your symptoms and get informational guidance.")
st.markdown("---")

# --- Helper Function to Update Selection ---
def update_selection(symptom_name):
    """Toggles the symptom in the session state list."""
    if symptom_name in st.session_state.selected_symptoms:
        st.session_state.selected_symptoms.remove(symptom_name)
    else:
        st.session_state.selected_symptoms.append(symptom_name)

# =========================================================================
# 🏗️ UI Structure using Streamlit Tabs
# =========================================================================
tab1, tab2, tab3 = st.tabs(["1️⃣ Select Symptoms", "2️⃣ Prediction & Info", "3️⃣ Chat with AI Doctor"])


# 1️⃣ Symptom Selection Tab
with tab1:
    st.header("🩹 Select Your Symptoms")
    st.markdown("Click on the boxes for all symptoms that match your condition. Icons indicate the affected body part.")

    if symptoms:
        N_COLS = 5
        cols = st.columns(N_COLS)

        for i, symptom in enumerate(symptoms):
            display_name = symptom.replace('_', ' ').title()
            img_url = SYMPTOM_IMAGES.get(symptom, DEFAULT_IMAGE)
            is_selected = symptom in st.session_state.selected_symptoms
            item_class = "symptom-card symptom-card-selected" if is_selected else "symptom-card"

            with cols[i % N_COLS]:
                # Use a form/button to capture the click and update the state
                with st.form(key=f"form_{symptom}"):
                    st.markdown(f"""
                    <div class="{item_class}">
                        <img src="{img_url}" alt="{symptom}">
                        <div class="symptom-name">{display_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    submitted = st.form_submit_button(
                        label=" ", 
                        use_container_width=True,
                        help=f"Toggle {display_name} selection"
                    )

                    if submitted:
                        update_selection(symptom)
                        st.rerun()

        st.markdown("---")
        
        # Display Selected Symptoms
        st.subheader("🧾 Your Current Selection:")
        if st.session_state.selected_symptoms:
            selected_display = [s.replace('_', ' ').title() for s in st.session_state.selected_symptoms]
            st.success(f"**Selected Symptoms:** {', '.join(selected_display)}")
        else:
            st.info("💡 Start by selecting symptoms above.")

        # Action Buttons
        col_predict, col_clear = st.columns([3, 1])
        with col_predict:
            predict_clicked = st.button("🔍 **Run Disease Prediction**", key="predict_btn_t1", use_container_width=True)
        with col_clear:
            if st.button("❌ Clear All", use_container_width=True, key="clear_btn_t1"):
                st.session_state.selected_symptoms = []
                st.session_state.prediction = None
                st.session_state.ai_explanation = None
                st.session_state.chat_history = [{"role": "system", "content": "You are a friendly and empathetic AI doctor."}]
                st.rerun()

    else:
        st.error("Cannot load symptom data. Check file paths and loading function.")

# 2️⃣ Prediction & Info Tab (Prediction Logic)
if st.session_state.get("predict_btn_t1"):
    del st.session_state["predict_btn_t1"]
    
    selected_symptoms = st.session_state.selected_symptoms
    if not selected_symptoms:
        pass
    else:
        st.session_state.prediction = None
        st.session_state.ai_explanation = None
        
        with st.spinner("Analyzing symptoms and consulting AI doctor..."):
            input_data = np.zeros(len(symptoms))
            for s in selected_symptoms:
                if s in symptoms:
                    input_data[list(symptoms).index(s)] = 1
            input_df = pd.DataFrame([input_data], columns=symptoms)
            
            if model and hasattr(model, "feature_names_in_"):
                input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

            try:
                prediction = model.predict(input_df)[0].title()
                st.session_state.prediction = prediction
                
                # Fetch AI explanation (with quota error handling)
                if client:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a friendly AI health assistant. Explain the disease in simple, structured paragraphs, covering symptoms, causes, treatments, and prevention. Stress non-professional advice."},
                                {"role": "user", "content": f"Explain what {prediction} is, its common symptoms, causes, general treatments, and prevention tips. Format the response using markdown headings for clarity."}
                            ],
                        )
                        st.session_state.ai_explanation = response.choices[0].message.content
                    except Exception as e:
                        # Graceful handling for quota errors
                        if "insufficient_quota" in str(e):
                            st.session_state.ai_explanation = "⚠️ **AI Explanation Failed:** The OpenAI API key has exceeded its usage quota."
                        else:
                            st.session_state.ai_explanation = f"⚠️ AI explanation failed: {e}"
                        st.error(st.session_state.ai_explanation)
                else:
                    st.session_state.ai_explanation = "⚠️ AI service unavailable due to API client failure."

            except Exception as e:
                st.session_state.prediction = "Prediction Failed"
                st.session_state.ai_explanation = f"❌ Prediction failed: {e}"

with tab2:
    st.header("🧠 Prediction & Detailed Information")
    
    if st.session_state.prediction and st.session_state.prediction != "Prediction Failed":
        st.subheader(f"✅ Predicted Condition: **{st.session_state.prediction}**")
        st.info(f"**Based on:** {', '.join([s.replace('_', ' ').title() for s in st.session_state.selected_symptoms])}")
        
        st.markdown("---")
        
        st.subheader("📚 AI Health Guide")
        st.markdown(st.session_state.ai_explanation)
        
        st.warning("🚨 **Disclaimer:** This is **not a professional diagnosis**. Always consult a qualified healthcare provider for any medical concerns.")
    elif st.session_state.prediction == "Prediction Failed":
        st.error(st.session_state.ai_explanation)
    else:
        st.info("💡 Select your symptoms in the **Select Symptoms** tab and click the **Run Disease Prediction** button to see your results here.")


# 3️⃣ Chat with AI Doctor Tab
with tab3:
    st.header("💬 Chat with AI Doctor")
    st.markdown("Ask follow-up questions about your condition or general health.")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            continue

        role = "assistant" if msg["role"] == "assistant" else "user"
        avatar = "🤖" if role == "assistant" else "🧑‍🔬"
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    # Chat Input Form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your question...",
            placeholder="e.g., What are the risk factors for this disease?",
            key="chat_input_box_unique"
        )
        send = st.form_submit_button("Send ⬆️", use_container_width=True)

    if send and user_input.strip() and client:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun()

    # AI response logic (runs after rerun triggered by send)
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user" and client:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI Doctor is thinking..."):
                try:
                    chat_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.chat_history
                    )
                    reply = chat_response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.markdown(reply)
                except Exception as e:
                    # Graceful handling for quota errors
                    if "insufficient_quota" in str(e):
                         st.error("❌ **Chat Failed:** The OpenAI API key has exceeded its usage quota. ")
                    else:
                        st.error(f"AI chat failed: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error and cannot respond right now."})
        st.rerun()

    # --- Save Chat Section ---
    st.markdown("---")
    st.subheader("📄 Save Your Consultation")
    if st.button("💾 Download Chat as PDF", use_container_width=True):
        if len(st.session_state.chat_history) > 1:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="AI Doctor Consultation Report", ln=True, align="C")
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True, align="C")
            pdf.ln(5)
            
            if st.session_state.prediction:
                pdf.set_font("Arial", "B", size=12)
                pdf.multi_cell(0, 7, f"Predicted Condition: {st.session_state.prediction}")
                pdf.ln(2)

            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 5, "-" * 50)
            pdf.ln(2)
            
            for msg in st.session_state.chat_history[1:]:
                role = "You" if msg["role"] == "user" else "AI Doctor"
                pdf.set_font("Arial", "B", size=12)
                pdf.multi_cell(0, 7, f"{role}:", align="L")
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 7, f"{msg['content']}")
                pdf.ln(2)
                
            chat_file = "AI_Doctor_Consultation.pdf"
            pdf.output(chat_file)
            
            with open(chat_file, "rb") as file:
                st.download_button("⬇️ Download PDF", data=file, file_name=chat_file, mime="application/pdf", use_container_width=True)
        else:
            st.warning("⚠️ Chat history is empty. Start a conversation first.")
            
# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;font-size:12px;'>© 2025 <b>PRIYANSHU RAJ</b> | **Disclaimer:** This tool is for informational purposes only. Consult a doctor for diagnosis.</p>", unsafe_allow_html=True)