# app/app.py
import streamlit as st
import sys
import os
from langdetect import detect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.search import NCOSearcher
from scripts.incremental_update import incremental_update
import json
import streamlit as st
import pandas as pd
import json
import os
from scripts.search import NCOSearcher, load_all_searchers, MODEL_ALIASES,ensemble_search
import subprocess
import csv
from datetime import datetime
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from typing import List, Dict
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pydub import AudioSegment
import io
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# ---- Create authenticator object ----
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
admin_users = ["harikrishnan123", "admin"]

# ---- Login widget ----
authenticator.login(location='main')

if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f"Welcome {st.session_state.get('name')}")
elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')
    st.stop()
elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')
    st.stop()



def log_search(query, results, user_id="anonymous"):
    top_codes = [r["Unit_Code"] for r in results] if results else []
    with open(SEARCH_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), user_id, query, "|".join(top_codes)])

def log_admin_action(user_id, action, details=""):
    with open(ADMIN_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), user_id, action, details])

current_user = st.session_state.get("name")


SEARCH_LOG_FILE = "logs/search_log.csv"
ADMIN_LOG_FILE = "logs/admin_actions_log.csv"
@st.cache_resource
def load_all_models():
    model_names = [
        "lbs",  # Alias for "krutrim-ai-labs/Vyakyarth"
        "min" # Alias for "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    # Get full model names using aliases
    full_model_names = [MODEL_ALIASES[alias] for alias in model_names]
    
    # Load all selected models
    searcher = load_all_searchers(full_model_names)
    return searcher

# Preload the searchers (this loads models only once on startup)
if "models_loaded" not in st.session_state:
    with st.spinner("Loading all models, embeddings, and indexes... This may take a moment."):
        st.session_state.searcher = load_all_models()
language_options = {
    "English (India)": "en-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Malayalam": "ml-IN",
    "Kannada": "kn-IN",
    "Telugu": "te-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN"
}

selected_language =None

def transcribe_audio(file_path, language_code):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data, language=language_code)
            return text
        except sr.UnknownValueError:
            st.warning("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("API connection error.")
    return None
searcher = st.session_state.searcher

st.title("NCO Semantic Search")
use_ensemble =False

# --- Search UI ---

if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False
if "query" not in st.session_state:
    st.session_state.query = ""
if "query_from_audio" in st.session_state:
    st.session_state.query = st.session_state.pop("query_from_audio")

def do_search():
    st.session_state.search_triggered = True

query_value = st.session_state.pop("query_from_audio", st.session_state.get("query", ""))
col_text,select, col_mic = st.columns([5,2, 1])

col_text, col_mic = st.columns([3, 1])
with col_text:
    query = st.text_input(
    "Enter job description or speak:",
    key="query",
    value=query_value,
    on_change=do_search
)   
    if st.button("Search") :
        do_search()

with col_mic:
    st.markdown(
            """
            <div style="padding-top: 29px;">
            """,
            unsafe_allow_html=True,
        )    
    audio = mic_recorder(
        start_prompt="🎤 Record", 
        stop_prompt="⏹ Stop", 
        just_once=True, 
        use_container_width=True
    )
    selected_language = st.selectbox("Select language for transcription:", options=list(language_options.keys()))

    if audio:
        audio_bytes = audio["bytes"]
        # Convert audio bytes to WAV format using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))  # Auto detects format
        audio_segment.export("temp.wav", format="wav")
        
        spoken_text = transcribe_audio("temp.wav",language_code=language_options[selected_language])

        if spoken_text:
            st.session_state["query_from_audio"] = spoken_text  # Store separately to avoid conflict
            st.success(f"Voice input: {spoken_text}")
            do_search()
            st.rerun()
INDIC_LANGS = {"hi", "ta", "te", "kn", "ml", "bn", "mr"}


if st.session_state.search_triggered and st.session_state.query.strip():
    with st.spinner("Searching..."):
        if use_ensemble:
           # Use the new ensemble_search function
            results, fallback_suggestions = ensemble_search(st.session_state.searcher, st.session_state.query, top_k=10)
            subheader_text = "Top matches (Ensemble)"
        else:
            detected_lang = detect(st.session_state.query)
            print(detected_lang)
            if detected_lang in INDIC_LANGS:
                 results, fallback_suggestions= st.session_state.searcher[MODEL_ALIASES['lbs']].search(st.session_state.query, top_k=10,use_synonyms=1)
            else:
                 results, fallback_suggestions= st.session_state.searcher[MODEL_ALIASES['min']].search(st.session_state.query, top_k=10,use_synonyms=1)
        
    
        if not results:
            st.warning("No results found.")
            if fallback_suggestions:
                st.subheader("Did you mean...?")
                for i, suggestion in enumerate(fallback_suggestions, 1):
                    if st.button(f"{i}. {suggestion}", key=f"suggestion_{i}"):
                        st.session_state.query = suggestion
                        st.session_state.search_triggered = True
                        st.rerun()
        else:
            log_search(st.session_state.query, results)

            st.subheader(f"Top {len(results)} matches:")

            if "selected_result" not in st.session_state:
                st.session_state.selected_result = None

            for i, res in enumerate(results, 1):
                with st.expander(f"{i}. {res['Title']} (Code: {res['Unit_Code']}) — Score: {res['Score']:.2f}"):
                    st.write(f"**Unit Code:** `{res['Unit_Code']}`")
                    st.write(f"**Title:** {res['Title']}")
                    st.write(f"**Confidence Score:** {res['Score']:.2f}")
                    st.write(res["Description"])


    # Reset search trigger to avoid repeated searches on rerun
    st.session_state.search_triggered = False

elif st.session_state.query.strip() == "":
    st.info("Enter a search query and press Enter.")

# Create two columns: left for Add Entry, right for Upload Dataset
col1, col2 = st.columns(2)
if st.session_state.get("username") in admin_users:
    with col1:
        # --- Add New Entry Section ---
        if "show_form" not in st.session_state:
            st.session_state.show_form = False

        def toggle_form():
            st.session_state.show_form = not st.session_state.show_form

        st.button("➕ Add New NCO Entry", on_click=toggle_form)

        if st.session_state.show_form:
            st.header("Add New NCO Entry")

            with st.form("add_entry_form", clear_on_submit=True):
                Division = st.text_input("Division")
                Division_Description = st.text_area("Division Description")
                Sub_Division = st.text_input("Sub Division")
                Sub_Division_Description = st.text_area("Sub Division Description")
                Group = st.text_input("Group")
                Group_Description = st.text_area("Group Description")
                Family = st.text_input("Family")
                Family_Description = st.text_area("Family Description")
                Unit_Code = st.text_input("Unit Code")
                Unit_Title = st.text_input("Unit Title")
                Unit_Description = st.text_area("Unit Description")
                NCO_2004 = st.text_input("NCO 2004")
                QP_NOS_Reference = st.text_input("QP NOS Reference")
                QP_NOS_Name = st.text_input("QP NOS Name")
                NSQF_Level = st.number_input("NSQF Level", min_value=1, max_value=10, step=1)

                submitted = st.form_submit_button("Submit New Entry")

                if submitted:
                    new_row = {
                        "Division": Division,
                        "Division_Description": Division_Description,
                        "Sub_Division": Sub_Division,
                        "Sub_Division_Description": Sub_Division_Description,
                        "Group": Group,
                        "Group_Description": Group_Description,
                        "Family": Family,
                        "Family_Description": Family_Description,
                        "Unit_Code": Unit_Code,
                        "Unit_Title": Unit_Title,
                        "Unit_Description": Unit_Description,
                        "NCO_2004": NCO_2004,
                        "QP_NOS Reference": QP_NOS_Reference,
                        "QP_NOS Name": QP_NOS_Name,
                        "NSQF_Level": NSQF_Level,
                    }
                    new_row['text'] = (new_row.get('Unit_Title', '') + ' ' + new_row.get('Unit_Description', '')).lower()

                    st.success("New entry added to dataset.")
                    log_admin_action(
        user_id="admin",  # replace with real user if you add auth
        action="Add New Entry",
        details=f"Unit_Code: {Unit_Code}, Unit_Title: {Unit_Title}"
    )
                    st.session_state.show_form = False  # Hide form after submit


                
                    # Trigger incremental update
                    with st.spinner("Updating embeddings and index incrementally..."):
                        incremental_update(
                            data_csv="data/processed/nco_cleaned.csv",
                            embeddings_path="embeddings/nco_embeddings_vya.npy",  # Match your paths
                            index_path="embeddings/nco_index-vya.faiss",
                            model_name="krutrim-ai-labs/Vyakyarth",  # Match your model
                            new_rows=[new_row]  # Pass the single new row as a list
                        )

                    # Clear the cached searcher so model/data reloads on next search
                    load_all_models.clear()
                    st.success("Embeddings and index updated successfully!.")

    with col2:
        if "show_upload" not in st.session_state:
            st.session_state.show_upload = False
        if "upload_done" not in st.session_state:
            st.session_state.upload_done = False

        def toggle_upload():
            st.session_state.show_upload = not st.session_state.show_upload
            if not st.session_state.show_upload:
                st.session_state.upload_done = False  # reset flag when form closes

        st.button("📤 Upload and Append Dataset", on_click=toggle_upload)

        if st.session_state.show_upload:
            uploaded_file = st.file_uploader("Upload CSV or XLSX file to append", type=["csv", "xlsx"], key="file_uploader")

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df_new = pd.read_csv(uploaded_file)
                    else:
                        df_new = pd.read_excel(uploaded_file)

                    st.write("Preview of uploaded data:")
                    st.dataframe(df_new.head())

                    new_rows = df_new.to_dict(orient="records")
                    for row in new_rows:
                        row['text'] = (str(row.get('Unit_Title', '')) + " " + str(row.get('Unit_Description', ''))).lower()

                    csv_path = "data/processed/nco_cleaned.csv"
                    if os.path.exists(csv_path):
                        df_existing = pd.read_csv(csv_path)

                        if list(df_new.columns) != list(df_existing.columns):
                            st.error("Uploaded file columns do NOT match existing dataset columns.")
                        else:
                            if st.button("Append and Save to Dataset"):
                                try:
                                    incremental_update(
                                        data_csv=csv_path,
                                        embeddings_path="embeddings/nco_embeddings_vya.npy",
                                        index_path="embeddings/nco_index-vya.faiss",
                                        model_name="krutrim-ai-labs/Vyakyarth",
                                        new_rows=new_rows
                                    )
                                    st.success("Data appended and index updated successfully!")
                                    log_admin_action(
                                        user_id=current_username,
                                        action="Upload and Append Dataset",
                                        details=f"Rows appended: {len(new_rows)}, File: {uploaded_file.name}"
                                    )
                                    st.session_state.show_upload = False
                                    st.session_state.upload_done = True
                                    # Clear file uploader by resetting key (optional)
                                    st.rerun()  # rerun to refresh UI with upload panel closed
                                except Exception as e:
                                    st.error(f"Incremental update failed: {e}")

                except Exception as e:
                    st.error(f"Error loading file: {e}")

        else:
            if st.session_state.upload_done:
                st.info("Upload completed. Click 'Upload and Append Dataset' to upload more.")


    # --- Admin-only features here ---
else:
    st.info("Add section is restricted to admin users.")



       
