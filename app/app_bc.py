# app/app.py
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.search import NCOSearcher
import json
import streamlit as st
import pandas as pd
import json
import os
from scripts.search import NCOSearcher
import subprocess

@st.cache_resource
def load_searcher():
    return NCOSearcher()

# Global searcher instance cached by Streamlit
searcher = load_searcher()

st.title("NCO Semantic Search")

# --- Search UI ---

if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False
if "query" not in st.session_state:
    st.session_state.query = ""

def do_search():
    st.session_state.search_triggered = True

query = st.text_input("Enter job description (e.g., 'sewing machine operator')",
                      key="query", on_change=do_search)

if st.session_state.search_triggered and st.session_state.query.strip():
    with st.spinner("Searching..."):
        results_json = searcher.search(st.session_state.query, top_k=10)
        results = json.loads(results_json)

        if not results:
            st.warning("No results found.")
        else:
            st.subheader(f"Top {len(results)} matches:")
            for i, res in enumerate(results, 1):
                st.markdown(f"**{i}. Code:** `{res['Unit_Code']}`  ")
                st.markdown(f"**Title:** {res['Title']}")
                st.markdown(f"**Confidence Score:** {res['Score']:.2f}")
                st.write(res['Description'])
                st.markdown("---")

    # Reset search trigger to avoid repeated searches on rerun
    st.session_state.search_triggered = False

elif st.session_state.query.strip() == "":
    st.info("Enter a search query and press Enter.")
st.markdown("---")

# Button to show/hide the new entry form
if "show_form" not in st.session_state:
    st.session_state.show_form = False

def toggle_form():
    st.session_state.show_form = not st.session_state.show_form

if st.button("Add New NCO Entry", on_click=toggle_form):
    pass  # The button click toggles show_form

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

            csv_path = "data/processed/nco_cleaned.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])

            df.to_csv(csv_path, index=False)

            st.success("New entry added to dataset.")
            st.session_state.show_form = False  # Hide form after submit

            # Clear the cached searcher so model/data reloads on next search
            load_searcher.clear()
            st.info("Please update embeddings and index to include the new entry.")
# Toggle for upload section visibility
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

def toggle_upload():
    st.session_state.show_upload = not st.session_state.show_upload

if st.button("Upload and Append Dataset", on_click=toggle_upload):
    pass  # Button toggles visibility

if st.session_state.show_upload:
    uploaded_file = st.file_uploader("Upload CSV or XLSX file to append", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_new = pd.read_csv(uploaded_file)
            else:
                df_new = pd.read_excel(uploaded_file)

            st.write("Preview of uploaded data:")
            st.dataframe(df_new.head())

            csv_path = "data/processed/nco_cleaned.csv"
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path)

                # Check if columns match exactly (order and names)
                if list(df_new.columns) != list(df_existing.columns):
                    st.error("Uploaded file columns do NOT match existing dataset columns.")
                else:
                    if st.button("Append and Save to Dataset"):
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        df_combined.to_csv(csv_path, index=False)
                        st.success(f"Data appended and saved to {csv_path}")
                        st.experimental_rerun()

            else:
                st.info("No existing dataset found. Upload will create a new dataset.")

                if st.button("Save uploaded data as new dataset"):
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    df_new.to_csv(csv_path, index=False)
                    st.success(f"New dataset saved to {csv_path}")
                    st.experimental_rerun()

        except Exception as e:
            st.error(f"Error loading file: {e}")

st.markdown("---")
st.header("Update Embeddings and FAISS Index")

if st.button("Run Update Pipeline (Preprocess → Embed → Index)"):
    with st.spinner("Running update pipeline... This may take several minutes."):

        # Example: Call your existing scripts sequentially using subprocess 
        # Make sure your scripts are executable and properly located!

        def run_command(cmd):
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Command failed: {cmd}\nError: {result.stderr}")
                raise RuntimeError(f"Update pipeline failed at: {cmd}")
            filename = os.path.basename(cmd.split()[1])  # gets 'preprocess.py'
            name_without_ext = os.path.splitext(filename)[0]
            st.write(f"Success: {name_without_ext}")

        try:
            # Adjust commands according to your actual scripts and parameters

            run_command("python scripts/preprocess.py")

            # 2. Generate embeddings
            run_command("python scripts/embed.py ")

            # 3. Build index
            run_command("python scripts/index.py")

            st.success("Update pipeline completed successfully!")
            
            # Clear searcher cache to reload after pipeline update
            load_searcher.clear()
        except Exception as e:
            st.error("Update pipeline terminated due to errors.")

       
