import streamlit as st
from scripts.search import search_nco
import json

st.title("NCO Semantic Search PoC")

query = st.text_input("Enter job description (e.g., 'sewing machine operator')")
if st.button("Search"):
    if query:
        results = json.loads(search_nco(query))
        st.write("Top Matches:")
        for res in results:
            st.markdown(f"**Code:** {res['code']} | **Title:** {res['title']} | **Score:** {res['score']:.2f}")
            st.text(res['description'])
    else:
        st.error("Please enter a query.")
