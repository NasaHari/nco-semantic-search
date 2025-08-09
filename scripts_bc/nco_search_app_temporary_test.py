import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Mic Recorder Test with streamlit-webrtc")
webrtc_streamer(key="test")
