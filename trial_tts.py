import streamlit as st
import tts_tutorial as tts

# Set the page title
st.title("Audio Player with Streamlit")
file_path = tts.text_to_speech("फिर, HC Sen Marg पर continue करें, और Paranthe Wali Gali तक drive करें।","hi-In")
# Upload audio file

if file_path is not None:
    st.audio(file_path, format='audio/mp3')  # Streamlit auto detects the format
    st.success("Playing your audio!")
else:
    st.info("Please upload an audio file to play.")
