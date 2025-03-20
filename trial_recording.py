import streamlit as st
from streamlit_mic_recorder import mic_recorder

st.title("🎤 Streamlit Mic Recorder")

# Record audio
audio_data = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None
)

if audio_data:
    # Access audio bytes
    audio_bytes = audio_data["bytes"]

    # Play the recorded audio
    st.audio(audio_bytes, format="audio/wav")

    # Save the audio to a file
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_bytes)
    st.success("Audio saved as recorded_audio.wav")
