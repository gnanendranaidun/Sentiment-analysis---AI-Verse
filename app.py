import streamlit as st
import whisper
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import wave
import numpy as np
import os
import uuid
st.set_option('server.fileWatcherType', 'none')

file_path = None

# Streamlit app structure
st.title("🎤 Multilingual ASR 💬")
# Load models
model = whisper.load_model("base")
st.write("Whisper Model Loaded!")
sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

st.write("Record your voice, and play the recorded audio:")
# audio=mic_recorder(start_prompt="⏺️",stop_prompt="⏹️",key='recorder')
audio = mic_recorder(
    start_prompt="Start Recording ⏺️",
    stop_prompt="Stop Recording ⏹️",
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None
)
if audio:       
    recordings_folder = "recordings"
    os.makedirs(recordings_folder, exist_ok=True)
    unique_filename = f"recording_{uuid.uuid4().hex}.mp3"
    file_path = os.path.join(recordings_folder, unique_filename)
    temp_audio_file_path = file_path
    
    # audio_bytes = audio["bytes"]
    # sample_width = audio["sample_width"]  # 2 bytes per sample for 16-bit PCM
    # sample_rate = audio["sample_rate"]  # 44.1 kHz sample rate
    # num_channels = 2  # 1 channel for mono, 2 for stereo

    # Create a new wave file and write the audio bytes
    # with wave.open(file_path, 'w') as wave_file:
    #     wave_file.setnchannels(num_channels)
    #     wave_file.setsampwidth(sample_width)
    #     wave_file.setframerate(sample_rate)
    #     wave_file.writeframes(audio_bytes)    
    audio_bytes = audio["bytes"]

    # Play the recorded audio
    st.audio(audio_bytes, format="audio/wav")

    # Save the audio to a file
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    st.success(f"Audio saved as {unique_filename}")
    
def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results

def get_sentiment_emoji(sentiment):
    # Define the emojis corresponding to each sentiment
    emoji_mapping = {
        "disappointment": "😞",
        "sadness": "😢",
	    "annoyance": "😠",
        "neutral": "😐",
        "disapproval": "👎",
        "realization": "😮",
        "nervousness": "😬",
        "approval": "👍",
        "joy": "😄",
        "anger": "😡",
        "embarrassment": "😳",
        "caring": "🤗",
        "remorse": "😔",
        "disgust": "🤢",
        "grief": "😥",
        "confusion": "😕",
        "relief": "😌",
        "desire": "😍",
        "admiration": "😌",
        "optimism": "😊",
        "fear": "😨",
        "love": "❤️",
        "excitement": "🎉",
        "curiosity": "🤔",
        "amusement": "😄",
        "surprise": "😲",
        "gratitude": "🙏",
        "pride": "🦁"
    }
    return emoji_mapping.get(sentiment, "")

def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score}\n"
    return sentiment_text

def inference(ans, sentiment_option):
    sentiment_results = analyze_sentiment(ans)
    sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)
    return sentiment_output

# Sentiment Option Radio
sentiment_option = st.radio("Select an option", ["Sentiment Only", "Sentiment + Score"], index=0)

# Button to trigger the processing
if st.button("Get sentiments"):
    st.write("Transcribing Audio...")
    result = model.transcribe(file_path)
    print(f"results are:\n {result}")
    ans = result["text"]
    st.write(ans)

    # Call the inference function with inputs and get outputs
    sentiment_output_value = inference(ans, sentiment_option)
    st.write(sentiment_output_value)

# Add a footer
st.markdown('''
    Whisper Model by [OpenAI](https://github.com/openai/whisper)
            ''')