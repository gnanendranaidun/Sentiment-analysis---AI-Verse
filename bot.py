# import streamlit as st
# import os
# import uuid
# from streamlit_mic_recorder import mic_recorder
# import Sarvam_STT
# import Google_Translate
# import whisper
# from transformers import pipeline   
# import requests
# import json

# # Load models
# model = whisper.load_model("base")
# sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

# # Title
# st.title("🎙️ AI Voice Emotion Bot")

# # Tabs
# tab1, tab2 = st.tabs(["🎤 Voice Analysis", "💬 Chat Bot"])

# def analyze_sentiment(text):
#     results = sentiment_analysis(text)
#     return {result['label']: result['score'] for result in results}

# def get_sentiment_emoji(sentiment):
#     emoji_mapping = {
#         "disappointment": "😞", "sadness": "😢", "annoyance": "😠",
#         "neutral": "😐", "disapproval": "👎", "realization": "😮",
#         "nervousness": "😬", "approval": "👍", "joy": "😄", "anger": "😡",
#         "embarrassment": "😳", "caring": "🤗", "remorse": "😔",
#         "disgust": "🤢", "grief": "😥", "confusion": "😕",
#         "relief": "😌", "desire": "😍", "admiration": "😌",
#         "optimism": "😊", "fear": "😨", "love": "❤️",
#         "excitement": "🎉", "curiosity": "🤔", "amusement": "😄",
#         "surprise": "😲", "gratitude": "🙏", "pride": "🦁"
#     }
#     return emoji_mapping.get(sentiment, "")

# def display_sentiment_results(sentiment_results, option):
#     sentiment_text = ""
#     for sentiment, score in sentiment_results.items():
#         emoji = get_sentiment_emoji(sentiment)
#         if option == "Sentiment Only":
#             sentiment_text += f"{sentiment} {emoji}\n"
#         elif option == "Sentiment + Score":
#             sentiment_text += f"{sentiment} {emoji}: {score:.2f}\n"
#     return sentiment_text

# def inference(ans, sentiment_option):
#     sentiment_results = analyze_sentiment(ans)
#     return display_sentiment_results(sentiment_results, sentiment_option)

# def ollama_response(user_input):
#     OLLAMA_API_URL = "http://localhost:11434/api/generate"
#     payload = {
#         "model": "llama3.2",  # Replace with your model name
#         "prompt": user_input,
#         "stream": True
#     }
#     full_response = ""

#     try:
#         with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
#             for line in response.iter_lines():
#                 if line:
#                     chunk = json.loads(line)
#                     if 'response' in chunk:
#                         full_response += chunk['response']
#         return full_response.strip()
#     except Exception as e:
#         st.error(f"Ollama streaming error: {str(e)}")
#         try:
#             response = requests.post(OLLAMA_API_URL, json={**payload, "stream": False})
#             if response.status_code == 200:
#                 return response.json()["response"]
#             else:
#                 return "Error communicating with Ollama API"
#         except Exception as e:
#             return f"Fallback failed: {str(e)}"

# # Voice Analysis Tab
# with tab1:
#     st.header("Voice Analysis")
#     audio = mic_recorder(
#         start_prompt="🎙️ Start Recording",
#         stop_prompt="⏹️ Stop Recording",
#         just_once=False,
#         use_container_width=True,
#         key='voice_recorder'
#     )

#     if audio:
#         try:
#             os.makedirs("recordings", exist_ok=True)
#             file_path = f"recordings/recording_{uuid.uuid4().hex}.wav"

#             st.audio(audio["bytes"], format="audio/wav")
#             with open(file_path, "wb") as f:
#                 f.write(audio["bytes"])

#             if st.button("🔍 Analyze Voice Emotion"):
#                 with st.spinner("Analyzing..."):
#                     results = Sarvam_STT.detect_and_translate(file_path)
#                     text = results["transcript"]
#                     st.success(f"Transcript: {text}")

#                     if results["language_code"] != "en":
#                         text = Google_Translate.detect_and_translate(text)

#                     sentiment_output = inference(text, "Sentiment + Score")
#                     st.markdown("#### Emotion Results")
#                     st.markdown(sentiment_output)

#                     st.info("""
#                         Note: AI accuracy depends on background noise, speech clarity, and dataset quality.
#                     """)
#         except Exception as e:
#             st.error(f"Processing Error: {str(e)}")

# # Chat Bot Tab (Ollama)
# with tab2:
#     st.header("Chat with Ollama LLM 💬")
#     user_prompt = st.text_input("Enter your question to Ollama LLM")
#     if st.button("Ask"):
#         if user_prompt.strip() != "":
#             with st.spinner("Generating response..."):
#                 response_text = ollama_response(user_prompt)
#                 st.markdown(response_text)
#         else:
#             st.warning("Please enter a valid question!")
import streamlit as st
import os
import uuid
from streamlit_mic_recorder import mic_recorder
import Sarvam_STT
import Google_Translate
import whisper
from transformers import pipeline   
import requests
import json

# Load models
model = whisper.load_model("base")
sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

# Title
st.title("🎙️ AI Voice Emotion Bot")

# Tabs
tab1, tab2 = st.tabs(["🎤 Voice Analysis to Chat", "💬 Text Chat with Ollama"])

def analyze_sentiment(text):
    results = sentiment_analysis(text)
    return {result['label']: result['score'] for result in results}

def get_sentiment_emoji(sentiment):
    emoji_mapping = {
        "disappointment": "😞", "sadness": "😢", "annoyance": "😠",
        "neutral": "😐", "disapproval": "👎", "realization": "😮",
        "nervousness": "😬", "approval": "👍", "joy": "😄", "anger": "😡",
        "embarrassment": "😳", "caring": "🤗", "remorse": "😔",
        "disgust": "🤢", "grief": "😥", "confusion": "😕",
        "relief": "😌", "desire": "😍", "admiration": "😌",
        "optimism": "😊", "fear": "😨", "love": "❤️",
        "excitement": "🎉", "curiosity": "🤔", "amusement": "😄",
        "surprise": "😲", "gratitude": "🙏", "pride": "🦁"
    }
    return emoji_mapping.get(sentiment, "")

def ollama_response(user_input):
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",  # Replace with your model name
        "prompt": user_input,
        "stream": True
    }
    full_response = ""

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_response += chunk['response']
        return full_response.strip()
    except Exception as e:
        st.error(f"Ollama streaming error: {str(e)}")
        try:
            response = requests.post(OLLAMA_API_URL, json={**payload, "stream": False})
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Error communicating with Ollama API"
        except Exception as e:
            return f"Fallback failed: {str(e)}"

# Voice Analysis Tab - Updated Flow
with tab1:
    st.header("🎤 Voice/Text to LLM with Translation")
    
    # Input mode selection
    input_mode = st.radio("Choose your input mode:", ["🎙️ Voice Input", "⌨️ Text Input"])

    original_text = ""
    source_lang = "en"

    if input_mode == "🎙️ Voice Input":
        audio = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=True,
            key='voice_recorder'
        )

        if audio:
            try:
                os.makedirs("recordings", exist_ok=True)
                file_path = f"recordings/recording_{uuid.uuid4().hex}.wav"

                st.audio(audio["bytes"], format="audio/wav")
                with open(file_path, "wb") as f:
                    f.write(audio["bytes"])

                if st.button("🔍 Process Voice"):
                    with st.spinner("Transcribing and Translating..."):
                        results = Sarvam_STT.detect_and_translate(file_path)
                        original_text = results["transcript"]
                        source_lang = results["language_code"]
                        st.success(f"Transcript: {original_text}")
            except Exception as e:
                st.error(f"Voice Processing Error: {str(e)}")

    elif input_mode == "⌨️ Text Input":
        original_text = st.text_area("Enter your message:")
        if original_text.strip():
            source_lang = Google_Translate.detect(original_text)

    # Proceed if text is available
    if original_text.strip() and st.button("🚀 Send to LLM"):
        try:
            # Translate to English if not already
            english_text = Google_Translate.detect_and_translate(original_text) if source_lang != "en" else original_text

            st.info(f"Text sent to LLM (English): {english_text}")

            # Get response from LLM
            with st.spinner("Getting response from Ollama..."):
                response_english = ollama_response(english_text)

            # Translate response back to original language if needed
            if source_lang != "en":
                response_translated = Google_Translate.translate_text(response_english, target_lang=source_lang)
                st.markdown(f"### 🗣️ Response in Original Language ({source_lang}):\n{response_translated}")
            else:
                st.markdown(f"### 🗣️ Response:\n{response_english}")

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

# Text Chat with Ollama Tab
with tab2:
    st.header("💬 Chat with Ollama LLM (Text Input)")
    user_prompt = st.text_input("Enter your question to Ollama LLM")
    if st.button("Ask"):
        if user_prompt.strip() != "":
            with st.spinner("Generating response..."):
                response_text = ollama_response(user_prompt)
                st.markdown(response_text)
        else:
            st.warning("Please enter a valid question!")
