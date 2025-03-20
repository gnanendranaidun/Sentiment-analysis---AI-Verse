
# 🎤 Multilingual ASR with Sentiment Analysis 💬

This Streamlit app allows you to:
- Record audio directly from your browser 🎙️
- Transcribe the audio using OpenAI's Whisper model 📝
- Perform sentiment analysis using the `go_emotions` model by Hugging Face 🤖
- Display emotions with intuitive emojis 😄😢😠

## 🚀 Features
✅ **Audio Recording:** Record audio within the app using `streamlit-mic-recorder`  
✅ **Multilingual ASR:** Uses Whisper's `base` model for speech-to-text in multiple languages  
✅ **Sentiment Analysis:** Leverages `SamLowe/roberta-base-go_emotions` for extracting emotions from transcriptions  
✅ **Emoji Display:** Maps each emotion to its relevant emoji for easy visualization  
✅ **Save Recordings:** Audio is saved locally in the `recordings` directory  

## 🛠 Tech Stack
- [Streamlit](https://streamlit.io/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [streamlit-mic-recorder](https://github.com/stefanrmmr/streamlit-mic-recorder)

## 📂 Project Structure
```
📁 recordings/        # Stores all recorded audio files
📄 app.py             # Main Streamlit application
📄 README.md          # Project Documentation
```

## 💻 How to Run Locally
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gnanendranaidun/Sentiment-analysis---AI-Verse.git
   cd Sentiment-analysis---AI-Verse
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## 📜 Example Usage
1. Click **Start Recording** to record your voice.
2. **Stop Recording** when done.
3. Click **Get Sentiments** to:
   - Transcribe your speech
   - Analyze emotions
   - Display sentiments with emojis

## 🎯 Example Sentiment Output:
```
joy 😄: 0.85
gratitude 🙏: 0.65
admiration 😌: 0.60
```

## ✅ To-Do / Improvements
- Add language selection for Whisper model
- Enable sentiment trend visualization (charts/graphs)
- Deploy to Streamlit Cloud or Hugging Face Spaces

## 🤖 Model References
- **Whisper:** [OpenAI Whisper](https://github.com/openai/whisper)
- **Sentiment Model:** [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)


