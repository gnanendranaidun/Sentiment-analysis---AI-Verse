# Multilingual AI Voice Assistant Bot

An AI-powered voice assistant that can understand and respond in multiple languages while maintaining a consistent character personality.

## Features
- Speech-to-text recognition
- Text-to-speech output
- Multilingual support
- Character-based responses
- Web interface for interaction

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Character Background
The bot is designed as a grumpy but knowledgeable librarian who reluctantly helps users find information. The character maintains a consistent personality while providing accurate responses in the user's preferred language.

## Technologies Used
- Python
- Flask
- SpeechRecognition
- pyttsx3
- Hugging Face Transformers (TinyLlama)
- Google Translate API

## System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- At least 8GB RAM
- Microphone for voice input
- Speakers for voice output 