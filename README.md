# Mindful AI - Emotion Analyzer

Mindful AI is a comprehensive emotional wellness companion that uses advanced AI models to analyze emotions through multiple modalities: voice, facial expressions, and heart rate data. The app also features an AI chat assistant and voice chat capabilities for personalized emotional insights.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Mindful AI leverages several state-of-the-art models and libraries to help users understand their emotional state. Using **Streamlit** as the frontend, it provides an interactive user interface that includes:

- **Voice Analysis:** Record your voice, transcribe audio with Whisper, and analyze sentiment with a transformer-based sentiment model.
- **Facial Detection:** Detect faces and annotate emotions using OpenCV and a pre-trained emotion classifier.
- **Heart Rate Monitoring:** Upload or stream heart rate data to visualize and analyze heart rate patterns.
- **AI Chat:** Chat with an AI assistant for personalized insights and emotional guidance.
- **Voice Chat:** Convert your voice into text, process it with an AI model, and listen to the AI-generated response using text-to-speech.

---

## Features

### Voice Analysis
- **Record Audio:** Capture your voice using a microphone.
- **Transcription & Emotion Detection:** Transcribe audio and detect voice emotion.
- **Sentiment Visualization:** Display sentiment analysis with corresponding emojis and scores.

### Facial Detection
- **Input Options:** Use live camera feed or uploaded images/videos.
- **Emotion Annotation:** Detect faces and annotate them with predicted emotions.

### Heart Rate Monitor
- **Data Options:** Upload CSV data or perform live monitoring.
- **Interactive Visuals:** Visualize heart rate data with interactive plots.
- **Metrics:** Display key metrics such as average, maximum, and minimum heart rate.

### AI Chat
- **Conversational Interface:** Chat with an AI assistant that streams responses.
- **History Maintenance:** Maintain a conversation history for context.

### Voice Chat
- **Voice Messaging:** Record and send voice messages.
- **Advanced Processing:** Process voice input through speech-to-text, translation, AI modeling, and text-to-speech for dynamic responses.

---

## Architecture

The project is built using the following libraries and frameworks:

- **Streamlit:** For creating the web application interface.
- **Transformers:** To load and run transformer models for sentiment analysis.
- **Whisper:** For automatic speech recognition.
- **TensorFlow & Keras:** For loading and running the emotion classification model.
- **OpenCV:** For image and video processing, including face detection.
- **Plotly:** To create interactive charts for heart rate monitoring.
- **Other Dependencies:** Including libraries such as `pandas`, `numpy`, `uuid`, `tempfile`, and custom modules like `Sarvam_STT`, `Google_Translate`, and `tts_tutorial`.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip (Python package manager)

### Clone the Repository
```bash
git clone https://github.com/yourusername/mindful-ai-emotion-analyzer.git
cd mindful-ai-emotion-analyzer
