import streamlit as st
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
import uuid
import wave
import scipy.io.wavfile as wavfile
from datetime import datetime
import base64
import tempfile
import time
import threading
import moviepy
import whisper
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model
# Set page config
st.set_page_config(
    page_title="Mindful AI - Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations and beautiful styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        background: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        color: #2c3e50;
    }
    .result-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 20px;
        padding: 10px 20px;
        margin: 0 5px;
        transition: all 0.3s ease;
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #FF6B6B;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6B6B !important;
        color: white !important;
    }
    .camera-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        background: white;
        padding: 1rem;
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        color: #2c3e50;
    }
    .header-container {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
        color: #2c3e50;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .emotion-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(255, 107, 107, 0.9);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    p, h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)
# Initialize session state variables
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None
if 'camera_thread' not in st.session_state:
    st.session_state.camera_thread = None

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to annotate emotions on the frame
def annotate_emotions(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Create a Pillow image for drawing annotations
    annotated_image = frame.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
        
        # Extract the face region
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        
        # Predict emotion
        prediction = model.predict(face)[0]
        emotion = emotion_labels[np.argmax(prediction)]
        
        # Annotate the emotion label
        draw.text((x, y - 15), emotion, fill="red")
    
    return annotated_image

# Camera loop function
def camera_loop():
    try:
        while st.session_state.camera_running:
            # Capture frame from webcam
            frame = st.camera_input("Camera Feed", key="live_camera")
            
            if frame is not None:
                # Convert the frame to a PIL image
                frame = Image.open(frame)
                
                # Annotate emotions on the frame
                annotated_frame = annotate_emotions(frame)
                
                # Display the annotated frame in the placeholder
                if st.session_state.frame_placeholder is not None:
                    st.session_state.frame_placeholder.image(annotated_frame, caption="Live Feed with Emotion Detection", use_container_width=True)
            
            time.sleep(0.1)
    finally:
        pass

# Initialize models
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Load face cascade first as it's essential
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        classifier = tf.keras.models.load_model('models/ResNet50_Transfer_Learning.keras')

        if face_cascade.empty():
            st.error("Error loading face detection model")
            return None, None

        # Load sentiment analysis model
        try:
            model = whisper.load_model("base")
            sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")
        except Exception as e:
            st.error(f"Error loading sentiment analysis model: {str(e)}")
            return (None, None), (None, None)

        return (sentiment_analysis, model), (face_cascade, classifier)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return (None, None), (None, None)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load models at startup
with st.spinner("Loading models..."):
    a,b = load_models()

    (sentiment_analysis,model) = a
    (face_cascade,classifier) = b

if not face_cascade:
    st.error("Failed to load face detection model. Please check your OpenCV installation.")
    st.stop()
if not classifier:
    st.error("Failed to load classifier. Please check your OpenCV installation.")
    st.stop()

if not sentiment_analysis:
    st.error("Failed to load sentiment analysis model. Please check your internet connection and try again.")
    st.stop()
if not model:
    st.error("Failed to load whisper model. Please check your internet connection and try again.")
    st.stop()

# Header with gradient background
st.markdown("""
    <div class="header-container">
        <h1 style="color: white;">🧠 Mindful AI - Emotion Analyzer</h1>
        <p style="color: white;">Your personal emotional wellness companion</p>
    </div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="feature-card">
            <h3>🎤 Voice Analysis</h3>
            <p>Record and analyze your voice to understand your emotional state</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <h3>👤 Facial Detection</h3>
            <p>Real-time facial analysis to track your emotional expressions</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <h3>❤️ Heart Rate Monitor</h3>
            <p>Track your heart rate patterns for emotional insights</p>
        </div>
    """, unsafe_allow_html=True)

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["🎤 Voice Analysis", "👤 Facial Detection", "❤️ Heart Rate Monitor"])

# Voice Analysis Tab
with tab1:
    st.header("Voice Emotion Analysis")
    
    # Voice recording section with better styling
    st.markdown("""
        <div class="feature-card">
            <h3>Record Your Voice</h3>
            <p>Speak naturally and let AI analyze your emotions</p>
        </div>
    """, unsafe_allow_html=True)
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
    # Live recording
    try:
        audio = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=True,
            key='voice_recorder'
        )
        
        if audio:
            try:
                recordings_folder = "recordings"
                os.makedirs(recordings_folder, exist_ok=True)
                unique_filename = f"recording_{uuid.uuid4().hex}.wav"
                file_path = os.path.join(recordings_folder, unique_filename)
                
                # Save and play audio with better styling
                audio_bytes = audio["bytes"]
                st.markdown("""
                    <div class="metric-container">
                        <h4>🎵 Your Recording</h4>
                """, unsafe_allow_html=True)
                st.audio(audio_bytes, format="audio/wav")
                st.markdown("</div>", unsafe_allow_html=True)
                
                with open(file_path, "wb") as f:
                    f.write(audio_bytes)
        
                # Analyze emotion directly from audio features
                if st.button("🔍 Analyze Voice Emotion"):
                    with st.spinner("Analyzing your voice..."):
                        try:
                            # For now, we'll use a simulated emotion detection
                            # In a real implementation, you would use a voice emotion detection model
                            
                            st.markdown("### 📊 Analysis Results")
                            st.markdown("#### Transcribing Audio...")
                            results = model.transcribe(file_path)
                            st.markdown(f"## {results["text"]}")
                            print(results)
                            sentiment_output_value = inference(results["text"], "Sentiment + Score")
                            st.markdown("#### "+ sentiment_output_value)
                            
                            
                            # Sort emotions by confidenc
                            
                            # Add a note about the analysis
                            st.info("""
                                Note: This is a demonstration of voice emotion analysis. 
                                For more accurate results, consider using a dedicated voice emotion recognition model.
                            """)
                            
                        except Exception as e:
                            st.error(f"Error analyzing voice: {str(e)}")
                            st.info("Please try recording again with clearer speech.")
            except Exception as e:
                st.error(f"Error saving recording: {str(e)}")
    except Exception as e:
        st.error(f"Error with microphone recorder: {str(e)}")
        st.info("Please make sure your microphone is properly connected and accessible.")

# Facial Detection Tab
with tab2:
    st.header("Facial Detection")
    
    # Input selection
    input_type = st.radio(
        "Choose input type",
        ["📹 Live Camera", "📸 Upload Image", "🎥 Upload Video"],
        horizontal=True
    )
    if input_type == "📹 Live Camera":
        st.markdown("""
            <div class="feature-card">
                <h3>Live Camera Feed</h3>
                <p>Real-time facial detection and emotion analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Camera control buttons
        col1, col2 = st.columns(2)
        with col1:
            start_camera = st.button("📹 Start Camera")
        with col2:
            stop_camera = st.button("⏹️ Stop Camera")

        # Create a placeholder for the camera feed
        if st.session_state.frame_placeholder is None:
            st.session_state.frame_placeholder = st.empty()

        # Start camera feed
        if start_camera:
            st.session_state.camera_running = True
            
            # Start camera in a separate thread
            if st.session_state.camera_thread is None:
                st.session_state.camera_thread = threading.Thread(target=camera_loop)
                st.session_state.camera_thread.start()

        # Stop camera feed
        if stop_camera:
            st.session_state.camera_running = False
            if st.session_state.camera_thread is not None:
                st.session_state.camera_thread.join()
                st.session_state.camera_thread = None
            st.session_state.frame_placeholder.empty()

    elif input_type == "📸 Upload Image":
        st.markdown("""
            <div class="feature-card">
                <h3>Upload Image</h3>
                <p>Analyze emotions from a static image</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Image"):
                # Convert PIL image to OpenCV format
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    st.success(f"Found {len(faces)} face(s)!")
                    
                    for (x, y, w, h) in faces:
                        # Draw rectangle around face with gradient color
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 107, 107), 2)
                        
                        # Simulate emotion detection (replace with actual emotion detection model)
                        emotions = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]
                        emotion = np.random.choice(emotions)
                        
                        # Add emotion text
                        cv2.putText(frame, emotion, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 107, 107), 2)
                    
                    # Display the processed image
                    st.image(frame, channels="BGR", caption="Processed Image with Emotion Detection", use_container_width=True)
                else:
                    st.warning("No faces detected in the image!")
    
    else:  # Video upload
        st.markdown("""
            <div class="feature-card">
                <h3>Upload Video</h3>
                <p>Analyze emotions from a video file</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])
        if uploaded_video:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.getvalue())
                video_path = tmp_file.name
            
            # Display video
            st.video(uploaded_video)
            
            if st.button("🔍 Analyze Video"):
                with st.spinner("Processing video..."):
                    # Process video frames
                    cap = cv2.VideoCapture(video_path)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Convert to grayscale for face detection
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Detect faces
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        for (x, y, w, h) in faces:
                            # Draw rectangle around face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 107, 107), 2)
                            
                            # Simulate emotion detection
                            emotions = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]
                            emotion = np.random.choice(emotions)
                            
                            # Add emotion text
                            cv2.putText(frame, emotion, (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 107, 107), 2)
                        
                        # Display processed frame
                        st.image(frame, channels="BGR", caption="Processed Video Frame", use_container_width=True)
                    
                    cap.release()

# Heart Rate Monitor Tab
with tab3:
    st.header("Heart Rate Monitor")
    
    # Heart rate monitoring with better styling
    st.markdown("""
        <div class="feature-card">
            <h3>Real-time Heart Rate Monitoring</h3>
            <p>Track your heart rate patterns for emotional insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader for heart rate data
    uploaded_hr = st.file_uploader("Upload heart rate data (CSV)", type=["csv"])
    
    if uploaded_hr:
        # Read heart rate data
        hr_data = pd.read_csv(uploaded_hr)
        
        # Create interactive plot with better styling
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hr_data.index,
            y=hr_data['heart_rate'],
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(color='#FF8E53', size=8)
        ))
        fig.update_layout(
            title='Heart Rate Analysis',
            xaxis_title='Time',
            yaxis_title='Heart Rate (BPM)',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyze heart rate patterns
        avg_hr = hr_data['heart_rate'].mean()
        max_hr = hr_data['heart_rate'].max()
        min_hr = hr_data['heart_rate'].min()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Heart Rate", f"{avg_hr:.1f} BPM")
        with col2:
            st.metric("Maximum Heart Rate", f"{max_hr:.1f} BPM")
        with col3:
            st.metric("Minimum Heart Rate", f"{min_hr:.1f} BPM")
        
        # Heart rate status
        if avg_hr < 60:
            st.warning("⚠️ Average heart rate is below normal range")
        elif avg_hr > 100:
            st.warning("⚠️ Average heart rate is above normal range")
        else:
            st.success("✅ Average heart rate is within normal range")
    
    else:
        if st.button("❤️ Start Live Heart Rate Monitoring"):
            # Generate sample data
            time = np.linspace(0, 10, 100)
            heart_rate = 70 + 5 * np.sin(time) + np.random.normal(0, 1, 100)
            
            # Create interactive plot with better styling
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time,
                y=heart_rate,
                mode='lines+markers',
                name='Heart Rate',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(color='#FF8E53', size=8)
            ))
            fig.update_layout(
                title='Real-time Heart Rate',
                xaxis_title='Time (seconds)',
                yaxis_title='Heart Rate (BPM)',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current heart rate with better styling
            current_hr = heart_rate[-1]
            st.markdown(f"""
                <div class="metric-container">
                    <h3>Current Heart Rate</h3>
                    <h2 style="color: #FF6B6B;">{current_hr:.1f} BPM</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Heart rate status with better styling
            if current_hr < 60:
                st.warning("⚠️ Heart rate is below normal range")
            elif current_hr > 100:
                st.warning("⚠️ Heart rate is above normal range")
            else:
                st.success("✅ Heart rate is within normal range")

# Footer with better styling
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(45deg, #FF6B6B, #FF8E53); border-radius: 20px; color: white;'>
        <p style='font-size: 1.2rem;'>Built with ❤️ using Streamlit</p>
        <p style='font-size: 0.9rem;'>© 2024 Mindful AI - Emotion Analyzer</p>
    </div>
""", unsafe_allow_html=True)

