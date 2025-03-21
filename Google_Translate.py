from langdetect import detect
from deep_translator import GoogleTranslator

def detect_and_translate(text):
    # Detect language
    detected_lang = detect(text)
    # print(f"Detected Language: {detected_lang}")
    # Translate to English
    translated_text = GoogleTranslator(source=detected_lang,target="en").translate(text)
    # print(f"Translated Text: {translated_text}")
    
    return translated_text


