from langdetect import detect
from deep_translator import GoogleTranslator

def detect_and_translate(text):
    """
    Detects the language of the input text and translates it to English.
    Returns the translated English text.
    """
    detected_lang = detect(text)
    translated_text = GoogleTranslator(source=detected_lang, target="en").translate(text)
    return translated_text

def change_to_target(text, target_lang):
    """
    Detects the language of the input text and translates it to the target language.
    Returns the translated text and the detected source language.
    """
    detected_lang = detect(text)
    translated_text = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
    return translated_text, detected_lang
