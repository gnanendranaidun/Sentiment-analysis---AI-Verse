def detect_and_translate(file_path):
    import requests

    url = "https://api.sarvam.ai/speech-to-text"

    headers = {
        "api-subscription-key": "96e9ce28-c143-4dbd-aa30-704d9bc41de4"
    }

    payload = {
        "with_timestamps": "false",
        "with_diarization": "false",
        "model": "saarika:v2",
        "language_code": "unknown"
    }

    # Open the audio file in binary mode
    files = {
        "file": ("audio.mp3", open(file_path, "rb"), "audio/mpeg")
    }

    response = requests.post(url, headers=headers, data=payload, files=files)
    return response.json()


