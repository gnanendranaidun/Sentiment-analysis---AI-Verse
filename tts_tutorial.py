def text_to_speech(text,language):
      
    import requests
    import base64
    import wave
    SARVAM_AI_API="96e9ce28-c143-4dbd-aa30-704d9bc41de4"
    d = {
        "hi":"hi-IN", 
        "bn":"bn-IN", 
        "kn":"kn-IN", 
        "ml":"ml-IN", 
        "mr":"mr-IN", 
        "od":"od-IN", 
        "pa":"pa-IN", 
        "ta":"ta-IN", 
        "te":"te-IN", 
        "en":"en-IN", 
        "gu":"gu-IN"
    }


    lang = d[language]
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_AI_API  # Replace with your valid API key
    }

    """### **Text to be converted into speech**"""

    # Split the text into chunks of 500 characters or less
    chunk_size = 500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Print the number of chunks
    print(f"Total chunks: {len(chunks)}")

    """## **Process Each Chunk**
    Iterate over each chunk, send it to the Sarvam AI API, and save the resulting audio as a `.wav` file.
    """

    # Iterate over each chunk and make the API call
    for i, chunk in enumerate(chunks):
        # Prepare the payload for the API request
        payload = {
            "inputs": [chunk],
            "target_language_code": lang,  # Target language code (Kannada in this case)
            "speaker": "neel",  # Speaker voice
            "model": "bulbul:v1",  # Model to use
            "pitch": 0,  # Pitch adjustment
            "pace": 1.0,  # Speed of speech
            "loudness": 1.0,  # Volume adjustment
            "enable_preprocessing": True,  # Enable text preprocessing
        }

        # Make the API request
        response = requests.post(url, json=payload, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Decode the base64-encoded audio data
            audio = response.json()["audios"][0]
            audio = base64.b64decode(audio)

            # Save the audio as a .wav file
            with wave.open(f"output{i}.wav", "wb") as wav_file:
                # Set the parameters for the .wav file
                wav_file.setnchannels(1)  # Mono audio
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(22050)  # Sample rate of 22050 Hz

                # Write the audio data to the file
                wav_file.writeframes(audio)

            print(f"Audio file {i} saved successfully as 'output{i}.wav'!")
        else:
            # Handle errors
            print(f"Error for chunk {i}: {response.status_code}")
            print(response.json())
        return f"output{i}.wav"

"""## **Output**

After running the notebook, you will have multiple `.wav` files (e.g., `output1.wav`, `output2.wav`, etc.) containing the speech for each chunk of text.

## **Conclusion**
This notebook provides a step-by-step guide to converting text into speech using the Sarvam AI API. You can modify the text, language, and other parameters to suit your specific needs.


### **Additional Resources**

For more details, refer to the our official documentation and we are always there to support and help you on our Discord Server:

- **Documentation**: [docs.sarvam.ai](https://docs.sarvam.ai)  
- **Community**: [Join the Discord Community](https://discord.gg/hTuVuPNF)

---

### **9. Final Notes**

- Keep your API key secure.
- Use clear audio for best results.

**Keep Building!** 🚀
"""