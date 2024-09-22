import requests
import streamlit as st
from groq import Groq
import os


API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/models"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

groq_client = Groq(api_key=API_KEY)

st.title("Whisper Audio Transcription App")

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])


def save_audio_file(uploaded_file):
    filename = os.path.join(os.path.dirname(__file__), "temp_audio.m4a")
    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filename


def remove_temp_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)


@st.cache_resource
def load_groq_model():
    try:
        response = requests.get(GROQ_API_URL, headers=HEADERS)
        response.raise_for_status()
        models = response.json()
        st.sidebar.success("Groq models loaded successfully!")
        return models
    except requests.RequestException as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None


def transcribe_audio(filepath, model="whisper-large-v3", language="fr", temperature=0.0):
    try:
        with open(filepath, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(filepath, f.read()),
                model=model,
                prompt="Specify context or spelling",
                response_format="json",
                language=language,
                temperature=temperature
            )
        return transcription.text
    except Exception as e:
        st.sidebar.error(f"Error during transcription: {e}")
        return None


def main():
    if st.sidebar.button("Load Whisper Model"):
        models = load_groq_model()
        if models:
            st.sidebar.success("Whisper Model Loaded")

    if st.sidebar.button("Transcribe Audio"):
        if audio_file is not None:
            st.sidebar.success("Processing Audio...")

            audio_path = save_audio_file(audio_file)
            transcription = transcribe_audio(audio_path)

            if transcription:
                st.text("Transcription Complete:")
                st.write(transcription)
                remove_temp_file(audio_path)
        else:
            st.sidebar.error("Please upload an audio file")


if __name__ == "__main__":
    main()
