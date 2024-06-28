# pip install streamlit
# pip install audio-recorder-streamlit
# pip install openai

import streamlit as st
from audio_recorder_streamlit import audio_recorder


from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

#to avoid this error: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def transcribe_text_to_voice(audio_location):
    # client = OpenAI(api_key=API_KEY)
    audio_file= open(audio_location, "rb")
    segments, info = model.transcribe(audio_file, beam_size=5,language="en")
    transcribed_text=""
    for segment in segments:
        transcribed_text+=segment.text

    return transcribed_text






st.title("Tally AI Assistant")
st.write("Hi how I can help you today?")

if model:
    audio_bytes = audio_recorder()

    if audio_bytes:
        ##Save the Recorded File
        audio_location = "audio_file.wav"
        with open(audio_location, "wb") as f:
            f.write(audio_bytes)

        #Transcribe the saved file to text
        text = transcribe_text_to_voice(audio_location)
        st.write(text)
else:
    st.write("Pls wait for model to get loaded")

    # #Use API to get an AI response
    # api_response = chat_completion_call(text)
    # st.write(api_response)

    # # Read out the text response using tts
    # speech_file_path = 'audio_response.mp3'
    # text_to_speech_ai(speech_file_path, api_response)
    # st.audio(speech_file_path)