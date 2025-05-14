from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
import streamlit as st
import wave
import numpy as np
import time
import pyaudio
import os
import joblib
import whisper
from langchain_core.language_models.chat_models import BaseChatModel


def read_prompt(file_name):
    with open(f"src/prompts/{file_name}.txt", "r", encoding="utf-8") as file:
        return file.read().strip()


def obtener_tools(db_uri, llm):

    if not isinstance(llm, BaseChatModel):
        raise ValueError(f"El LLM pasado no es un modelo de chat válido: {type(llm).__name__}")

    db = SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()


def load_or_get_whisper(MODEL_PATH):
    if os.path.exists(MODEL_PATH):
        print("Cargando modelo desde archivo...")
        return joblib.load(MODEL_PATH)
    else:
        print("Cargando modelo en memoria por primera vez...")
        model = whisper.load_model("large", device="cpu")
        joblib.dump(model, MODEL_PATH)
        return model


def grabar_audio(output_filename, silencio_max=2, rate=16000 , chunk=512, threshold=500):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []
    silence_time = 0
    with st.spinner("🎤 Grabando... Habla ahora."):
        while True:
            data = stream.read(chunk)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            if volume < threshold:
                if silence_time == 0:
                    silence_time = time.time()
                if time.time() - silence_time >= silencio_max:
                    break
            else:
                silence_time = 0
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    return output_filename


MODEL_PRICES = {
    "OpenAI GPT-4.1": {"input": 0.000002, "output": 0.000008},
    "OpenAI GPT-4o": {"input": 0.0000025, "output": 0.00001},
    "OpenAI GPT-4 Turbo": {"input": 0.00001, "output": 0.00003},
    "OpenAI GPT-3.5": {"input": 0.0000005, "output": 0.0000015},


    "Claude-3.5": {"input": 0.000003, "output": 0.000015},
    "Claude-3 Opus": {"input": 0.000015, "output": 0.000075},
    "Claude-3 Haiku": {"input": 0.00000025, "output": 0.00000125},
    "Claude 3.7 Sonnet (20250219)": {"input": 0.000003, "output": 0.000015},

    "Ollama Mistral": {"input": 0, "output": 0},
    "Ollama SmolLM2": {"input": 0, "output": 0},
    "Ollama DeepSeek": {"input": 0, "output": 0},
    "Ollama TinyLlama": {"input": 0, "output": 0},
    "Ollama LLaMA3": {"input": 0, "output": 0},  
    "Ollama phi": {"input": 0, "output": 0},        

    "Deepseek-V3": {"input": 0.00000027, "output": 0.0000011},

    "Gemini 2.5 Pro": {"input": 0.00000125, "output": 0.00001},
    "Gemini 2.0 Flash": {"input": 0.0000001, "output": 0.0000004},
    "Gemini 2.0 Flash-Lite": {"input": 0.000000075, "output": 0.0000003},

    "Cohere Command R+": {"input": 0.0000025, "output": 0.00001},

    "Mistral-AI Mixtral": {"input": 0.0007, "output": 0.0007},
    "Mistral Large 2.1": {"input": 0.003, "output": 0.006},
    "Mistral Small 25.01": {"input": 0.00015, "output": 0.00015},
    "Mistral Medium": {"input": 0.0007, "output": 0.0007},
}


def calcular_coste_conversacion(input_tokens, output_tokens, model_name):
    precios = MODEL_PRICES.get(model_name)
    if not precios:
        raise ValueError(f"Modelo no soportado: {model_name}")

    coste = (input_tokens * precios["input"]) + (output_tokens * precios["output"])
    return coste
