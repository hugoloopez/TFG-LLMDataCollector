from datetime import datetime
import os
import time
import streamlit as st

response_times = []
prompt_logged = False
current_log_filepath = None
conversation_start_time = time.time()

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
    precios = MODEL_PRICES.get(model_name, {"input": 0, "output": 0})
    coste_total = (input_tokens * precios["input"]) + (output_tokens * precios["output"])
    return round(coste_total, 6)


def iniciar_conversacion(model_name):
    global current_log_filepath, prompt_logged, response_times, conversation_start_time
    fecha = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("log", fecha, model_name.replace(" ", "_"))
    os.makedirs(log_dir, exist_ok=True)
    current_log_filepath = os.path.join(log_dir, f"conversacion.log")
    prompt_logged = False
    response_times = []
    conversation_start_time = time.time()

    mensaje_inicial = (
                        "¡Hola! Soy tu asistente FitnessLife y me alegra darte la bienvenida al equipo. "
                        "¡Este es tu primer paso hacia una vida más saludable y activa! Estamos emocionados "
                        "de acompañarte en este camino lleno de bienestar, energía y motivación."
    )
    if "messages" in st.session_state:
        st.session_state.messages.append({
            "role": "assistant",
            "content": mensaje_inicial
        })

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_entry = (
        f"Timestamp: {timestamp}\n"
        f"Mensaje del Asistente:\n{mensaje_inicial}\n\n"
    )

    with open(current_log_filepath, 'a', encoding='utf-8') as log_file:
        log_file.write(log_entry + "\n" + ("-" * 80) + "\n")


def log_conversation(model_name, prompt, user_message, ai_response, response_time, reasoning=None, tools_used=None, state=None, cost=0, dni="sin_dni"):
    global response_times, prompt_logged, current_log_filepath

    if current_log_filepath is None:
        iniciar_conversacion(model_name)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    total_conversation_time = time.time() - conversation_start_time
    response_times.append(response_time)
    response_times_average = sum(response_times) / len(response_times)

    fecha = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("log", fecha, model_name.replace(" ", "_"))
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, f"{dni}.log")

    if state == 'Final' or state == "Correcto" or state == "Fallido":
        log_entry = (
            f"Timestamp: {timestamp}\n"
            f"Estado Final: {state}\n"
            f"Tiempo total de conversación: {total_conversation_time:.2f} segundos\n"
            f"Tiempo promedio por respuesta: {response_times_average:.2f} segundos\n"
            f"Coste de la conversación: {cost:.6f} USD\n"
        )
    else:
        log_entry = (
            f"Timestamp: {timestamp}\n"
            f"Modelo: {model_name}\n"
            + (f"Prompt:\n{prompt}\n\n" if not prompt_logged else "") +
            f"Mensaje del Usuario:\n{user_message}\n\n"
            f"Respuesta IA:\n{ai_response}\n\n"
            f"Tiempo de respuesta actual: {response_time:.2f} segundos\n"
            f"Tiempo total de conversación: {total_conversation_time:.2f} segundos\n"
            f"Herramientas usadas:\n{tools_used if tools_used else 'No proporcionado'}\n\n"
        )

    prompt_logged = True

    with open(filepath, 'a', encoding='utf-8') as log_file:
        log_file.write(log_entry + "\n" + ("-" * 80) + "\n")


def finalizar_conversacion():
    global conversation_start_time, response_times, prompt_logged, current_log_filepath
    conversation_start_time = None
    response_times = []
    prompt_logged = False
    current_log_filepath = None
