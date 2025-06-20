import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import functions as F
from utils import logger
import random
import tempfile
import time
import re

load_dotenv()
db_uri = os.getenv("DB_CONNECTION_URL")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DEEP_SEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

AVAILABLE_MODELS = {
    "OpenAI GPT-4.1": ChatOpenAI(model_name="gpt-4.1-2025-04-14", openai_api_key=OPENAI_API_KEY, temperature=0.4),
    "OpenAI GPT-4o": ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.4),
    "OpenAI GPT-4 Turbo": ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.4),
    "OpenAI GPT-3.5": ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.4),
    "Claude-3.5": ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_API_KEY, temperature=0.4),
    "Claude-3 Opus": ChatAnthropic(model="claude-3-opus-20240229", api_key=ANTHROPIC_API_KEY, temperature=0.4),
    "Claude-3 Haiku": ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPIC_API_KEY, temperature=0.4),
    "Claude 3.7 Sonnet (20250219)": ChatAnthropic(model="claude-3-7-sonnet-20250219", api_key=ANTHROPIC_API_KEY, temperature=0.4),
    "Deepseek-V3": ChatDeepSeek(model="deepseek-chat", api_key=DEEP_SEEK_API_KEY, api_base='https://api.deepseek.com', temperature=0.4),
    "Cohere Command R+": ChatCohere(model="command-r-plus", cohere_api_key=COHERE_API_KEY, temperature=0.4),
    "Gemini 2.5 Pro": ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY, temperature=0.4),
    "Gemini 2.0 Flash": ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.4),
    "Gemini 2.0 Flash-Lite": ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GOOGLE_API_KEY, temperature=0.4),
}

st.title("Plataforma de usuario FitnessLife")

USER_AVATAR = "👤"
BOT_AVATAR = "src/images/bot_logo.jpeg"

@st.cache_resource
def get_cached_agent_executor(model_name: str, db_uri: str):
    llm = AVAILABLE_MODELS[model_name]
    tools = F.obtener_tools(db_uri, llm)
    system_message = F.read_prompt("system_message")
    return create_react_agent(llm, prompt=system_message, tools=tools, checkpointer=None)

# if "whisper_model" not in st.session_state:
#     @st.cache_resource
#     def load_model():
#         return F.load_or_get_whisper("src/whisper_model.pkl")
#     st.session_state.whisper_model = load_model()

if "conversation_start_time" not in st.session_state:
    st.session_state.update({
        "input_tokens": 0,
        "output_tokens": 0,
        "conversation_start_time": None,
        "response_times": [],
        "messages": [],
        "initialized_conversation": False,
        "model": None,
    })

with st.sidebar:
    st.header("Configuración del Modelo")
    selected_model = st.selectbox("Selecciona un modelo:", ["Seleccione un modelo..."] + list(AVAILABLE_MODELS.keys()), index=0)
    if selected_model != "Seleccione un modelo...":
        st.session_state["model"] = selected_model
        if not st.session_state["initialized_conversation"]:
            st.session_state["conversation_start_time"] = time.time()
            logger.iniciar_conversacion(selected_model)
            st.session_state["initialized_conversation"] = True
            st.session_state.agent_executor = get_cached_agent_executor(selected_model, db_uri)

    if st.button("Delete Chat History"):
        st.session_state.messages = []
        logger.finalizar_conversacion()

    if st.button("Finalizar y guardar tiempos en log") and st.session_state["model"]:
        if not st.session_state["response_times"]:
            st.warning("Aún no se han realizado preguntas.")
        else:
            total = time.time() - st.session_state["conversation_start_time"]
            promedio = sum(st.session_state["response_times"]) / len(st.session_state["response_times"])
            coste = F.calcular_coste_conversacion(
                st.session_state["input_tokens"],
                st.session_state["output_tokens"],
                st.session_state["model"]
            )
            logger.log_conversation(st.session_state["model"], "Resumen final de tiempos", "N/A", "N/A", 0, "N/A", "Final", coste)
            st.success("⏱️ Tiempos guardados exitosamente en log.")
            st.write(f"**Tiempo total:** {total:.2f}s")
            st.write(f"**Tiempo promedio:** {promedio:.2f}s")
            st.write(f"**Coste:** {coste} USD")
            logger.finalizar_conversacion()

    if st.button("🎤 Grabar audio"):
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        F.grabar_audio(temp_path)
        with st.spinner("🔊 Procesando audio..."):
            try:
                result = st.session_state.whisper_model.transcribe(temp_path)
                st.session_state["audio_input"] = result["text"]
                st.rerun()
            except Exception as e:
                st.error(f"Error al procesar el audio: {e}")
            os.remove(temp_path)

if st.session_state["model"] is None:
    st.warning("⚠️ Por favor, selecciona un modelo en la barra lateral para comenzar la conversación..")
    st.stop()

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

user_input = st.chat_input("¿En qué puedo ayudarte?")
if "audio_input" in st.session_state:
    user_input = st.session_state.pop("audio_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    agent_executor = st.session_state.agent_executor
    config = {"configurable": {"thread_id": random.randint(0, 1000000)}}

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        placeholder = st.empty()
        start_time = time.time()
        with st.spinner("Pensando..."):
            full_response = ""
            chat_tools_used = []
            for response in agent_executor.stream({"messages": st.session_state["messages"]}, config):
                if "agent" in response:
                    msg_list = response["agent"].get("messages", [])
                    if msg_list:
                        ai_message = msg_list[0]
                        if hasattr(ai_message, "content") and isinstance(ai_message.content, str):
                            content = re.sub(r"<thinking>.*?</thinking>", "", ai_message.content, flags=re.DOTALL)
                            content = re.sub(r"\([^)]*\)", "", content).strip()
                            full_response += content

                        if hasattr(ai_message, "usage_metadata"):
                            meta = ai_message.usage_metadata
                            if isinstance(meta, dict):
                                st.session_state["input_tokens"] += meta.get("input_tokens", 0)
                                st.session_state["output_tokens"] += meta.get("output_tokens", 0)

                if "tools" in response:
                    messages = response["tools"].get("messages", [])
                    if messages:
                        tool_message = messages[0]
                        if hasattr(tool_message, "name"):
                            chat_tools_used.append(tool_message.name)
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

        duracion = time.time() - start_time
        st.session_state["response_times"].append(duracion)
        logger.log_conversation(
            model_name=st.session_state["model"],
            prompt="[Prompt omitido para brevedad]",
            user_message=user_input,
            ai_response=full_response,
            response_time=duracion,
            tools_used=chat_tools_used
        )
        st.session_state.messages.append({"role": "assistant", "content": full_response})
