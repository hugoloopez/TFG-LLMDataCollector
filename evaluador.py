import random
import time
import os
import nltk
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from utils import functions as F
from app import AVAILABLE_MODELS
from utils.logger import iniciar_conversacion, log_conversation, finalizar_conversacion, calcular_coste_conversacion
from utils.verification import verificar_registro_completo
from utils.metricas_logger import iniciar_metricas_conversacion, log_metricas_turno, finalizar_metricas_conversacion
from langchain_core.runnables.config import RunnableConfig
load_dotenv()

config = RunnableConfig(
    recursion_limit=50,
    configurable={"thread_id": random.randint(1, 999999)},
    max_concurrency=1
)

db_uri = os.getenv("DB_CONNECTION_URL")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
PROMPT = F.read_prompt("system_message")


def generar_usuarios(n=3):
    nombres = ["Lucía", "Carlos", "Eva", "Hugo", "Almudena", "Javier", "Chema", "Maria", "Antonio", "Cristina", "Pablo", "Laura", "Sara", "David", "Raúl"]
    apellidos = ["Pérez", "Ramírez", "Martín", "López", "Juan", "Peiro", "Fernandez", "Alonso", "García", "Sánchez", "Martínez", "Gómez", "Díaz", "Moreno", "Torres", "Vázquez"]
    dominios = ["gmail.com", "hotmail.com"]
    usuarios = []
    for _ in range(n):
        nombre = random.choice(nombres)
        apellido = f"{random.choice(apellidos)} {random.choice(apellidos)}"
        email = f"{nombre.lower()}.{apellido.split()[0].lower()}{random.randint(1,99)}@{random.choice(dominios)}"
        dni = f"{random.randint(10000000, 99999999)}{random.choice('TRWAGMYFPDXBNJZSQVHLCKE')}"
        direccion = f"Calle {random.choice(apellidos)} {random.randint(1, 50)}, Valencia, España, {random.randint(28000, 28999)}"
        telefono = f"6{random.randint(10000000, 99999999)}"
        tarjeta = f"{random.randint(4000,4999)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"
        expiracion = f"{random.randint(1,12):02d}/{random.randint(24,28)}"
        cvv = f"{random.randint(100,999)}"
        usuarios.append({
            "nombre": nombre,
            "apellidos": apellido,
            "dni": dni,
            "email": email,
            "telefono": telefono,
            "direccion": direccion,
            "tarjeta": tarjeta,
            "expiracion": expiracion,
            "cvv": cvv,
            "titular_tarjeta": f"{nombre} {apellido}"
        })
    return usuarios


def generar_mensajes(usuario):
    return [
        "Hola, quiero registrarme!",
        "¿Qué planes hay disponibles?",
        "Quiero registrarme al plan standard",
        f"{usuario['nombre']} {usuario['apellidos']}",
        usuario["dni"],
        usuario["email"],
        usuario["telefono"],
        usuario["direccion"],
        f"{usuario['tarjeta']} {usuario['expiracion']} {usuario['cvv']}",
        usuario["titular_tarjeta"],
        "es correcto gracias"
    ]


def generar_respuestas_esperadas(usuario):
    nombre = usuario["nombre"]
    apellidos = usuario["apellidos"]
    nombre_completo = f"{nombre} {apellidos}"
    dni = usuario["dni"]
    email = usuario["email"]
    telefono = usuario["telefono"]
    direccion = usuario["direccion"]
    titular = usuario["titular_tarjeta"]

    return [
        " ¡Hola! Soy tu asistente FitnessLife y me alegra darte la bienvenida al equipo. ¡Este es tu primer paso hacia una vida más saludable y activa! Estamos emocionados de acompañarte en este camino lleno de bienestar, energía y motivación.",
        "Perfecto, estos son nuestros planes:\n1. Plan Standard: $29.99\n2. Plan Pro: $49.99\n¿Cuál prefieres?",
        "Excelente, comenzaremos con tu registro en el plan Standard. ¿Cuál es tu nombre completo?",
        f"Gracias, {nombre}. ¿Me puedes dar tu DNI, por favor?",
        "Perfecto. Ahora necesito tu dirección de correo electrónico.",
        "Gracias. ¿Cuál es tu número de teléfono?",
        "¿Podrías confirmarme tu dirección completa? Incluye calle, número, ciudad, país y código postal.",
        "Ahora necesito tus datos de tarjeta: número, expiración y CVV.",
        "¿A nombre de quién está la tarjeta?",
        f"¡Perfecto! Ya tenemos toda la información. Tu registro ha sido completado. Plan: Standard. Nombre: {nombre_completo}. Email: {email}. Teléfono: {telefono}. Dirección: {direccion}. DNI: {dni}. Titular: {titular}. ¿Hay algo más en lo que pueda ayudarte?",
        "Ha sido un placer ayudarte. ¡Disfruta de tu membresía y tu camino hacia una vida más saludable!"
    ]


def calcular_metricas(ref, pred):
    ref_tokens = nltk.word_tokenize(ref.lower())
    pred_tokens = nltk.word_tokenize(pred.lower())
    emb_ref = embedding_model.encode([ref])
    emb_pred = embedding_model.encode([pred])
    cosine = float((emb_ref @ emb_pred.T)[0][0])
    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)
    meteor = meteor_score([ref_tokens], pred_tokens)
    rouge_l = rouge.score(ref, pred)['rougeL'].fmeasure
    return cosine, bleu, meteor, rouge_l


def transformar_usuario_a_expected_data(usuario):
    direccion_parts = usuario["direccion"].split(", ")
    calle_y_numero = direccion_parts[0].replace("Calle ", "")
    calle_parts = calle_y_numero.split(" ")
    calle = " ".join(calle_parts[:-1])
    numero = calle_parts[-1]
    ciudad = direccion_parts[1]
    pais = direccion_parts[2]
    codigo_postal = direccion_parts[3]

    return {
        "cliente": {
            "nombre": usuario["nombre"],
            "apellidos": usuario["apellidos"],
            "correo": usuario["email"],
            "telefono": usuario["telefono"],
            "dni": usuario["dni"]
        },
        "direccion": {
            "calle": calle,
            "numero": numero,
            "ciudad": ciudad,
            "pais": pais,
            "codigo_postal": codigo_postal
        },
        "pago": {
            "numero_tarjeta": usuario["tarjeta"].replace(" ", ""),
            "titular_tarjeta": usuario["titular_tarjeta"],
            "fecha_expiracion": usuario["expiracion"],
            "cvv": usuario["cvv"]
        }
    }


def simular_conversaciones():
    for modelo, llm in list(AVAILABLE_MODELS.items())[:]:
        usuarios = generar_usuarios(5)
        tools = F.obtener_tools(db_uri, llm)
        agent_executor = create_react_agent(llm, prompt=PROMPT, tools=tools)
        for usuario in usuarios:
            mensajes = generar_mensajes(usuario)
            respuestas_esperadas = generar_respuestas_esperadas(usuario)
            dni = usuario["dni"]
            iniciar_conversacion(modelo)
            iniciar_metricas_conversacion(modelo, dni)
            messages = [{"role": "system", "content": PROMPT}]
            input_tokens_totales = 0
            output_tokens_totales = 0
            for i, user_msg in enumerate(mensajes):
                try:
                    messages.append({"role": "user", "content": user_msg})
                    t0 = time.time()
                    response = ""
                    tools_usadas = []

                    for paso in agent_executor.stream({"messages": messages}, config=config):
                        if "agent" in paso:
                            r = paso["agent"]["messages"][0]
                            if isinstance(r.content, list):
                                response += ''.join(str(c) for c in r.content)
                            else:
                                response += str(r.content)
                            if hasattr(r, "usage_metadata") and r.usage_metadata:
                                input_tokens_totales += r.usage_metadata.get("input_tokens", 0)
                                output_tokens_totales += r.usage_metadata.get("output_tokens", 0)

                        if isinstance(paso, dict) and 'tools' in paso:
                            tools_response = paso['tools'].get("messages", [])
                            if tools_response:
                                tools_usadas.extend([tool.name for tool in tools_response if hasattr(tool, "name")])

                    duracion = time.time() - t0
                    ref = respuestas_esperadas[i] if i < len(respuestas_esperadas) else ""
                    log_conversation(modelo, PROMPT, user_msg, response, duracion, tools_used=tools_usadas, state="En progreso", dni=dni)
                    cosine, bleu, meteor, rouge = calcular_metricas(ref, response)
                    log_metricas_turno(user_msg, response, ref, cosine, bleu, meteor, rouge, i+1)
                    messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    print(f"Error durante el turno {i+1} con el modelo {modelo}: {e}")
                    log_conversation(modelo, PROMPT, user_msg, "[ERROR DURANTE RESPUESTA]", 0, tools_used=[], state="Error", dni=dni)
                    continue

            expected_data = transformar_usuario_a_expected_data(usuario)
            registro_ok, detalle = verificar_registro_completo(usuario["email"], expected_data, db_uri)
            campos_faltantes = []
            if not registro_ok and "incompleto" in detalle.lower():
                campos_faltantes = detalle.replace("Registro incompleto: ", "").split(", ")
            estado_final = "Correcto" if registro_ok else "Fallido"
            coste_conversacion = calcular_coste_conversacion(input_tokens_totales, output_tokens_totales, modelo)
            log_conversation(modelo, PROMPT, "N/A", "N/A", 0, state=estado_final, cost=coste_conversacion, dni=dni)
            finalizar_conversacion()
            finalizar_metricas_conversacion(
                turnos_totales=len(mensajes),
                registro_ok=registro_ok,
                detalle_fallo=detalle,
                campos_faltantes=campos_faltantes
            )


simular_conversaciones()
