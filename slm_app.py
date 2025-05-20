import os
import json
import re
import streamlit as st
from datetime import date
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema.agent import AgentFinish

load_dotenv()
DB_CONNECTION_URL = os.getenv("DB_CONNECTION_URL")
engine = create_engine(DB_CONNECTION_URL)
llm = OllamaLLM(model="mistral")


def extraer_validados_de_respuesta(respuesta):
    matches = re.findall(
        r"Action: validar_([a-z_]+)\s*Action Input: ['\"]?([^'\"]+)['\"]?\s*Observation: (✅ .*?)(?:\n|$)",
        respuesta,
        re.DOTALL
    )
    for campo_tool, valor, observacion in matches:
        campo = campo_tool.replace("validar_", "")
        if "✅" in observacion and campo in st.session_state.registro_datos:
            st.session_state.registro_datos[campo] = valor.strip()


class TolerantOutputParser(ReActSingleInputOutputParser):
    def parse(self, text):
        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=text)

        return super().parse(text)


def guardar_en_bd(datos):
    with engine.begin() as conn:
        cliente_sql = text("""
            INSERT INTO clientes (nombre, apellidos, dni, correo, telefono)
            VALUES (:nombre, :apellidos, :dni, :correo, :telefono)
            RETURNING cliente_id
        """)
        direccion_sql = text("""
            INSERT INTO direcciones (cliente_id, calle, numero, ciudad, pais, codigo_postal)
            VALUES (:cliente_id, :calle, :numero, :ciudad, :pais, :codigo_postal)
        """)
        pago_sql = text("""
            INSERT INTO metodos_pago (cliente_id, numero_tarjeta, titular_tarjeta, fecha_expiracion, cvv)
            VALUES (:cliente_id, :numero_tarjeta, :titular_tarjeta, :fecha_expiracion, :cvv)
        """)
        plan_sql = text("""
            INSERT INTO clientes_planes (cliente_id, plan_id, fecha_inicio, activo)
            VALUES (:cliente_id, :plan_id, :fecha_inicio, :activo)
        """)
        cliente_id = conn.execute(cliente_sql, datos["cliente"]).scalar()
        datos["direccion"]["cliente_id"] = cliente_id
        datos["pago"]["cliente_id"] = cliente_id
        datos["plan"]["cliente_id"] = cliente_id
        conn.execute(direccion_sql, datos["direccion"])
        conn.execute(pago_sql, datos["pago"])
        conn.execute(plan_sql, datos["plan"])


def obtener_planes():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT plan_id, nombre, descripcion FROM planes"))
        planes = result.fetchall()
        if not planes:
            return "⚠️ No hay planes disponibles."
        return "\n".join([f"ID: {row.plan_id} | Nombre: {row.nombre} | {row.descripcion}" for row in planes])


@tool
def validar_dni(dni: str) -> str:
    """Valida que el DNI tenga 8 números seguidos de una letra (no se comprueba la letra oficial)."""
    dni = dni.strip().upper().replace('"', '').replace("'", '')
    if re.match(r'^\d{8}[A-Z]$', dni):
        return "✅ DNI válido."
    return "❌ DNI inválido. Debe tener 8 dígitos seguidos de una letra."


@tool
def validar_correo(correo: str) -> str:
    """Valida si un correo electrónico tiene un formato correcto."""
    correo = correo.strip().lower().replace('"', '').replace("'", '')
    if re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', correo):
        return "✅ Correo válido."
    return "❌ Correo electrónico no es válido."


@tool
def validar_telefono(telefono: str) -> str:
    """Valida si el número de teléfono español es correcto (empieza por 6 o 7 y tiene 9 dígitos)."""
    telefono = telefono.strip().replace('"', '').replace("'", '')
    if re.match(r'^[67]\d{8}$', telefono):
        return "✅ Teléfono válido."
    return "❌ Teléfono no válido. Debe empezar por 6 o 7 y tener 9 dígitos."


@tool
def validar_tarjeta(numero: str) -> str:
    """Valida si un número de tarjeta tiene 16 dígitos."""
    numero = numero.strip().replace('"', '').replace("'", '')
    if re.match(r'^\d{16}$', numero):
        return "✅ Número de tarjeta válido."
    return "❌ Número de tarjeta inválido. Debe tener 16 dígitos."


@tool
def validar_cvv(cvv: str) -> str:
    """Valida si el código CVV tiene 3 dígitos."""
    cvv = cvv.strip().replace('"', '').replace("'", '')
    if re.match(r'^\d{3}$', cvv):
        return "✅ CVV válido."
    return "❌ CVV inválido. Debe tener 3 dígitos."


@tool
def validar_fecha_expiracion(fecha: str) -> str:
    """Valida si la fecha de expiración está en formato MM/AA o AAAA-MM-DD."""
    if re.match(r'^(0[1-9]|1[0-2])/([0-9]{2})$', fecha) or re.match(r'^\d{4}-\d{2}-\d{2}$', fecha):
        return "✅ Fecha de expiración válida."
    return "❌ Fecha inválida. Usa MM/AA o AAAA-MM-DD."


@tool
def fecha_actual(dummy: str) -> str:
    """Devuelve la fecha actual del sistema en formato AAAA-MM-DD."""
    return date.today().isoformat()


@tool
def plantilla_json(dummy: str) -> str:
    """Devuelve un JSON de ejemplo con todos los datos necesarios para registrar un cliente."""
    return json.dumps({
        "cliente": {
            "nombre": "Carlos",
            "apellidos": "Pérez",
            "dni": "12345678X",
            "correo": "carlos@example.com",
            "telefono": "612345678"
        },
        "direccion": {
            "calle": "Gran Vía",
            "numero": "25",
            "ciudad": "Madrid",
            "pais": "España",
            "codigo_postal": "28013"
        },
        "pago": {
            "numero_tarjeta": "4111111111111111",
            "titular_tarjeta": "Carlos Pérez",
            "fecha_expiracion": "2028-12-01",
            "cvv": "123"
        },
        "plan": {
            "plan_id": 2,
            "fecha_inicio": "2024-05-05",
            "activo": True
        }
    }, indent=2)


@tool
def mostrar_planes(dummy: str) -> str:
    """Consulta y muestra los planes disponibles para suscribirse desde la base de datos."""
    return obtener_planes()


@tool
def validar_titular(nombre: str) -> str:
    """No valida nada, simplemente acepta el nombre del titular tal como está."""
    return f"✅ Titular recibido: {nombre}"


tools = [
    Tool(name="validar_dni", func=validar_dni, description="Valida el DNI."),
    Tool(name="validar_correo", func=validar_correo, description="Valida un correo electrónico."),
    Tool(name="validar_telefono", func=validar_telefono, description="Valida un teléfono español."),
    Tool(name="validar_tarjeta", func=validar_tarjeta, description="Valida el número de tarjeta."),
    Tool(name="validar_cvv", func=validar_cvv, description="Valida el código CVV."),
    Tool(name="validar_fecha_expiracion", func=validar_fecha_expiracion, description="Valida la fecha de expiración."),
    Tool(name="validar_titular", func=validar_titular, description="Confirma el nombre del titular de la tarjeta."),
    Tool(name="fecha_actual", func=fecha_actual, description="Devuelve la fecha de hoy."),
    Tool(name="plantilla_json", func=plantilla_json, description="Proporciona un ejemplo de JSON de cliente."),
    Tool(name="mostrar_planes", func=mostrar_planes, description="Muestra los planes disponibles.")
]
parser = TolerantOutputParser()

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
    "system_message": (
        "Actúas como un asistente de registro de clientes. "
        "Tu trabajo es validar datos personales usando exclusivamente las herramientas disponibles. "
        "Nunca debes inventar funciones ni escribir código. "
        "Solo puedes usar herramientas existentes como validar_dni, validar_correo, etc. "
        "Importante: Nunca debes incluir un 'Final Answer' en la misma respuesta en la que haces un 'Action'.\n"
        "Usa siempre esta estructura en cada turno:\n\n"
        "1. Si necesitas validar o consultar algo, responde así:\n"
        "Thought: Necesito validar el DNI del usuario.\n"
        "Action: validar_dni\n"
        "Action Input: \"12345678X\"\n\n"
        "2. Después de recibir la 'Observation', continúa así:\n"
        "Observation: ✅ DNI válido.\n"
        "Thought: El DNI es válido. Ahora puedo informar al usuario.\n"
        "Final Answer: El DNI proporcionado es válido. ¿Deseas continuar?\n\n"
        "No mezcles nunca 'Observation' y 'Final Answer' en la misma respuesta. Espera siempre a la observación antes de dar una respuesta final."
    ),
        "output_parser": parser
                }
)


st.title("Plataforma de usuario FitnessLife")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "registro_datos" not in st.session_state:
    st.session_state.registro_datos = {
        "nombre": None,
        "apellidos": None,
        "dni": None,
        "correo": None,
        "telefono": None,
        "calle": None,
        "numero": None,
        "ciudad": None,
        "pais": None,
        "codigo_postal": None,
        "numero_tarjeta": None,
        "titular_tarjeta": None,
        "fecha_expiracion": None,
        "cvv": None,
        "plan_id": None,
        "fecha_inicio": None,
        "activo": None
    }


for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])


prompt = st.chat_input("Escribe algo para registrarte...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.spinner("Procesando..."):
        try:
            respuesta = agent.run(prompt)
            extraer_validados_de_respuesta(respuesta)
        except Exception as e:
            respuesta = f"❌ Error del agente: {e}"
        st.chat_message("assistant").write(respuesta)
        st.session_state.chat_history.append({"role": "assistant", "content": respuesta})

    def extraer_datos_parciales(texto):
        prompt_extract = f"""
                        Extrae los datos personales de este texto en español. Devuelve un JSON parcial solo con los campos encontrados:
                        nombre, apellidos, dni, correo, telefono, calle, numero, ciudad, pais, codigo_postal, numero_tarjeta, titular_tarjeta, fecha_expiracion, cvv, plan_id, fecha_inicio, activo.

                        Texto del usuario:
                        {texto}
                        """
        res = llm.invoke(prompt_extract)
        try:
            match = re.search(r"\{[\s\S]*?\}", res)
            return json.loads(match.group(0)) if match else {}
        except:
            return {}

    nuevos_datos = extraer_datos_parciales(prompt)
    for campo, valor in nuevos_datos.items():
        if valor not in [None, ""]:
            st.session_state.registro_datos[campo] = valor

    st.chat_message("assistant").write("📋 Estado actual del registro:")
    st.code(json.dumps(st.session_state.registro_datos, indent=2))

    faltan = [k for k, v in st.session_state.registro_datos.items() if v is None]

    if not faltan:
        try:
            datos_finales = {
                "cliente": {
                    "nombre": st.session_state.registro_datos["nombre"],
                    "apellidos": st.session_state.registro_datos["apellidos"],
                    "dni": st.session_state.registro_datos["dni"],
                    "correo": st.session_state.registro_datos["correo"],
                    "telefono": st.session_state.registro_datos["telefono"]
                },
                "direccion": {
                    "calle": st.session_state.registro_datos["calle"],
                    "numero": st.session_state.registro_datos["numero"],
                    "ciudad": st.session_state.registro_datos["ciudad"],
                    "pais": st.session_state.registro_datos["pais"],
                    "codigo_postal": st.session_state.registro_datos["codigo_postal"]
                },
                "pago": {
                    "numero_tarjeta": st.session_state.registro_datos["numero_tarjeta"],
                    "titular_tarjeta": st.session_state.registro_datos["titular_tarjeta"],
                    "fecha_expiracion": st.session_state.registro_datos["fecha_expiracion"],
                    "cvv": st.session_state.registro_datos["cvv"]
                },
                "plan": {
                    "plan_id": st.session_state.registro_datos["plan_id"],
                    "fecha_inicio": st.session_state.registro_datos["fecha_inicio"] or date.today().isoformat(),
                    "activo": bool(st.session_state.registro_datos["activo"])
                }
            }
            guardar_en_bd(datos_finales)
            st.success("✅ Cliente registrado con éxito.")
        except Exception as e:
            st.error(f"❌ Error al guardar en base de datos: {e}")
    else:
        siguiente = f"Puedes proporcioname el dato: **{faltan[0]}**"
        st.chat_message("assistant").write(siguiente)
        st.session_state.chat_history.append({"role": "assistant", "content": siguiente})
