# TFG-LLMDataCollector

**Recogida de datos utilizando agentes basados en modelos grandes de lenguaje (LLM)**

Este repositorio contiene el desarrollo del Trabajo de Fin de Grado (TFG) titulado _"Recogida de datos utilizando agentes basados en modelos grandes de lenguaje (LLM)"_, cuyo objetivo es diseñar, implementar y evaluar un sistema automatizado de recolección de datos utilizando modelos de lenguaje de última generación.

##  Objetivos del proyecto

- Desarrollar una herramienta que permita recopilar y validar datos estructurados de forma conversacional mediante interacciones con modelos de lenguaje (LLMs).
- Automatizar la simulación de conversaciones y la evaluación de respuestas y resultados obtenidos.
- Evaluar y comparar el desempeño de diferentes modelos de lenguaje en la tarea de recogida de datos.

##  Estructura del repositorio
```
TFG-LLMDataCollector/
├── src/
│ ├── images/ # Recursos gráficos (logos, etc.)
│ │ └── bot_logo.jpeg
│ ├── prompts/ # Prompts y mensajes del sistema
│ │ └── system_message.txt
│ └── utils/ # Funciones auxiliares y utilidades
│ ├── functions.py
│ ├── logger.py
│ ├── metricas_logger.py
│ └── verification.py
├── app.py # Script principal de ejecución de la app
├── dashboard.py # Visualización e interfaz de resultados
├── evaluador.py # Evaluación automática de resultados generados
├── slm_app.py # Aproximación con slm
├── .gitignore # Exclusión de archivos del control de versiones
├── README.md # Documentación principal
```
##  Configuración del entorno 
Debes crear un archivo ```.env``` en la raíz del proyecto con el siguiente contenido:
```
#API keys
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
COHERE_API_KEY =
ANTHROPIC_API_KEY =
GOOGLE_API_KEY = 

#database
DB_USERNAME2= #username con permisos de select
DB_PASSWORD2= #password con permisos de select
DB_USERNAME= #username sin permisos de select
DB_PASSWORD= #username con permisos de select
DB_HOSTNAME=
DB_PORT=
DB_NAME=
DB_SCHEMA=
DB_CONNECTION_URL=
```

##  Ejecución

Para desplegar y abrir el agente conversacional debes ejecutar el script principal:
```
streamlit run app.py
```

Para ejecutar la simulación de conversaciones debes ejecutar:
```
python3 evauador.py
```
