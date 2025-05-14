import os
import csv
from datetime import datetime

current_metricas = []
current_usuario_dni = None
current_modelo = None
metricas_folder = None


def iniciar_metricas_conversacion(modelo, dni):
    global current_metricas, current_usuario_dni, current_modelo, metricas_folder
    current_metricas = []
    current_usuario_dni = dni
    current_modelo = modelo
    fecha = datetime.now().strftime("%Y-%m-%d")
    metricas_folder = os.path.join("log", fecha, modelo.replace(" ", "_"))
    os.makedirs(metricas_folder, exist_ok=True)


def log_metricas_turno(input_usuario, respuesta_modelo, respuesta_esperada, cosine, bleu, meteor, rouge_l, turno):
    global current_metricas
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_metricas.append({
        "turno": turno,
        "timestamp": timestamp,
        "input_usuario": input_usuario,
        "respuesta_esperada": respuesta_esperada,
        "respuesta_modelo": respuesta_modelo,
        "cosine": cosine,
        "bleu": bleu,
        "meteor": meteor,
        "rougeL": rouge_l
    })


def finalizar_metricas_conversacion(turnos_totales, registro_ok, detalle_fallo, campos_faltantes=None):
    global current_metricas, current_usuario_dni, current_modelo, metricas_folder
    resumen = {
        "cosine_avg": sum(m["cosine"] for m in current_metricas) / len(current_metricas),
        "bleu_avg": sum(m["bleu"] for m in current_metricas) / len(current_metricas),
        "meteor_avg": sum(m["meteor"] for m in current_metricas) / len(current_metricas),
        "rougeL_avg": sum(m["rougeL"] for m in current_metricas) / len(current_metricas)
    }
    filepath = os.path.join(metricas_folder, f"metricas_{current_usuario_dni}.csv")
    with open(filepath, "w", newline='', encoding="utf-8") as f:
        fieldnames = ["turno", "timestamp", "input_usuario", "respuesta_esperada", "respuesta_modelo", "cosine", "bleu", "meteor", "rougeL"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in current_metricas:
            writer.writerow(m)
    with open(filepath, "a", newline='', encoding="utf-8") as f:
        f.write("\nResumen por conversación\n")
        f.write(f"turnos_totales,{turnos_totales}\n")
        f.write(f"registro_ok,{registro_ok}\n")
        f.write(f"detalle_fallo,{detalle_fallo}\n")
        if campos_faltantes:
            f.write(f"campos_faltantes,{','.join(campos_faltantes)}\n")
        for key, val in resumen.items():
            f.write(f"{key},{val:.4f}\n")
