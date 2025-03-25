from fastapi import FastAPI, HTTPException
from typing import Optional
import random
import json
import time
import csv
import os

MIN_UPDATE_TIME = 300  # Tiempo mínimo (en segundos) para cambiar el bloque de información

app = FastAPI()

# Comentario: Las columnas del CSV son:
# Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology,
# Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
# Horizontal_Distance_To_Fire_Points, Wilderness_Area, Soil_Type, Cover_Type

@app.get("/")
async def root():
    return {"Proyecto 2": "Extracción de datos, entrenamiento de modelos."}

# Cargar los datos del archivo CSV
data = []
with open('/data/covertype.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # Saltar encabezado
    for row in reader:
        data.append(row)

# Calcular el tamaño de cada batch (asumiendo que dividimos uniformemente)
batch_size = len(data) // 10

# Definir la función para generar la fracción de datos aleatoria
def get_batch_data(batch_number: int, batch_size: int = batch_size):
    # Usar (batch_number - 1) para calcular el índice inicial
    start_index = (batch_number - 1) * batch_size
    end_index = batch_number * batch_size
    # Asegurarse de no exceder el largo de data (especialmente para el último batch)
    if end_index > len(data):
        end_index = len(data)
    # Verificar que hay suficientes filas para muestrear
    subset = data[start_index:end_index]
    required_sample = batch_size // 10
    if len(subset) < required_sample:
        raise Exception(f"No hay suficientes datos en el batch {batch_number} para muestrear {required_sample} filas.")
    random_data = random.sample(subset, required_sample)
    return random_data

# Cargar información previa si existe
if os.path.isfile('/data/timestamps.json'):
    with open('/data/timestamps.json', "r") as f:
        timestamps = json.load(f)
else:
    # Inicializar el diccionario para almacenar los timestamps de cada grupo (grupos de 1 a 10)
    timestamps = {str(group_number): [0, -1] for group_number in range(1, 11)}

# Definir la ruta de la API para obtener datos
@app.get("/data")
async def read_data(group_number: int):
    global timestamps

    # Verificar si el número de grupo es válido (1 a 10)
    if group_number < 1 or group_number > 10:
        raise HTTPException(status_code=400, detail="Número de grupo inválido")
    # Verificar si ya se ha recolectado la información mínima para ese grupo
    if timestamps[str(group_number)][1] >= 10:
        raise HTTPException(status_code=400, detail="Ya se recolectó toda la información mínima necesaria")

    current_time = time.time()
    last_update_time = timestamps[str(group_number)][0]

    # Actualizar el timestamp y el conteo si ha pasado el tiempo mínimo
    if current_time - last_update_time > MIN_UPDATE_TIME:
        timestamps[str(group_number)][0] = current_time
        timestamps[str(group_number)][1] += 2 if timestamps[str(group_number)][1] == -1 else 1

    # Obtener la porción de datos para el grupo solicitado
    random_data = get_batch_data(group_number)
    with open('/data/timestamps.json', 'w') as file:
        file.write(json.dumps(timestamps))

    return {
        "group_number": group_number,
        "batch_number": timestamps[str(group_number)][1],
        "data": random_data
    }

@app.get("/restart_data_generation")
async def restart_data(group_number: int):
    # Verificar si el número de grupo es válido
    if group_number < 1 or group_number > 10:
        raise HTTPException(status_code=400, detail="Número de grupo inválido")
    timestamps[str(group_number)][0] = 0
    timestamps[str(group_number)][1] = -1
    with open('/data/timestamps.json', 'w') as file:
        file.write(json.dumps(timestamps))
    return {'ok'}
