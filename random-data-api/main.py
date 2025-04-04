from fastapi import FastAPI, HTTPException
import random
import json
import time
import csv
import os

app = FastAPI()

MIN_UPDATE_TIME = 300  # 5 minutos

@app.get("/")
async def root():
    return {"Proyecto 2": "Extracción de datos, entrenamiento de modelos."}

# Cargar datos desde /data/covertype.csv
data = []
with open('/data/covertype.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # saltar encabezado
    for row in reader:
        data.append(row)

batch_size = len(data) // 10

if not os.path.isfile('/data/timestamps.json'):
    # Inicializa timestamps para 10 grupos (1..10)
    timestamps = {str(i): [0, -1] for i in range(1, 11)}
    with open('/data/timestamps.json', 'w') as f:
        json.dump(timestamps, f)
else:
    with open('/data/timestamps.json', 'r') as f:
        timestamps = json.load(f)

def get_batch_data(batch_number: int):
    start_index = (batch_number - 1) * batch_size
    end_index = batch_number * batch_size
    if end_index > len(data):
        end_index = len(data)
    subset = data[start_index:end_index]
    required_sample = batch_size // 10 or 1
    if len(subset) < required_sample:
        raise Exception(f"No hay suficientes datos en el batch {batch_number}.")
    random_data = random.sample(subset, required_sample)
    return random_data

@app.get("/data")
async def read_data(group_number: int):
    if group_number < 1 or group_number > 10:
        raise HTTPException(status_code=400, detail="Número de grupo inválido")

    if timestamps[str(group_number)][1] >= 10:
        raise HTTPException(status_code=400, detail="Ya se recolectó toda la información mínima necesaria")

    current_time = time.time()
    last_update_time = timestamps[str(group_number)][0]

    if current_time - last_update_time > MIN_UPDATE_TIME:
        timestamps[str(group_number)][0] = current_time
        timestamps[str(group_number)][1] += 2 if timestamps[str(group_number)][1] == -1 else 1

    random_data = get_batch_data(group_number)

    with open('/data/timestamps.json', 'w') as f:
        json.dump(timestamps, f)

    return {
        "group_number": group_number,
        "batch_number": timestamps[str(group_number)][1],
        "data": random_data
    }

@app.get("/restart_data_generation")
async def restart_data(group_number: int):
    if group_number < 1 or group_number > 10:
        raise HTTPException(status_code=400, detail="Número de grupo inválido")
    timestamps[str(group_number)][0] = 0
    timestamps[str(group_number)][1] = -1
    with open('/data/timestamps.json', 'w') as f:
        json.dump(timestamps, f)
    return {"ok": True}
