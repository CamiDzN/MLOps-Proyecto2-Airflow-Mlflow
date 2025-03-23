from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import mlflow.pyfunc  # Usamos mlflow para cargar el modelo desde el registro
import os
import pandas as pd
from mlflow.tracking import MlflowClient

app = FastAPI()

# Lista de nombres de modelos a buscar en el registry de MLflow
model_names = ["random_forest", "decision_tree", "svm", "logistic_regression"]

# Diccionario para almacenar los modelos cargados (clave: nombre, valor: modelo mlflow)
models = {}

def promote_models():
    """
    Promociona una versión específica de cada modelo al stage 'Production'
    usando MlflowClient. Ajusta los números de versión según tu registro.
    """
    client = MlflowClient(tracking_uri="http://10.43.101.173:5000")

    # Diccionario de modelos y versiones a promocionar
    models_to_promote = {
        "random_forest": 29,        # Ajusta el número de versión
        "decision_tree": 24,        # Ajusta el número de versión
        "svm": 25,                  # Ajusta el número de versión
        "logistic_regression": 22   # Ajusta el número de versión
    }

    for model_name, version in models_to_promote.items():
        try:
            client.transition_model_version_stage(
                name=model_name,   # Nombre exacto del modelo
                version=version,   # Número de versión que se desea promocionar
                stage="Production" # Stage al que se quiere pasar
            )
            print(f"Promovida la versión {version} del modelo '{model_name}' a Production.")
        except Exception as e:
            print(f"Error al promocionar el modelo '{model_name}' versión {version}: {e}")

def load_models():
    """
    Carga los modelos registrados en MLflow para cada nombre presente en model_names.
    Se construye el URI de cada modelo con el formato: models:/{model_name}/Production.
    """
    mlflow.set_tracking_uri("http://10.43.101.173:5000")  # Asegúrate de usar la IP/hostname correcto
    global models
    models = {}  # Reinicia el diccionario
    print("Cargando modelos desde MLflow para:", model_names)
    for name in model_names:
        model_uri = f"models:/{name}/Production"
        try:
            # Cargar el modelo usando mlflow.pyfunc
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            models[name] = loaded_model
            print(f"Modelo '{name}' cargado exitosamente desde {model_uri}")
        except Exception as e:
            print(f"Error al cargar el modelo '{name}' desde {model_uri}: {e}")
    print("Modelos actualmente cargados:", list(models.keys()))

# Promocionar modelos (asegúrate de ajustar las versiones)
promote_models()

# Cargar los modelos al iniciar la API
load_models()

# Modelo seleccionado por defecto
selected_model = "random_forest"

# Esquema de entrada para la predicción (en este caso, características de pingüinos)
class PenguinFeatures(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: int

@app.post("/predict/")
def predict(features: PenguinFeatures):
    """
    Realiza la predicción usando el modelo seleccionado.
    Se recibe un JSON con las características del pingüino, se construye un DataFrame y se envía al modelo.
    Se mapea el resultado numérico a una etiqueta (por defecto: 1 -> "MALE", 0 -> "FEMALE").
    """
    global selected_model
    if selected_model not in models:
        raise HTTPException(status_code=400, detail=f"Modelo '{selected_model}' no encontrado.")

    # Preparar los datos de entrada en un DataFrame
    input_data = pd.DataFrame([{
        "culmen_length_mm": features.culmen_length_mm,
        "culmen_depth_mm": features.culmen_depth_mm,
        "flipper_length_mm": features.flipper_length_mm,
        "body_mass_g": features.body_mass_g,
        "island": features.island
    }])

    try:
        # Realizar la predicción directamente con el modelo cargado
        prediction = models[selected_model].predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la predicción: {str(e)}")

    # Mapear el resultado numérico a una etiqueta
    sex = "MALE" if prediction[0] == 1 else "FEMALE"
    return {"selected_model": selected_model, "prediction": sex}

@app.put("/select_model/{model_name}")
def select_model(model_name: str):
    """
    Permite seleccionar un modelo distinto al actual.
    Si el modelo solicitado no ha sido cargado, se retorna un error.
    """
    global selected_model
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no encontrado.")
    selected_model = model_name
    return {"message": f"Modelo cambiado a '{model_name}'"}

@app.get("/")
def home():
    return {"message": "API de Predicción de Pingüinos con FastAPI basada en MLflow"}

@app.post("/reload_models/")
def reload_models():
    """
    Permite recargar los modelos desde el registro MLflow en caso de actualizarlos.
    """
    load_models()
    return {"message": "Modelos recargados", "models_loaded": list(models.keys())}

@app.post("/promote_models/")
def promote_models_endpoint():
    """
    Endpoint para promocionar (transition) las versiones de los modelos al stage Production.
    """
    promote_models()
    return {"message": "Modelos promocionados a Production"}
