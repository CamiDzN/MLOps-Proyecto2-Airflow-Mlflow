from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc  # Para cargar el modelo desde el registro
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow

app = FastAPI()

# Lista de nombres de modelos a buscar en el registry de MLflow
model_names = ["random_forest", "decision_tree", "svm", "logistic_regression"]

# Diccionario para almacenar los modelos cargados (clave: nombre, valor: modelo MLflow)
models = {}

def promote_models():
    """
    Promociona una versión específica de cada modelo al stage "Production"
    usando MlflowClient. Ajusta los números de versión según tu registro.
    """
    client = MlflowClient(tracking_uri="http://mlflow:5000")
    # Diccionario de modelos y versiones a promocionar (ajusta los números según corresponda)
    models_to_promote = {
        "random_forest": 29,
        "decision_tree": 24,
        "svm": 25,
        "logistic_regression": 22
    }
    
    for model_name, version in models_to_promote.items():
        try:
            client.transition_model_version_stage(
                name=model_name,   # Nombre exacto del modelo
                version=version,   # Versión a promocionar
                stage="Production" # Stage al que se desea pasar
            )
            print(f"Promovida la versión {version} del modelo '{model_name}' a Production.")
        except Exception as e:
            print(f"Error al promocionar el modelo '{model_name}' versión {version}: {e}")

def load_models():
    """
    Carga los modelos registrados en MLflow para cada nombre presente en model_names.
    Se construye el URI de cada modelo con el formato: models:/{model_name}/Production.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    global models
    models = {}  # Reinicia el diccionario
    print("Cargando modelos desde MLflow para:", model_names)
    for name in model_names:
        model_uri = f"models:/{name}/Production"
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            models[name] = loaded_model
            print(f"Modelo '{name}' cargado exitosamente desde {model_uri}")
        except Exception as e:
            print(f"Error al cargar el modelo '{name}' desde {model_uri}: {e}")
    print("Modelos actualmente cargados:", list(models.keys()))

# Promocionar y cargar modelos al iniciar la API
promote_models()
load_models()

# Modelo seleccionado por defecto
selected_model = "random_forest"

# Esquema de entrada para la predicción: únicamente las 10 variables numéricas,
# de acuerdo al preprocesamiento realizado en el DAG.
class CovertypeFeatures(BaseModel):
    Elevation: float
    Aspect: float
    Slope: float
    Horizontal_Distance_To_Hydrology: float
    Vertical_Distance_To_Hydrology: float
    Horizontal_Distance_To_Roadways: float
    Hillshade_9am: float
    Hillshade_Noon: float
    Hillshade_3pm: float
    Horizontal_Distance_To_Fire_Points: float

@app.post("/predict/")
def predict(features: CovertypeFeatures):
    """
    Realiza la predicción usando el modelo seleccionado.
    Se recibe un JSON con las 10 variables numéricas, se construye un DataFrame y se
    envía al modelo para obtener la predicción del Cover_Type (valor numérico).
    """
    global selected_model
    if selected_model not in models:
        raise HTTPException(status_code=400, detail=f"Modelo '{selected_model}' no encontrado.")

    # Preparar los datos de entrada en un DataFrame
    input_data = pd.DataFrame([features.dict()])

    try:
        # Realizar la predicción directamente con el modelo cargado
        prediction = models[selected_model].predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la predicción: {str(e)}")

    # Se retorna el valor numérico predicho para Cover_Type
    return {"selected_model": selected_model, "prediction": int(prediction[0])}

@app.put("/select_model/{model_name}")
def select_model(model_name: str):
    """
    Permite seleccionar un modelo distinto al actual.
    Si el modelo solicitado no ha sido cargado, retorna un error.
    """
    global selected_model
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no encontrado.")
    selected_model = model_name
    return {"message": f"Modelo cambiado a '{model_name}'"}

@app.get("/")
def home():
    return {"message": "API de Predicción de Covertype con FastAPI y MLflow"}

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