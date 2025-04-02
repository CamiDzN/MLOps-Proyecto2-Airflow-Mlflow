from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd

##############################################################################
# 1) Configuración de la API
##############################################################################
app = FastAPI()

# Nombre con el que se registró el modelo en MLflow. 
MODEL_NAME = "CovertypeModel"
loaded_model = None  # Variable global; cada worker tendrá su propia copia.

##############################################################################
# 2) Evento de startup (se ejecuta en cada worker de Uvicorn/Gunicorn al inicio)
##############################################################################
@app.on_event("startup")
def load_model_on_startup():
    """
    Esta función se llama cuando cada proceso/worker de FastAPI arranca. 
    Cargamos el modelo de la registry (stage Production). 
    Así, cada worker tendrá la variable 'loaded_model' en memoria.
    """
    global loaded_model
    # Ajusta la URI de tracking
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Construye la URI para stage Production
    model_uri = f"models:/{MODEL_NAME}/Production"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"Modelo '{MODEL_NAME}' cargado exitosamente desde: {model_uri}")
    except Exception as e:
        print(f"Error al cargar el modelo '{MODEL_NAME}': {str(e)}")
        loaded_model = None


##############################################################################
# 3) Definición del esquema de entrada para la predicción
##############################################################################
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


##############################################################################
# 4) Endpoints
##############################################################################
@app.get("/")
def home():
    """
    Endpoint básico para verificar que la API esté funcionando.
    """
    return {"message": "API de predicción con MLflow - CovertypeModel"}


@app.post("/reload_models/")
def reload_models():
    """
    En caso de actualizar el modelo en Production, 
    este endpoint vuelve a llamar load_model_on_startup().
    
    (Cada worker llamará a su version on_startup, 
    pero para simplificar, aquí forzamos en el thread actual.)
    """
    global loaded_model
    mlflow.set_tracking_uri("http://mlflow:5000")
    model_uri = f"models:/{MODEL_NAME}/Production"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        return {"message": f"Modelo {MODEL_NAME} recargado desde {model_uri}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recargar: {str(e)}")


@app.post("/predict/")
def predict(features: CovertypeFeatures):
    """
    Endpoint para realizar la predicción con las 10 variables.
    """
    global loaded_model

    if loaded_model is None:
        raise HTTPException(
            status_code=500, 
            detail="No hay modelo cargado en memoria. Llama /reload_models/ o reinicia la API."
        )

    # Convertir a DataFrame
    input_data = pd.DataFrame([features.dict()])

    try:
        prediction = loaded_model.predict(input_data)
        # Asumimos que devuelves un entero con la clase
        return {
            "model": MODEL_NAME,
            "prediction": int(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")
