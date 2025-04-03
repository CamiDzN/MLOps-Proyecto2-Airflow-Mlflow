# Proyecto MLOps: Entrenamiento y Despliegue de Modelos con Airflow, MLflow y FastAPI

## 1. Descripción General del Proyecto

Este proyecto implementa una arquitectura MLOps que automatiza todo el ciclo de vida de un modelo de machine learning, desde la recolección de datos hasta su despliegue en un API REST para inferencia.

El objetivo es recolectar datos sintéticos del dataset *Covertype* desde una API, procesarlos, entrenar varios modelos, seleccionar el mejor según su rendimiento y desplegarlo automáticamente en producción, todo mediante un DAG de Apache Airflow.

El proyecto busca ilustrar cómo herramientas como Airflow, MLflow, MinIO, MySQL y FastAPI pueden integrarse para gestionar de manera escalable, reproducible y trazable experimentos de machine learning.

## 2. Arquitectura y Servicios (Docker Compose)

La arquitectura está conformada por los siguientes servicios desplegados mediante Docker Compose y conectados en la red `mlops_net`:

- **Airflow**: orquesta el flujo de trabajo del ML pipeline mediante un DAG. Utiliza PostgreSQL como base de metadatos y Redis como broker.
- **PostgreSQL**: almacena la metadata de Airflow.
- **Redis**: gestiona la cola de tareas de Celery para Airflow.
- **MySQL**: almacena los datos crudos y preprocesados del dataset, y también sirve como backend de MLflow.
- **MinIO**: sistema de almacenamiento de artefactos compatible con S3 (se usa como store de modelos para MLflow).
- **MLflow**: servidor de tracking y Model Registry para registrar experimentos y modelos.
- **FastAPI**: API REST para servir el modelo entrenado y realizar predicciones.
- **Random Data API**: servicio auxiliar que simula la generación incremental de datos del dataset Covertype.

![image](https://github.com/user-attachments/assets/c301c903-1fa1-492a-9945-3dd5a5513cef)


## 3. Tareas del DAG `covertype_workflow`

El DAG principal define 5 tareas:

1. **clear_training_data_tables**: elimina las tablas previas de datos crudos y preprocesados en MySQL.
2. **collect_data**: llama a la API de datos aleatorios y guarda los resultados en la tabla `covertype_raw`.
3. **preprocess_data**: limpia los datos, escala las variables numéricas y guarda los datos en la tabla `covertype_preprocessed`.
4. **train_models**: entrena varios modelos, registra los resultados en MLflow y promueve el mejor a "Production".
5. **notify_api_reload**: notifica al servicio FastAPI para recargar el último modelo de MLflow.

![image](https://github.com/user-attachments/assets/e8ce7ada-d84e-4ee4-8d08-e059a5322c96)


## 4. Registro de Modelos y Artefactos con MLflow

- MLflow está configurado para usar **MySQL** como backend store (`mlflow_db`) y **MinIO** como artifact store (`s3://mlflows3/artifacts`).
- Cada ejecución del DAG crea un nuevo *run* con los siguientes elementos registrados:
  - Hiperparámetros del modelo
  - Métrica de accuracy
  - Modelo serializado (formato `mlflow.sklearn`)
- El mejor modelo se registra en el **Model Registry** bajo el nombre `CovertypeModel`, y se promueve automáticamente al stage **Production**.

## 5. API de Inferencia (FastAPI)

La API corre en `http://localhost:8081` y ofrece los siguientes endpoints:

- `POST /predict/`: recibe un JSON con las 10 variables numéricas y retorna la clase `Cover_Type` predicha.
- `POST /reload_models/`: recarga los modelos en memoria desde MLflow.
- `PUT /select_model/{name}`: cambia el modelo activo por nombre.
- `POST /promote_models/`: promueve una versión específica al stage Production (opcional).

![image](https://github.com/user-attachments/assets/f0769a60-5b49-4049-b267-90ed5cb7a2ba)

![image](https://github.com/user-attachments/assets/10cb1e9b-d682-4bc4-9c44-92bfc6ae6850)



## 6. Despliegue del Proyecto

### Estructura de Carpetas

```
MLOps-Proyecto2/
├── airflow/
│   └── dags/
│       └── covertype_workflow.py
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── app.py
│   └── requirements.txt
├── random-data-api/
│   ├── main.py
│   └── requirements.txt
├── Data/
│   └── covertype.csv
├── docker-compose.yml
├── .env
└── docs/
    └── arquitectura-mlops.png
```

### Pasos para Desplegar

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/<usuario>/MLOps-Proyecto2-Airflow-Mlflow.git
   cd MLOps-Proyecto2-Airflow-Mlflow
   ```

2. Crear el archivo `.env` con:
   ```dotenv
   AIRFLOW_UID=1001
   _AIRFLOW_WWW_USER_USERNAME=airflow
   _AIRFLOW_WWW_USER_PASSWORD=airflow
   ```

3. Levantar los servicios:
   ```bash
   docker-compose up -d --build
   ```

4. Acceder a Airflow UI:
   [http://localhost:8080](http://localhost:8080)

5. Trigger el DAG `covertype_workflow` desde la UI o vía CLI:
   ```bash
   docker-compose exec airflow-webserver airflow dags trigger covertype_workflow
   ```

6. Probar la inferencia:
   ```bash
   curl -X POST http://localhost:8081/predict/ \
     -H "Content-Type: application/json" \
     -d '{
       "Elevation": 2500,
       "Aspect": 45,
       "Slope": 9,
       "Horizontal_Distance_To_Hydrology": 100,
       "Vertical_Distance_To_Hydrology": 10,
       "Horizontal_Distance_To_Roadways": 500,
       "Hillshade_9am": 200,
       "Hillshade_Noon": 220,
       "Hillshade_3pm": 150,
       "Horizontal_Distance_To_Fire_Points": 300
     }'
   ```

---

> ✅ Si deseas contribuir o reportar errores, abre un *issue* o *pull request*. Este proyecto es parte de una práctica de MLOps aplicada para entornos reales.

