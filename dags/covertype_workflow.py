from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy
import joblib
import requests
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Parámetros por defecto del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'covertype_workflow',
    default_args=default_args,
    description='Workflow para recolectar, procesar y entrenar modelos con el dataset covertype',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Tarea 1: Borrar las tablas existentes en MySQL (utilizando la conexión "mysql_default" que apunta a model_db)
clear_tables = MySqlOperator(
    task_id='clear_training_data_tables',
    mysql_conn_id='mysql_default',  # Apunta a model_db según la configuración en docker-compose.
    sql="""
        DROP TABLE IF EXISTS covertype_raw;
        DROP TABLE IF EXISTS covertype_preprocessed;
    """,
    dag=dag,
)

# Tarea 2: Recolectar los datos vía API desde random-data-api y guardarlos en MySQL
def collect_covertype_data(**kwargs):
    import requests
    import pandas as pd
    import sqlalchemy

    data_list = []
    # Se recorren los 10 grupos (batchs)
    for group_number in range(1, 11):
        url = f"http://random-data-api:8001/data?group_number={group_number}"
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            # result["data"] contiene la porción aleatoria del batch solicitado
            data_list.extend(result.get("data", []))
        else:
            raise Exception(f"Error al obtener datos para el grupo {group_number}: {response.text}")

    # Las columnas que entrega el API en el siguiente orden:
    # Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology,
    # Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
    # Horizontal_Distance_To_Fire_Points, Wilderness_Area, Soil_Type, Cover_Type
    columns = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
        "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area", "Soil_Type", "Cover_Type"
    ]

    df = pd.DataFrame(data_list, columns=columns)

    # Conectar a la base de datos model_db usando SQLAlchemy.
    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')
    # Guardar el dataset en la tabla "covertype_raw". Se usa if_exists="replace" para actualizar la información.
    df.to_sql('covertype_raw', con=engine, if_exists='replace', index=False)
    print("Datos crudos guardados en MySQL en la tabla covertype_raw.")

collect_data_task = PythonOperator(
    task_id='collect_data',
    python_callable=collect_covertype_data,
    dag=dag,
)

# Tarea 3: Leer los datos crudos desde MySQL, preprocesarlos y guardarlos como preprocesados
def preprocess_covertype_data(**kwargs):
    import pandas as pd
    import sqlalchemy
    from sklearn.preprocessing import StandardScaler

    # Conectar a la base de datos model_db
    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')
    # Leer los datos crudos de la tabla "covertype_raw"
    df = pd.read_sql('SELECT * FROM covertype_raw', con=engine)
    
    # Eliminar filas con datos faltantes
    df_clean = df.dropna()
    
    # Eliminar las variables categóricas, ya que se entrena únicamente con variables numéricas.
    df_numeric = df_clean.drop(columns=["Wilderness_Area", "Soil_Type"])
    
    # Escalar las variables numéricas (las 10 primeras columnas)
    num_cols = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
        "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
    ]
    scaler = StandardScaler()
    df_numeric[num_cols] = scaler.fit_transform(df_numeric[num_cols])
    
    # Guardar el dataset preprocesado (únicamente numérico) en la tabla "covertype_preprocessed"
    df_numeric.to_sql('covertype_preprocessed', con=engine, if_exists='replace', index=False)
    print("Datos preprocesados (únicamente numéricos) guardados en MySQL en la tabla covertype_preprocessed.")

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_covertype_data,
    dag=dag,
)

# Tarea 4: Entrenar modelos, registrar experimentos en MLflow y marcar el mejor modelo en Producción
def train_and_log_models(**kwargs):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    
    # Definir el Tracking URI de MLflow para el servidor
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    import pandas as pd
    import sqlalchemy
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')
    df = pd.read_sql('SELECT * FROM covertype_preprocessed', con=engine)

    # "Cover_Type" sigue siendo la variable objetivo; las características son ahora únicamente las variables numéricas.
    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(),
        "svm": SVC(kernel="linear", probability=True),
        "logistic_regression": LogisticRegression(max_iter=1000)
    }

    best_model_name = None
    best_accuracy = 0
    best_run_id = None  # Identificador del mejor run

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, artifact_path=name)
            mlflow.end_run()
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_run_id = run.info.run_id

    # Registrar el mejor modelo en la etapa "Production"
    if best_run_id is not None:
        model_uri = f"runs:/{best_run_id}/{best_model_name}"
        mlflow.register_model(model_uri, "CovertypeModel")
        client = MlflowClient()
        latest_versions = client.get_latest_versions("CovertypeModel", stages=["None"])
        if latest_versions:
            version = latest_versions[0].version
            client.transition_model_version_stage(
                name="CovertypeModel",
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
    print("Entrenamiento completado. Mejor modelo registrado en Producción.")

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_and_log_models,
    dag=dag,
)

# Tarea 5: Notificar a la API de inferencia para recargar el último modelo
def notify_api_reload():
    try:
        response = requests.post("http://fastapi:8000/reload_models/")
        print("Respuesta del API de inferencia:", response.json())
    except Exception as e:
        print("Error al notificar al API de inferencia:", e)

notify_reload_task = PythonOperator(
    task_id='notify_api_reload',
    python_callable=notify_api_reload,
    dag=dag,
)

# Secuencia de ejecución
clear_tables >> collect_data_task >> preprocess_data_task >> train_models_task >> notify_reload_task
