from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from datetime import datetime, timedelta
import logging
import pandas as pd
import sqlalchemy
import requests
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    description='Workflow para recolección, preprocesamiento y entrenamiento con MLflow',
    schedule_interval=timedelta(days=1),
    catchup=False
)

################################
# Tarea 1: Borrar tablas
################################
clear_tables = MySqlOperator(
    task_id='clear_training_data_tables',
    mysql_conn_id='mysql_default',
    sql="""
        DROP TABLE IF EXISTS covertype_raw;
        DROP TABLE IF EXISTS covertype_preprocessed;
    """,
    dag=dag,
)

################################
# Tarea 2: Recolectar
################################
def collect_covertype_data():
    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')
    data_list = []

    # Pedimos datos a random-data-api (internamente lo llamamos: http://random-data-api:8001/data)
    for group_number in range(1, 11):
        url = f"http://10.43.101.172:8001/data?group_number={group_number}"
        resp = requests.get(url)
        if resp.status_code == 200:
            result = resp.json()
            data_list.extend(result.get("data", []))
        elif resp.status_code == 400 and "Ya se recolectó" in resp.text:
            logging.info(f"[Grupo {group_number}] {resp.text}")
        else:
            raise Exception(f"Error al obtener datos del grupo {group_number}: {resp.text}")

    columns = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon",
        "Hillshade_3pm","Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area","Soil_Type","Cover_Type"
    ]

    df = pd.DataFrame(data_list, columns=columns)
    df.to_sql('covertype_raw', con=engine, if_exists='replace', index=False)
    logging.info("Datos crudos guardados en la tabla covertype_raw.")

collect_data_task = PythonOperator(
    task_id='collect_data',
    python_callable=collect_covertype_data,
    dag=dag,
)

################################
# Tarea 3: Preprocesado
################################
def preprocess_covertype_data():
    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')
    df = pd.read_sql('SELECT * FROM covertype_raw', con=engine)

    if df.empty:
        logging.info("Tabla covertype_raw vacía -> covertype_preprocessed vacía")
        empty_cols = [
            "Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
            "Hillshade_9am","Hillshade_Noon","Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points","Cover_Type"
        ]
        pd.DataFrame(columns=empty_cols).to_sql('covertype_preprocessed', con=engine, if_exists='replace', index=False)
        return

    df_clean = df.dropna()
    if df_clean.empty:
        logging.info("Tras dropna, sin filas -> covertype_preprocessed vacía")
        empty_cols = [
            "Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
            "Hillshade_9am","Hillshade_Noon","Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points","Cover_Type"
        ]
        pd.DataFrame(columns=empty_cols).to_sql('covertype_preprocessed', con=engine, if_exists='replace', index=False)
        return

    df_numeric = df_clean.drop(columns=["Wilderness_Area","Soil_Type"], errors='ignore')
    num_cols = [
        "Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
        "Hillshade_9am","Hillshade_Noon","Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ]

    if df_numeric[num_cols].empty:
        logging.info("Sin columnas numéricas -> covertype_preprocessed vacía")
        empty_cols = num_cols + ["Cover_Type"]
        pd.DataFrame(columns=empty_cols).to_sql('covertype_preprocessed', con=engine, if_exists='replace', index=False)
        return

    scaler = StandardScaler()
    df_numeric[num_cols] = scaler.fit_transform(df_numeric[num_cols])
    df_numeric.to_sql('covertype_preprocessed', con=engine, if_exists='replace', index=False)
    logging.info("Preprocesado -> covertype_preprocessed.")

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_covertype_data,
    dag=dag,
)

################################
# Tarea 4: Entrenar + MLflow
################################
def train_and_log_models():
    # Internamente Airflow habla con mlflow:5000
    mlflow.set_tracking_uri("http://mlflow:5000")
    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')

    # Revisamos si la tabla preprocessed existe
    inspector = sqlalchemy.inspect(engine)
    if 'covertype_preprocessed' not in inspector.get_table_names():
        logging.info("No existe covertype_preprocessed. Omito entrenamiento.")
        return

    df = pd.read_sql('SELECT * FROM covertype_preprocessed', con=engine)
    if df.empty:
        logging.info("covertype_preprocessed está vacía. Omito entrenamiento.")
        return

    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]
    if X.empty or y.empty:
        logging.info("Sin datos para entrenar. Omito.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "svm": SVC(kernel="linear", probability=True),
        "logistic_regression": LogisticRegression(max_iter=1000)
    }

    best_model_name = None
    best_accuracy = 0
    best_run_id = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, artifact_path=name)

            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_run_id = run.info.run_id

    if best_run_id:
        # Registrar en Model Registry -> "Models" en la UI
        model_uri = f"runs:/{best_run_id}/{best_model_name}"
        result = mlflow.register_model(model_uri, "CovertypeModel")
        logging.info(f"Registrado en Model Registry: {result.name} v{result.version}")

        client = MlflowClient()
        client.transition_model_version_stage(
            name="CovertypeModel",
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )
        logging.info(f"Mejor modelo: {best_model_name}, acc={best_accuracy}, stage=Production")
    else:
        logging.info("No se encontró un best_run_id, quizás no había datos.")

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_and_log_models,
    dag=dag,
)

################################
# Tarea 5: Notificar a la API de inferencia
################################
def notify_api_reload():
    try:
        r = requests.post("http://fastapi:8081/reload_models/")
        logging.info(f"Respuesta reload_models: {r.json()}")
    except Exception as e:
        logging.error(f"Error al notificar reload: {e}")

notify_reload_task = PythonOperator(
    task_id='notify_api_reload',
    python_callable=notify_api_reload,
    dag=dag,
)

clear_tables >> collect_data_task >> preprocess_data_task >> train_models_task >> notify_reload_task
