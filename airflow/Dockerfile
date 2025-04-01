FROM apache/airflow:2.6.0

# 1) Cambiamos a usuario root para instalar librerías del sistema
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libssl-dev \
    libffi-dev \
    build-essential \
    libgomp1 \
    cmake \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Volvemos al usuario 'airflow' para instalar dependencias de Python
USER airflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
