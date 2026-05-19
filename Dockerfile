FROM apache/airflow:3.0.2

USER root

# Install system dependency required by LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY src/ /opt/airflow/src/
COPY data/Flight_Price_Dataset_of_Bangladesh.csv /opt/airflow/data/Flight_Price_Dataset_of_Bangladesh.csv

USER airflow

# Pin all versions to match requirements.txt — keeps Docker and local envs identical.
# xgboost must stay at 2.0.3: model artefacts are serialised with this version.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.0 \
    scikit-learn==1.5.0 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    joblib==1.4.2 \
    streamlit==1.35.0
