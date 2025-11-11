# Estudo 01 fastAPI

import os
import json
import mlflow
import uvicorn
import numpy as np
from dotenv import load_dotenv
from os import getenv
from pydantic import BaseModel
from fastapi import FastAPI

DOTENV_PATH = '.env'
load_dotenv(dotenv_path=DOTENV_PATH)

class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float




app = FastAPI(title="Fetal Health API",
              openapi_tags=[{
                             "name":"Health",
                             "description":"Get api health"
                             },
                             {
                              "name":"Prediction",
                              "description":"Model prediction"
                              }])



def load_model():
    MLFLOW_TRACKING_URI = getenv('MLFLOW_TRACKING_URI', '')
    MLFLOW_TRACKING_USERNAME = getenv('MLFLOW_TRACKING_USERNAME', '')
    MLFLOW_TRACKING_PASSWORD = getenv('MLFLOW_TRACKING_PASSWORD', '')
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    registered_model = client.get_registered_model(getenv('MODEL_NAME', 'fetal_health'))
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model

@app.on_event(event_type="startup")
def startup_event():
    global loaded_model
    loaded_model = load_model()



@app.get(path="/", tags=["Health"])
def api_health():
    return {"status": "healthy"}


@app.post(path="/predict", tags=["Prediction"])
def predict(request : FetalHealthData):
    global loaded_model

    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)

    prediction = loaded_model.predict(received_data)

    print(prediction)

    return {"status": str(prediction[0])}



