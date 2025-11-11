# Estudo 01 fastAPI

import os
import json
import mlflow
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI


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
    #MLFLOW_TRACKING_URI = 'https://dagshub.com/rogeriotbs/mlops-ead-puc.mlflow'
    MLFLOW_TRACKING_URI = 'https://dagshub.com/renansantosmendes/puc_lectures_mlops.mlflow'
    #MLFLOW_TRACKING_USERNAME = 'rogeriotbs'
    MLFLOW_TRACKING_USERNAME = 'renansantosmendes'
    #MLFLOW_TRACKING_PASSWORD = 'abe754107b721364b6f587be1d0d3366ebdf30dd'
    MLFLOW_TRACKING_PASSWORD = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'

    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    registered_model = client.get_registered_model('fetal_health')
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



