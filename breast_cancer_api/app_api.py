from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Initializa app

app = FastAPI(title = "Breast Cancer Prediction API")


# We will load our model:

model = joblib.load("model_log.pkl")


# Define what kind of data will API expects:


class CancerInput(BaseModel):

    mean_radius: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float


@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running  🚀"}


@app.post("/predict")
def predict(data: CancerInput):
    input_data = np.array([[data.mean_radius, data.texture_mean, data.perimeter_mean, 
                            data.area_mean, data.smoothness_mean, data.compactness_mean, 
                            data.concavity_mean, data.concave_points_mean, 
                            data.symmetry_mean, data.fractal_dimension_mean]])
    

    # predict part

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # return result 

    return {
        "Prediction": "Malignant" if pred==1 else "Benign!", 
        "Probability of tumor": prob
    }
