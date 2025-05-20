from fastapi import FastAPI
from api.request_schema import PredictRequest
from api.model_loader import predict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Support Ticket Classifier API"}

@app.post("/predict")
def classify(req: PredictRequest):
    return predict(req.text)
