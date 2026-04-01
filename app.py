from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")

# Your details
NAME = "SibyKanna"
ROLL_NO = "2022BCD0039"


@app.get("/")
def home():
    return {
        "message": "MLOps API is running",
        "Name": NAME,
        "Roll No": ROLL_NO
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "Name": NAME,
        "Roll No": ROLL_NO
    }


@app.post("/predict")
def predict(data: dict):
    try:
        # Expected input: Pclass, Sex, Age, Fare
        features = np.array([
            data["Pclass"],
            data["Sex"],
            data["Age"],
            data["Fare"]
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]

        return {
            "prediction": int(prediction),
            "Name": NAME,
            "Roll No": ROLL_NO
        }

    except Exception as e:
        return {"error": str(e)}