from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Comment Category Prediction API",
    description="A minimal prediction API built for the workshop.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

class PredictionRequest(BaseModel):
    comment: str


class PredictionResponse(BaseModel):
    prediction: int | float


@app.get("/")
def root():
    return {"message": "Workshop ML API is running. See /docs for usage."}


@app.get("/health")
def health():
    """Basic health check endpoint — useful for deployment platforms & load balancers."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Did you run the training notebook?")

    pred = model.predict([request.comment])[0]
    return PredictionResponse(prediction=float(pred))

@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comment Category Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; background:#f4f6f8; }
            h2 { text-align: center; }
            textarea { width: 100%; font-size: 16px; padding: 10px; margin: 10px 0; }
            button { width: 100%; padding: 10px; background: #0078d7; color: white; border: none; border-radius: 4px; }
            button:hover { background: #005a9e; }
            .result { margin-top: 15px; padding: 10px; background: #eef; border-radius: 4px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>Comment Category Prediction</h2>
        <textarea id="comment" rows="4" placeholder="Enter a comment..."></textarea>
        <button onclick="predict()">Predict</button>
        <div id="output" class="result"></div>

        <script>
            async function predict() {
                const comment = document.getElementById("comment").value;

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ comment })
                });

                if (response.ok) {
                    const data = await response.json();
                    document.getElementById("output").innerText =
                        `Prediction (number): ${data.prediction}\\nProbability: ${data.probability}`;
                } else {
                    document.getElementById("output").innerText = "Error: " + response.statusText;
                }
            }
        </script>
    </body>
    </html>
    """