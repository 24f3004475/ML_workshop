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
    probability: float


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
    prob = model.predict_proba([request.comment]).max()
    return PredictionResponse(prediction=float(pred), probability=float(prob))

@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comment Category Prediction</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 750px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f7fa;
                color: #222;
            }

            .container {
                background: white;
                padding: 35px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            }

            h1 {
                text-align: center;
                margin-bottom: 10px;
            }

            .description {
                text-align: center;
                color: #555;
                line-height: 1.6;
                margin-bottom: 25px;
            }

            .model-info {
                background: #f0f4f8;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 25px;
                font-size: 14px;
                color: #444;
            }

            textarea {
                width: 100%;
                box-sizing: border-box;
                font-size: 16px;
                padding: 12px;
                margin: 10px 0 15px;
                border: 1px solid #ccc;
                border-radius: 6px;
                resize: vertical;
            }

            button {
                width: 100%;
                padding: 12px;
                background: #0078d7;
                color: white;
                font-size: 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }

            button:hover {
                background: #005fa3;
            }

            .result {
                margin-top: 20px;
                padding: 15px;
                background: #eef2ff;
                border-radius: 8px;
                display: none;
            }

            .result strong {
                font-size: 18px;
            }

            .error {
                color: #b00020;
            }

            .footer {
                text-align: center;
                margin-top: 25px;
                font-size: 13px;
                color: #777;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <h1>Comment Category Predictor</h1>

            <p class="description">
                This application uses a machine learning model to classify
                comments into one of four categories represented by labels
                0, 1, 2, and 3 based on its text.
            </p>

            <div class="model-info">
                <strong>How it works:</strong><br>
                The comment is converted into TF-IDF features and passed
                to a LightGBM classification model to predict its category.
            </div>

            <label for="comment">
                <strong>Enter a comment</strong>
            </label>

            <textarea
                id="comment"
                rows="5"
                placeholder="Type or paste a comment here..."
            ></textarea>

            <button onclick="predict()">Predict Category</button>

            <div id="output" class="result"></div>

            <div class="footer">
                Comment Category Prediction • FastAPI + TF-IDF + LightGBM
            </div>

        </div>

        <script>
            async function predict() {

                const comment =
                    document.getElementById("comment").value.trim();

                const output =
                    document.getElementById("output");

                if (!comment) {
                    output.style.display = "block";
                    output.innerHTML =
                        '<span class="error">Please enter a comment.</span>';
                    return;
                }

                output.style.display = "block";
                output.innerHTML = "Predicting...";

                try {

                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            comment: comment
                        })
                    });

                    const data = await response.json();

                    if (response.ok) {

                        const percentage =
                            (data.probability * 100).toFixed(2);

                        output.innerHTML = `
                            <strong>Predicted Category: ${data.prediction}</strong>
                            <br><br>
                            Confidence: ${percentage}%
                        `;

                    } else {

                        output.innerHTML =
                            `<span class="error">
                                Error: ${data.detail || "Prediction failed."}
                            </span>`;
                    }

                } catch (error) {

                    output.innerHTML =
                        `<span class="error">
                            Unable to connect to the prediction API.
                        </span>`;
                }
            }
        </script>

    </body>
    </html>
    """