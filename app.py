import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import uvicorn
import logging

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_PATH = "house_prices.csv"
TARGET_COLUMN = "Price"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real Estate Price Prediction API")

model = None
feature_columns = None

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
def train_and_register_model(df):
    global model, feature_columns

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Drop ID column
    if "Property_ID" in X.columns:
        X = X.drop("Property_ID", axis=1)

    # One-hot encoding
    X = pd.get_dummies(
        X,
        columns=["Location", "Property_Type"],
        drop_first=True
    )

    feature_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Model trained | MAE: {mae:.2f} | R2: {r2:.2f}")

# --------------------------------------------------
# FASTAPI STARTUP
# --------------------------------------------------
@app.on_event("startup")
def startup_event():
    df = load_data()
    train_and_register_model(df)

# --------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------
class PredictionInput(BaseModel):
    Area: float
    Bedrooms: int
    Bathrooms: int
    Age: int
    Location: str
    Property_Type: str

# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------
@app.post("/predict")
def predict_price(data: PredictionInput):
    input_df = pd.DataFrame([data.dict()])

    input_df = pd.get_dummies(
        input_df,
        columns=["Location", "Property_Type"],
        drop_first=True
    )

    # Align columns with training data
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }

# --------------------------------------------------
# RUN APP
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

