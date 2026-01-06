"""
Food Demand Forecasting - Prediction Service

FastAPI web service for serving food demand predictions.
Uses sklearn Pipeline with DictVectorizer for feature transformation.

Usage:
    uvicorn predict:app --host 0.0.0.0 --port 8080 --reload

Endpoints:
    - GET  /: Health check
    - POST /predict: Single prediction
    - POST /predict_batch: Batch predictions
"""

import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn


# Initialize FastAPI app
app = FastAPI(
    title="Food Demand Forecasting API",
    description="Predict meal demand for fulfillment centers using DictVectorizer Pipeline",
    version="2.0.0"
)


# Load model pipeline on startup
pipeline = None
feature_cols = None


@app.on_event("startup")
async def load_model():
    """
    Load trained pipeline when API starts
    """
    global pipeline, feature_cols

    try:
        with open('final_model.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        print("✓ Pipeline loaded successfully")
        print(f"  Pipeline steps: {list(pipeline.named_steps.keys())}")

        with open('feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        print(f"✓ Feature columns loaded successfully ({len(feature_cols)} features)")

        # Show pipeline info
        dict_vec = pipeline.named_steps['dict_vectorizer']
        print(f"  DictVectorizer features: {len(dict_vec.feature_names_)}")

    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        raise


# Request models
class PredictionRequest(BaseModel):
    """
    Single prediction request schema
    """
    week: int = Field(..., description="Week number (e.g., 146-155)", ge=1)
    center_id: int = Field(..., description="Fulfillment center ID")
    meal_id: int = Field(..., description="Meal ID")
    checkout_price: float = Field(..., description="Final checkout price", ge=0)
    base_price: float = Field(..., description="Base price of meal", ge=0)
    emailer_for_promotion: int = Field(..., description="Emailer sent (0 or 1)", ge=0, le=1)
    homepage_featured: int = Field(..., description="Featured on homepage (0 or 1)", ge=0, le=1)
    city_code: int = Field(..., description="City code")
    region_code: int = Field(..., description="Region code")
    center_type: str = Field(..., description="Center type (TYPE_A, TYPE_B, etc.)")
    op_area: float = Field(..., description="Operating area in km²", ge=0)
    category: str = Field(..., description="Meal category")
    cuisine: str = Field(..., description="Meal cuisine")

    class Config:
        schema_extra = {
            "example": {
                "week": 146,
                "center_id": 55,
                "meal_id": 1885,
                "checkout_price": 136.83,
                "base_price": 152.0,
                "emailer_for_promotion": 0,
                "homepage_featured": 0,
                "city_code": 647,
                "region_code": 56,
                "center_type": "TYPE_A",
                "op_area": 3.5,
                "category": "Beverages",
                "cuisine": "Thai"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request schema
    """
    predictions: List[PredictionRequest]


class PredictionResponse(BaseModel):
    """
    Prediction response schema
    """
    predicted_orders: float = Field(..., description="Predicted number of orders")
    week: int
    center_id: int
    meal_id: int


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response schema
    """
    predictions: List[PredictionResponse]
    count: int


def create_features(data: dict) -> dict:
    """
    Apply feature engineering to input data

    Args:
        data: Input dictionary with raw features

    Returns:
        dict: Dictionary with engineered features
    """
    # Create a copy
    features = data.copy()

    # Price-based features
    features['discount'] = features['base_price'] - features['checkout_price']
    features['discount_percentage'] = (features['discount'] / features['base_price']) * 100
    if pd.isna(features['discount_percentage']):
        features['discount_percentage'] = 0.0

    # Promotional features
    features['total_promotion'] = features['emailer_for_promotion'] + features['homepage_featured']

    # Time-based cyclical features
    features['week_mod_4'] = features['week'] % 4
    features['week_mod_13'] = features['week'] % 13
    features['week_mod_52'] = features['week'] % 52

    # Convert categorical variables to strings (as expected by DictVectorizer)
    features['center_id'] = str(features['center_id'])
    features['meal_id'] = str(features['meal_id'])
    features['city_code'] = str(features['city_code'])
    features['region_code'] = str(features['region_code'])
    features['center_type'] = str(features['center_type'])
    features['category'] = str(features['category'])
    features['cuisine'] = str(features['cuisine'])

    return features


@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Food Demand Forecasting API",
        "version": "2.0.0",
        "pipeline_type": "DictVectorizer + RandomForestRegressor",
        "model_loaded": pipeline is not None
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    dict_vec = pipeline.named_steps['dict_vectorizer']

    return {
        "status": "healthy",
        "pipeline": "loaded",
        "pipeline_steps": list(pipeline.named_steps.keys()),
        "input_features": len(feature_cols) if feature_cols else 0,
        "vectorized_features": len(dict_vec.feature_names_)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction

    Args:
        request: Prediction request with meal and center details

    Returns:
        PredictionResponse: Predicted number of orders
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        # Convert request to dictionary
        data = request.dict()

        # Apply feature engineering
        features = create_features(data)

        # Make prediction using pipeline (expects list of dicts)
        prediction = pipeline.predict([features])[0]
        prediction = max(0, float(prediction))  # Ensure non-negative

        return PredictionResponse(
            predicted_orders=round(prediction),
            week=request.week,
            center_id=request.center_id,
            meal_id=request.meal_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions

    Args:
        request: Batch prediction request

    Returns:
        BatchPredictionResponse: List of predictions
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        predictions = []

        # Process each item
        for item in request.predictions:
            # Convert request to dictionary
            data = item.dict()

            # Apply feature engineering
            features = create_features(data)

            # Make prediction
            prediction = pipeline.predict([features])[0]
            prediction = max(0, float(prediction))

            predictions.append(
                PredictionResponse(
                    predicted_orders=round(prediction),
                    week=item.week,
                    center_id=item.center_id,
                    meal_id=item.meal_id
                )
            )

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/predict_csv")
async def predict_from_data(data: List[dict]):
    """
    Make predictions from raw data (useful for CSV uploads)

    Args:
        data: List of dictionaries containing feature data

    Returns:
        List of predictions
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        # Apply feature engineering to all items
        features_list = [create_features(item) for item in data]

        # Make predictions using pipeline
        predictions = pipeline.predict(features_list)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative

        # Create response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "index": i,
                "week": int(data[i].get('week')) if 'week' in data[i] else None,
                "center_id": int(data[i].get('center_id')) if 'center_id' in data[i] else None,
                "meal_id": int(data[i].get('meal_id')) if 'meal_id' in data[i] else None,
                "predicted_orders": round(float(pred))
            })

        return {
            "predictions": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
