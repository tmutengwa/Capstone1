"""
Test script for Food Demand Forecasting API

Usage:
    python test_api.py
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def test_single_prediction():
    """Test single prediction endpoint"""
    print("Testing single prediction...")

    payload = {
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

    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("Testing batch prediction...")

    payload = {
        "predictions": [
            {
                "week": 146,
                "center_id": 55,
                "meal_id": 1885,
                "checkout_price": 158.11,
                "base_price": 159.11,
                "emailer_for_promotion": 0,
                "homepage_featured": 0,
                "city_code": 647,
                "region_code": 56,
                "center_type": "TYPE_C",
                "op_area": 2.0,
                "category": "Beverages",
                "cuisine": "Thai"
            },
            {
                "week": 146,
                "center_id": 55,
                "meal_id": 1993,
                "checkout_price": 160.11,
                "base_price": 159.11,
                "emailer_for_promotion": 0,
                "homepage_featured": 0,
                "city_code": 647,
                "region_code": 56,
                "center_type": "TYPE_C",
                "op_area": 2.0,
                "category": "Beverages",
                "cuisine": "Thai"
            }
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/predict_batch", json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()


if __name__ == "__main__":
    test_health_check()
    test_single_prediction()
    test_batch_prediction()
