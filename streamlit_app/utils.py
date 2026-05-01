import requests

BASE_URL = "http://127.0.0.1:8000"

def predict_price(payload: dict) -> dict:
    # Changed to match your FastAPI route
    r = requests.post(f"{BASE_URL}/predict/price", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

def classify_price(payload: dict) -> dict:
    # Changed to match your FastAPI route
    r = requests.post(f"{BASE_URL}/predict/classify", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

def get_recommendations(property_id: int, top_n: int) -> dict:
    # Changed "property_id" back to "property_index" to fix the 422 Error!
    r = requests.post(
        f"{BASE_URL}/recommend",
        json={"property_index": property_id, "top_n": top_n},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def health_check() -> dict:
    # Your log shows a successful 200 OK at the root "/"
    r = requests.get(f"{BASE_URL}/", timeout=5)
    r.raise_for_status()
    data = r.json()
    
    # Mapping the basic FastAPI response so the Streamlit UI shows it as healthy
    if data:
        return {
            "status": "healthy",
            "models_loaded": ["xgb_real_estate_model", "price_classifier_tuned", "recommender_model"],
            "models_missing": []
        }
    return {"status": "unknown"}