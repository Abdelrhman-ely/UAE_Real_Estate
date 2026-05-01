import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
# Function to load all models and transformers once
def load_models():
    """Loads all models and preprocessors into a dictionary."""
    models = {}
    try:
        models['preprocessor']        = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
        models['price_scaler']         = joblib.load(os.path.join(MODELS_DIR, "price_scaler.pkl"))
        models['xgb_model']            = joblib.load(os.path.join(MODELS_DIR, "xgb_real_estate_model.pkl"))
        models['classifier']           = joblib.load(os.path.join(MODELS_DIR, "price_classifier_tuned.pkl"))
        models['optimal_threshold']    = joblib.load(os.path.join(MODELS_DIR, "optimal_threshold.pkl"))
        models['recommender']          = joblib.load(os.path.join(MODELS_DIR, "recommender_model.pkl"))
        models['properties_data']      = joblib.load(os.path.join(MODELS_DIR, "properties_data.pkl"))
        models['recommendation_matrix']= joblib.load(os.path.join(MODELS_DIR, "recommendation_matrix.pkl"))
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

# Load models once when the module is imported
loaded_models = load_models()

def predict_property_price(features_dict: dict) -> float:
    """Predicts the price of a property in AED."""
    try:
        df = pd.DataFrame([features_dict])
        log_price_pred = loaded_models['xgb_model'].predict(df)   # pass raw DataFrame directly
        final_price = float(np.expm1(log_price_pred[0]))
        return final_price
 
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")


def classify_property_market(features_dict: dict) -> dict:
    """Classifies whether a property is Overpriced or a Fair Deal."""
    try:
        df = pd.DataFrame([features_dict])
        probs = loaded_models['classifier'].predict_proba(df)[:, 1]
        prob  = float(probs[0])
        threshold = float(loaded_models['optimal_threshold'])   # 0.40 from notebook
        if prob >= threshold:
            label = "Overpriced"
        else:
            label = "Fair Deal"
 
        return {
            "label": label,
            "probability": round(prob, 4)
        }
 
    except Exception as e:
        raise ValueError(f"Classification failed: {str(e)}")

def get_property_recommendations(property_index: int, top_n: int = 5) -> list:
    """Gets the top-N most similar properties using cosine nearest neighbours."""
    try:
        recommender = loaded_models['recommender']          # NearestNeighbors object
        matrix      = loaded_models['recommendation_matrix']# numpy array (n_props × n_features)
        data        = loaded_models['properties_data']      # original DataFrame
 
        distances, indices = recommender.kneighbors(
            matrix[property_index].reshape(1, -1)
        )
        similar_indices    = indices[0][1: top_n + 1]           # skip self (index 0)
        similarity_scores  = 1 - distances[0][1: top_n + 1]    # cosine dist → similarity
 
        recommendations = data.iloc[similar_indices].copy()
        recommendations['Match_Score'] = [f"{round(s * 100, 1)}%" for s in similarity_scores]
 
        display_cols = ['Match_Score', 'price', 'community', 'bedrooms',
                        'bathrooms', 'sizeMin_sqft', 'furnishing']
        return recommendations[display_cols].to_dict(orient='records')
 
    except Exception as e:
        raise ValueError(f"Recommendation failed: {str(e)}")
 
 