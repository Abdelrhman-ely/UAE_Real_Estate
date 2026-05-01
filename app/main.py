import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PropertyFeatures,
    RecommendationRequest,
    PricePredictionResponse,
    PriceClassificationResponse,
    RecommendationResponse,
    RecommendedProperty,
    HealthResponse,
)
from .utils import (
    loaded_models,
    predict_property_price,
    classify_property_market,
    get_property_recommendations,
)

# Constants
AED_TO_USD = 0.272

REQUIRED_MODELS = [
    "preprocessor", "price_scaler", "xgb_model",
    "classifier", "optimal_threshold",
    "recommender", "properties_data", "recommendation_matrix",
]


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [k for k in REQUIRED_MODELS if k not in loaded_models]
    if missing:
        print(f"Missing models on startup: {missing}")
    else:
        print("All models ready.")
    yield
    print("Shutting down.")


# App
app = FastAPI(
    title="UAE Real Estate Intelligence API",
    description=(
        "AI-powered real estate system for the UAE market.\n\n"
        "| Endpoint | What it does |\n"
        "|---|---|\n"
        "| `POST /predict/price` | XGBoost price regression (AED) |\n"
        "| `POST /predict/classify` | Overpriced vs Fair Deal |\n"
        "| `POST /recommend` | Top-N similar properties |\n"
        "| `GET  /health` | Model availability check |"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# make api work with all frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helpers
def _require(*keys: str) -> None:
    """Raise 503 if any required model is not loaded."""
    missing = [k for k in keys if k not in loaded_models]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Model(s) not loaded: {missing}. Check /health for details.",
        )


# Routes
@app.get("/", tags=["System"])
def root():
    return {
        "message": "UAE Real Estate Intelligence API",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Returns which models are loaded and ready."""
    loaded  = [k for k in REQUIRED_MODELS if k in loaded_models]
    missing = [k for k in REQUIRED_MODELS if k not in loaded_models]
    return HealthResponse(
        status="healthy" if not missing else "degraded",
        models_loaded=loaded,
        models_missing=missing,
    )


@app.post("/predict/price", response_model=PricePredictionResponse, tags=["Prediction"])
def predict_price(features: PropertyFeatures):
    """
    Predict the **sale price** of a UAE property in AED.

    - Model: XGBoost wrapped in `TransformedTargetRegressor`
    - The pipeline preprocesses the input internally — no manual transformation needed.
    - Output is inverse-transformed from log-space back to raw AED.
    """
    _require("xgb_model")

    try:
        price_aed = predict_property_price(features.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PricePredictionResponse(
        predicted_price_aed=round(price_aed, 2),
        predicted_price_usd=round(price_aed * AED_TO_USD, 2),
        price_per_sqft_aed =round(price_aed / features.sizeMin_sqft, 2),
    )


@app.post("/predict/classify", response_model=PriceClassificationResponse, tags=["Prediction"])
def classify_price(features: PropertyFeatures):
    """
    Classify whether a property is **Overpriced** or a **Fair / Good Deal**.

    - Model: XGBoost Classifier trained on pricing residuals
    - Threshold: **0.40** (tuned for high Recall — buyer-safe)
    - `probability` = P(Overpriced). Values >= 0.40 flagged as Overpriced.
    """
    _require("classifier", "optimal_threshold")

    try:
        result = classify_property_market(features.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PriceClassificationResponse(
        label=result["label"],
        probability=result["probability"],
        threshold=float(loaded_models["optimal_threshold"]),
    )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
def recommend(request: RecommendationRequest):
    """
    Return the **top-N most similar properties** to a reference property.

    - Model: `NearestNeighbors` with cosine similarity
    - Matrix: preprocessed features + RobustScaler-scaled price
    - `property_index` is the **row index** (0-based) in the original dataset.
    """
    _require("recommender", "recommendation_matrix", "properties_data")

    n_properties = len(loaded_models["properties_data"])
    if request.property_index >= n_properties:
        raise HTTPException(
            status_code=404,
            detail=f"property_index {request.property_index} out of range. "
                   f"Dataset has {n_properties} properties (0 to {n_properties - 1}).",
        )

    try:
        raw = get_property_recommendations(request.property_index, request.top_n)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    recommendations = [RecommendedProperty(**r) for r in raw]

    return RecommendationResponse(
        reference_property_index=request.property_index,
        top_n=request.top_n,
        recommendations=recommendations,
    )