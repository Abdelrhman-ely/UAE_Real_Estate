from pydantic import BaseModel, Field
from typing import Optional
 
 
class PropertyFeatures(BaseModel):
    """
    Input features that match the exact column names used during model training.
    """
    bedrooms:     float = Field(..., ge=0, example=3,
                                description="Number of bedrooms (0 = Studio)")
    bathrooms:    float = Field(..., ge=0, example=2,
                                description="Number of bathrooms")
    sizeMin_sqft: float = Field(..., gt=0, example=1500.0,
                                description="Property size in square feet")
    community:    str   = Field(..., example="Dubai Marina",
                                description="Main community / area extracted from address")
    furnishing:   str   = Field(..., example="YES",
                                description="Furnishing status: YES | PARTLY | NO")
 
    model_config = {
        "json_schema_extra": {
            "example": {
                "bedrooms":     3,
                "bathrooms":    2,
                "sizeMin_sqft": 1500.0,
                "community":    "Dubai Marina",
                "furnishing":   "YES"
            }
        }
    }
 
 
class RecommendationRequest(BaseModel):
    property_index: int = Field(..., ge=0, example=0,
                                description="Row index of the property in the dataset (0-based)")
    top_n:          int = Field(5, ge=1, le=20,
                                description="Number of similar properties to return")
 
    model_config = {
        "json_schema_extra": {
            "example": {"property_index": 0, "top_n": 5}
        }
    }
 
 
# RESPONSE SCHEMAS
 
class PricePredictionResponse(BaseModel):
    predicted_price_aed: float = Field(..., description="Predicted price in AED")
    predicted_price_usd: float = Field(..., description="Predicted price in USD (1 AED = 0.272 USD)")
    price_per_sqft_aed:  float = Field(..., description="Price per sqft in AED") 
 
class PriceClassificationResponse(BaseModel):
    label:       str   = Field(..., description="'Overpriced' or 'Fair Deal'")
    probability: float = Field(..., description="P(Overpriced) — higher = more likely overpriced")
    threshold:   float = Field(..., description="Decision threshold used (0.40)")
    note:        str   = Field(
        default="Optimised for Recall (buyer-safe). High recall = fewer missed overpriced deals.",
        description="Business context for the threshold choice"
    )
 
 
class RecommendedProperty(BaseModel):
    Match_Score:  str            = Field(..., example="96.3%")
    price:        float          = Field(..., example=1_800_000)
    community:    str            = Field(..., example="Dubai Marina")
    bedrooms:     float          = Field(..., example=3)
    bathrooms:    float          = Field(..., example=2)
    sizeMin_sqft: float          = Field(..., example=1480.0)
    furnishing:   str            = Field(..., example="YES")
 
 
class RecommendationResponse(BaseModel):
    reference_property_index: int
    top_n:                    int
    recommendations:          list[RecommendedProperty]
 
 
class HealthResponse(BaseModel):
    status:         str       = Field(..., example="healthy")
    models_loaded:  list[str]
    models_missing: list[str]