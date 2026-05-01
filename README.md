# UAE Real Estate Intelligence System

This is an end-to-end Machine Learning system for analyzing UAE real estate data. It is designed with a production-ready architecture that includes a FastAPI backend and a Streamlit frontend for interactive usage.

---

## Overview

The system provides intelligent real estate analytics including:
- Property price prediction
- Market fairness classification (Fair Deal vs Overpriced)
- Similar property recommendations

It is built with scalability, modularity, and real-world deployment in mind.

---

## System Design

The project follows a clean, production-style architecture:

- **Separation of Concerns**: Machine learning logic, API layer, and UI are fully decoupled.
- **Backend Service**: FastAPI handles all inference requests asynchronously.
- **Input Validation**: Pydantic ensures strict data validation before model processing.
- **Frontend Interface**: Streamlit provides an interactive user experience connected via API calls.

---

## Model Performance

- Price Prediction Model: XGBoost Regressor (R² ≈ 0.90+)
- Classification Model: Overpriced vs Fair Deal detector
- Recommendation System: Cosine similarity-based nearest neighbors

The models were trained on a 2024 UAE real estate dataset with extensive feature engineering and preprocessing.

---

## Business Value

This system helps users and investors by:
- Identifying overpriced properties in the market.
- Estimating fair property values.
- Discovering similar investment opportunities.
- Supporting data-driven real estate decisions.

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic  
- **Frontend**: Streamlit, Requests  
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, NumPy, Joblib  
- **Language**: Python 3.10+  

---

## Project Structure

```text
app/              -> FastAPI backend (routes, schemas, inference logic)
streamlit_app/    -> Streamlit frontend interface
models/           -> Trained ML models (excluded from GitHub)
data/             -> Raw datasets (excluded from GitHub)
notebook/         -> Full training pipeline (EDA, feature engineering, modeling)
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Abdelrhman-ely/UAE_Real_Estate.git
cd UAE_Real_Estate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run FastAPI backend**
```bash
uvicorn app.main:app --reload
```

**4. Run Streamlit frontend**
Open a new terminal window and run:
```bash
streamlit run streamlit_app/1_Price_Prediction.py
```

---

## Final Result

This project demonstrates a production-grade machine learning system with real-world architecture, focusing on scalability, modularity, and business impact.
