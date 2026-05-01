import streamlit as st
from utils import predict_price

st.set_page_config(page_title="Price Prediction", layout="wide")

# Custom CSS for Calm & Relaxing Colors
st.markdown("""
<style>
    .subtitle-text {
        color: #64748B; 
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    .result-card {
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        background-color: #F0FDF4; 
        border-left: 5px solid #86EFAC;
        color: #334155;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header 
st.title("Price Prediction")
st.markdown("<div class='subtitle-text'>Enter the property details below, and the AI model will estimate its price.</div>", unsafe_allow_html=True)
st.divider()

# Input Form 
with st.form("price_form"):
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1,
                                   help="0 = Studio")
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)
        size = st.number_input("Size (sqft)", min_value=100.0, max_value=50000.0,
                               value=1500.0, step=50.0)

    with col2:
        community = st.text_input("Community", value="Dubai Marina",
                                  help="Example: Dubai Marina, Downtown Dubai, Jumeirah")
        furnishing = st.selectbox("Furnishing", options=["YES", "PARTLY", "NO"],
                                  help="YES = Fully Furnished | PARTLY = Partially Furnished | NO = Unfurnished")

    # Use a visually distinct submit button
    submitted = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

# Result
if submitted:
    # Payload matching the exact features expected by your Notebook/FastAPI
    payload = {
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "sizeMin_sqft": float(size),
        "community": community.strip(),
        "furnishing": furnishing,
    }

    with st.spinner("Analyzing market data..."):
        try:
            result = predict_price(payload)
            
            # Safely extract the price (handling both our previous FastAPI structure and your custom keys)
            predicted_aed = result.get('predicted_price_aed') or result.get('predicted_price', 0)
            
            # Calculate USD and Price/Sqft locally if the API doesn't return them directly
            predicted_usd = result.get('predicted_price_usd', predicted_aed / 3.6725)
            price_per_sqft = result.get('price_per_sqft_aed', predicted_aed / size)
            model_used = result.get('model', 'XGBoost Regressor')

            st.divider()
            st.subheader("Prediction Results")

            # Display results in columns
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated Price (AED)", f"{predicted_aed:,.0f} AED")
            c2.metric("Estimated Price (USD)", f"${predicted_usd:,.0f}")
            c3.metric("Price per Sqft (AED)", f"{price_per_sqft:,.1f} AED")

            # Soft green success card for a calm UI
            st.markdown(f"""
            <div class="result-card">
                <strong>Model Information:</strong> Prediction generated successfully using the <code>{model_used}</code> model based on current UAE real estate trends.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error communicating with the API: {e}")