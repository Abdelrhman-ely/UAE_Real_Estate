import streamlit as st
from utils import classify_price

st.set_page_config(page_title="Price Classification", layout="wide")

# Custom CSS for Calm & Relaxing Colors 
st.markdown("""
<style>
    .subtitle-text {
        color: #64748B; 
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    .pastel-card {
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        color: #334155;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .card-green { 
        background-color: #F0FDF4; 
        border-left: 5px solid #86EFAC; 
    }
    .card-red { 
        background-color: #FEF2F2; 
        border-left: 5px solid #FCA5A5; 
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Price Classification")
st.markdown("<div class='subtitle-text'>Find out if a property is an <b>Overpriced</b> or a <b>Fair Deal</b> based on its features.</div>", unsafe_allow_html=True)
st.divider()

# Input Form
with st.form("classify_form"):
    col1, col2 = st.columns(2)

    with col1:
        bedrooms  = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1, help="0 = Studio")
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)
        size      = st.number_input("Size (sqft)", min_value=100.0, max_value=50000.0, value=1500.0, step=50.0)

    with col2:
        community  = st.text_input("Community", value="Dubai Marina", help="Example: Dubai Marina, Downtown Dubai, Jumeirah")
        furnishing = st.selectbox("Furnishing", options=["YES", "PARTLY", "NO"], help="YES = Fully Furnished | PARTLY = Partially Furnished | NO = Unfurnished")

    submitted = st.form_submit_button("Classify Property", type="primary", use_container_width=True)

# Result
if submitted:
    payload = {
        "bedrooms":     float(bedrooms),
        "bathrooms":    float(bathrooms),
        "sizeMin_sqft": float(size),
        "community":    community.strip(),
        "furnishing":   furnishing,
    }

    with st.spinner("Analyzing market classification..."):
        try:
            result = classify_price(payload)

            st.divider()
            st.subheader("Classification Results")

            # Graceful extraction in case the API returns slightly different keys
            market_class = result.get("market_class", -1)
            # Default to checking market_class if 'label' isn't explicitly in the response
            label = result.get("label", "Overpriced" if market_class == 1 else "Fair Deal")
            prob  = result.get("probability", 0.0)
            thr   = result.get("threshold", 0.5) # Default to 50% if threshold isn't provided

            # Pastel UI Feedback based on label
            if label == "Overpriced":
                st.markdown(f"""
                <div class="pastel-card card-red">
                    <h3 style="margin-top: 0; color: #991B1B;"> {label}</h3>
                    <p style="margin-bottom: 0;">This property is likely priced above the current market average.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pastel-card card-green">
                    <h3 style="margin-top: 0; color: #166534;"> {label}</h3>
                    <p style="margin-bottom: 0;">This property is fairly priced according to current market trends.</p>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            col1.metric("P(Overpriced)", f"{prob:.1%}")
            col2.metric("Model Threshold", f"{thr:.0%}")

            # Probability bar
            st.markdown("**Model Confidence Level:**")
            # Ensure the probability stays between 0.0 and 1.0 to avoid Streamlit errors
            st.progress(min(max(prob, 0.0), 1.0))

            note = result.get("note", "")
            if note:
                st.caption(note)

        except Exception as e:
            st.error(f"Error communicating with the API: {e}")