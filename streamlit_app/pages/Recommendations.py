import streamlit as st
import pandas as pd
from utils import get_recommendations

st.set_page_config(page_title="Recommendations",  layout="wide")

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
        padding: 15px;
        margin: 10px 0;
        background-color: #F0FDF4; 
        border-left: 5px solid #86EFAC;
        color: #334155;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("Property Recommendations")
st.markdown("<div class='subtitle-text'>Enter a Property ID to find the most similar listings currently available in the database.</div>", unsafe_allow_html=True)
st.divider()

# Input Form
with st.form("rec_form"):
    col1, col2 = st.columns(2)

    with col1:
        property_id = st.number_input(
            "🔢 Property ID (Index)",
            min_value=0, value=0, step=1,
            help="The row index in the dataset — starting from 0"
        )
    with col2:
        top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    submitted = st.form_submit_button("Find Similar Properties", type="primary", use_container_width=True)

# Result
if submitted:
    with st.spinner("Searching for similar properties..."):
        try:
            result = get_recommendations(int(property_id), int(top_n))

            recs = result.get("recommendations", [])

            if not recs:
                st.warning("No recommendations found.")
            else:
                st.divider()
                
                st.markdown(f"""
                <div class="pastel-card">
                    <h3 style="margin-top: 0; color: #166534;"> Top {len(recs)} Matches for Property ID: {property_id}</h3>
                </div>
                """, unsafe_allow_html=True)

                df = pd.DataFrame(recs)

                # Rename columns for a clean English display
                df = df.rename(columns={
                    "Match_Score":  "Similarity Score",
                    "price":        "Price (AED)",
                    "community":    "Community",
                    "bedrooms":     "Bedrooms",
                    "bathrooms":    "Bathrooms",
                    "sizeMin_sqft": "Size (sqft)",
                    "furnishing":   "Furnishing",
                })

                # Safely format price column
                if "Price (AED)" in df.columns:
                    def format_price(x):
                        try:
                            return f"{float(x):,.0f}" if pd.notnull(x) else x
                        except ValueError:
                            return x
                    df["Price (AED)"] = df["Price (AED)"].apply(format_price)
                
                # Safely format Match Score
                if "Similarity Score" in df.columns:
                    def format_score(x):
                        if pd.isnull(x):
                            return x
                        # If it's already a string with a '%', leave it alone
                        if isinstance(x, str) and '%' in x:
                            return x
                        # Otherwise, try to format it as a percentage
                        try:
                            return f"{float(x):.1%}"
                        except ValueError:
                            return x
                            
                    df["Similarity Score"] = df["Similarity Score"].apply(format_score)

                st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error communicating with the API: {e}")
            if "404" in str(e) or "400" in str(e):
                st.info("Ensure the Property ID exists in the dataset and the Recommender model is loaded.")