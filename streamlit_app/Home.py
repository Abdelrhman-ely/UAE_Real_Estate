import streamlit as st
from utils import health_check

# Page Configuration 
st.set_page_config(
    page_title="UAE Real Estate AI",
    layout="wide",
)

# Custom CSS for Calm & Relaxing Colors 
st.markdown("""
<style>
    /* Soft text colors for headers */
    .subtitle-text {
        color: #64748B; 
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Pastel cards for the overview section */
    .pastel-card {
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        color: #334155;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Soft Blue for Prediction */
    .card-blue { 
        background-color: #F0F9FF; 
        border-left: 5px solid #7DD3FC; 
    }
    
    /* Soft Yellow for Classification */
    .card-yellow { 
        background-color: #FEFCE8; 
        border-left: 5px solid #FDE047; 
    }
    
    /* Soft Green for Recommendations */
    .card-green { 
        background-color: #F0FDF4; 
        border-left: 5px solid #86EFAC; 
    }
    
    h4 {
        margin-top: 0;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# Header 
st.title("UAE Real Estate Intelligence System")
st.markdown("<div class='subtitle-text'>Welcome! Please select a service from the sidebar to get started.</div>", unsafe_allow_html=True)

st.divider()

# ── API Status ───────────────────────────────────────────────
st.subheader("API Status")

try:
    data = health_check()
    status = data.get("status", "unknown")

    if status == "healthy":
        st.success("**API is Online** — All models loaded successfully")
    else:
        st.warning("**API is Online** — Some models are missing")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Loaded:**")
        loaded_models = data.get("models_loaded", [])
        if loaded_models:
            for m in loaded_models:
                st.markdown(f"- `{m}`")
        else:
            st.markdown("- *None*")
            
    with col2:
        st.markdown("**Missing:**")
        missing_models = data.get("models_missing", [])
        if missing_models:
            for m in missing_models:
                st.markdown(f"- `{m}`")
        else:
            st.markdown("- *None*")

except Exception as e:
    st.error(f"**API is Offline** — {e}")
    st.info("Make sure your FastAPI server is running on `http://127.0.0.1:8000`")

st.divider()

# ── Pages Overview ───────────────────────────────────────────
st.subheader("Available Pages")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="pastel-card card-blue">
        <h4>Price Prediction</h4>
        <p>Enter property details to get an accurate estimated price in AED.</p>
    </div>
    """, unsafe_allow_html=True)
    
with c2:
    st.markdown("""
    <div class="pastel-card card-yellow">
        <h4>Market Classification</h4>
        <p>Find out if a property is fairly priced or overvalued based on the market.</p>
    </div>
    """, unsafe_allow_html=True)
    
with c3:
    st.markdown("""
    <div class="pastel-card card-green">
        <h4>Recommendations</h4>
        <p>Discover the top 5 properties most similar to a specific listing.</p>
    </div>
    """, unsafe_allow_html=True)