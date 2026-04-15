import streamlit as st
import numpy as np
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Optimized Model Loading (Cached for speed)
@st.cache_resource
def load_models():
    model = pickle.load(open("crop_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, scaler, label_encoder

model, scaler, label_encoder = load_models()

# 3. Sidebar - Input Parameters (Matching your SS)
with st.sidebar:
    st.markdown("## 🌱 Soil Parameters")
    n = st.slider("Nitrogen (N)", 0, 150, 31, help="Amount of Nitrogen in soil")
    p = st.slider("Phosphorus (P)", 0, 150, 30, help="Amount of Phosphorus in soil")
    k = st.slider("Potassium (K)", 0, 250, 49, help="Amount of Potassium in soil")
    ph = st.slider("pH value", 0.0, 14.0, 14.0, step=0.1)

    st.markdown("---") # Visual separator
    
    st.markdown("## ☁️ Climate Parameters")
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 50.0, step=0.1)
    hum = st.slider("Humidity (%)", 0.0, 100.0, 98.79, step=0.01)
    rain = st.slider("Rainfall (mm)", 0.0, 300.0, 272.18, step=0.01)

# 4. Main Page Header
st.markdown("<h1 style='text-align: center; color: white;'>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Machine Learning based crop suggestion using Soil & Climate data</p>", unsafe_allow_html=True)
st.divider()

# 5. The Dashboard Grid (2 Columns)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📊 Entered Values Overview")
    # Displaying values as big numbers like the screenshot
    st.write("Temperature (°C)")
    st.title(f"{temp}")
    
    st.write("Humidity (%)")
    st.title(f"{hum}")
    
    st.write("Rainfall (mm)")
    st.title(f"{rain}")

with col2:
    st.markdown("### 🧪 Soil Health")
    # Using metrics for a professional dashboard look
    st.metric(label="Nitrogen (N)", value=n)
    st.metric(label="Phosphorus (P)", value=p)
    st.metric(label="Potassium (K)", value=k)
    st.metric(label="pH value", value=ph)

st.divider()

# 6. Prediction Logic
if st.button("Recommend Crop", use_container_width=True, type="primary"):
    # Create input array
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    
    # Apply Scaling
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)
    crop = label_encoder.inverse_transform(prediction)[0]
    
    # Result UI
    st.balloons()
    st.success(f"### ✨ Recommended Crop: **{crop.upper()}**")
    
    # Added a simple tip based on the result
    st.info(f"The soil and climate conditions provided are ideal for growing **{crop}**.")
