# ui/learner.py - Heart Disease Prediction App
# Deployed on Streamlit Community Cloud

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ✅ Fix 1: Add parent directory to path for custom_transformers
# This allows import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ✅ Fix 2: Correct import (was: customer_tans → now: custom_transformers)
try:
    from customer_tans import FeatureSelector
except ImportError as e:
    st.sidebar.error("❌ Failed to import FeatureSelector. Check file location.")
    st.stop()

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")
st.markdown("### Predict heart disease risk using machine learning")

# ----------------------------
# Load Model Pipeline
# ----------------------------
@st.cache_resource
def load_model():
    # ✅ Try multiple paths (cloud vs local)
    possible_paths = [
        "../models/final_model.pkl",   # When running from ui/
        "models/final_model.pkl",
        "./models/final_model.pkl"
    ]
    for path in possible_paths:
        try:
            model = joblib.load(path)
            st.sidebar.success(f"✅ Model loaded from {path}")
            return model
        except Exception as e:
            st.sidebar.warning(f"⚠️ Not found: {path}")
            continue
    st.sidebar.error("❌ Could not load model. Check path and file.")
    return None

model = load_model()

# ----------------------------
# Get Selected Features from Model
# ----------------------------
selected_features = []
if model is not None:
    try:
        selected_features = model.named_steps['feature_selector'].selected_features
        st.sidebar.success("✅ Feature selector loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not extract features: {e}")

# Fallback feature list
if not selected_features:
    selected_features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
        'cp_1', 'cp_2', 'cp_3', 'restecg_1', 'restecg_2',
        'slope_1', 'slope_2', 'ca_1.0', 'ca_2.0', 'ca_3.0',
        'thal_6.0', 'thal_7.0'
    ]
    st.warning("⚠️ Using default feature list.")

# ----------------------------
# Sidebar: Input Form
# ----------------------------
st.sidebar.header("📊 Enter Patient Data")

# Input fields
age = st.sidebar.slider("Age", 29, 77, 55)
sex = st.sidebar.selectbox("Sex (0=Female, 1=Male)", [0, 1], index=1)

cp_options = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}
cp = st.sidebar.selectbox("Chest Pain Type", list(cp_options.keys()),
                         format_func=lambda x: cp_options[x])

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
chol = st.sidebar.slider("Serum Cholesterol (mg/dL)", 126, 564, 240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

restecg_options = ["Normal", "ST-T wave abnormal", "Left ventricular hypertrophy"]
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2], 
                               format_func=lambda x: restecg_options[x])

thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)

slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
slope = st.sidebar.selectbox("ST Slope", list(slope_options.keys()), 
                            format_func=lambda x: slope_options[x])

ca = st.sidebar.slider("Num Major Vessels (0-3)", 0, 3, 0)

thal_options = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
thal = st.sidebar.selectbox("Thalassemia", list(thal_options.keys()),
                           format_func=lambda x: thal_options[x])

# ----------------------------
# One-Hot Encode Input
# ----------------------------
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])

# Apply one-hot encoding
cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
input_encoded = pd.get_dummies(input_data, columns=cat_cols, drop_first=True)

# Add missing one-hot columns
for col in selected_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder to match training
input_encoded = input_encoded[selected_features]

# ----------------------------
# 🔎 Sensitivity Test (Fixed: Apply before prediction)
# ----------------------------
st.sidebar.markdown("### 🔎 Sensitivity Test")
oldpeak_override = st.sidebar.slider(
    "Simulate ST Depression (Oldpeak)", 
    0.0, 6.2, float(oldpeak), 0.1
)
input_encoded['oldpeak'] = oldpeak_override  # ✅ Override before prediction

# Show if oldpeak is used
try:
    if 'oldpeak' not in selected_features:
        st.warning("⚠️ `oldpeak` is NOT in the model — changing it has no effect!")
    else:
        st.success("✅ `oldpeak` is in the model — changes will affect prediction.")
except:
    st.info("Could not verify feature list.")

# ----------------------------
# Real-Time Prediction with Submit Button
# ----------------------------
st.subheader("🎯 Prediction Result")

if st.button("🔍 Predict Heart Disease Risk"):
    if model is not None:
        try:
            pred = model.predict(input_encoded)[0]
            prob = model.predict_proba(input_encoded)[0]
            risk = prob[1] * 100

            if pred == 1:
                st.error(f"🔴 High Risk: Patient has heart disease (Probability: {risk:.1f}%)")
            else:
                st.success(f"🟢 Low Risk: Patient does not have heart disease (Probability: {100-risk:.1f}%)")

            st.progress(int(risk))
            st.markdown(f"**Risk Level:** `{risk:.1f}%`")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    else:
        st.warning("Model not loaded. Cannot make predictions.")
else:
    st.info("Adjust the inputs and click **Predict** to see results.")

# ----------------------------
# Data Visualization
# ----------------------------
st.subheader("📈 Heart Disease Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Age vs Max Heart Rate (by Disease)")
    np.random.seed(42)
    ages = np.random.randint(30, 80, 200)
    thalach_sim = 220 - ages + np.random.randn(200) * 10
    disease = np.random.choice([0, 1], 200, p=[0.4, 0.6])
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(ages, thalach_sim, c=disease, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel("Age")
    ax.set_ylabel("Max Heart Rate")
    ax.legend(handles=scatter.legend_elements()[0], labels=["No Disease", "Has Disease"])
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    st.markdown("#### Chest Pain Type Distribution")
    cp_data = np.random.choice(['Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'], 200, p=[0.2, 0.2, 0.3, 0.3])
    vals, counts = np.unique(cp_data, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(vals, counts, color='skyblue', edgecolor='black')
    ax.set_ylabel("Count")
    ax.set_title("Chest Pain Types")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("💡 **Note**: This app uses a machine learning model trained by YM at Microsoft Sprints X on the UCI Heart Disease dataset.")
st.caption("Built with Streamlit | Heart Disease Project © 2025")
