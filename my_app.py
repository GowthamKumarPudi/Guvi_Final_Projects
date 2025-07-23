
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# --- Load Pretrained Objects ---
@st.cache_resource
def load_model_and_encoders():
    # Load model
    model = pickle.load(open('rf_model.pkl', 'rb'))
    ohe = pickle.load(open('ohe_encoder.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, ohe, tfidf

model, ohe, tfidf = load_model_and_encoders()

# --- Streamlit UI ---
st.title("💼 US Job Salary Predictor")
st.markdown("Enter job details to estimate the Mean Salary (USD).")

# --- Input Fields ---
remote = st.selectbox("Remote", ['Yes', 'No', 'Unknown'])
revenue = st.selectbox("Company Revenue", ['Unknown', '<$1M', '$1M-$10M', '$10M-$100M', '$100M-$1B', '>$1B'])
employee = st.selectbox("Employee Size", ['Unknown', '1-10', '11-50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10000+'])
sector = st.selectbox("Sector", ['Information Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Unknown'])
sector_group = st.selectbox("Sector Group", ['Tech', 'Business', 'Healthcare', 'Unknown'])
state = st.selectbox("Job State", ['CA', 'NY', 'TX', 'FL', 'IL', 'Remote', 'Unknown'])
jobs_group = st.selectbox("Job Group", ['Engineering', 'Marketing', 'Sales', 'Operations', 'Unknown'])
skills = st.text_area("Skills / Keywords", "Python, SQL, Excel")

# --- Predict Button ---
if st.button("Predict Salary"):
    # Format input
    input_df = pd.DataFrame([{
        'Remote': remote,
        'Revenue': revenue,
        'Employee': employee,
        'Sector': sector,
        'Sector_Group': sector_group,
        'State': state,
        'Jobs_Group': jobs_group,
        'Skills': skills
    }])

    # One-hot encode categoricals
    cat_features = ohe.transform(input_df[['Remote', 'Revenue', 'Employee', 'Sector', 'Sector_Group', 'State', 'Jobs_Group']])

    # TF-IDF for Skills
    skill_features = tfidf.transform(input_df['Skills'])

    # Combine features
    final_input = hstack([cat_features, skill_features])

    # Predict
    prediction = model.predict(final_input)[0]
    st.success(f"💰 Estimated Mean Salary: **${prediction:,.2f} USD**")
