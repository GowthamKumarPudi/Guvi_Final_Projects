
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    df = pd.read_csv('/content/0000-1 (1).csv')

    # Drop unwanted index column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Drop rows with missing values in key features
    df.dropna(subset=[
        'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
        'key', 'loudness', 'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
    ], inplace=True)

    # Convert 'explicit' to numeric
    df['explicit'] = df['explicit'].astype(str).map({'true': 1, 'false': 0})
    df['explicit'] = df['explicit'].fillna(0)

    # Drop non-numeric columns
    df_numeric = df.drop(columns=['track_id', 'track_name', 'artists', 'album_name', 'track_genre'])

    # Fill any unexpected NaNs with column means
    df_numeric = df_numeric.fillna(df_numeric.mean(numeric_only=True))

    # Scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_numeric)

    return df, df_numeric.columns.tolist(), scaler, scaled


# Load data
df, feature_cols, scaler, scaled_features = load_data()

# App Title
st.title("🎧Spotify Song Recommender")
st.markdown("Adjust the audio features below to get personalized song suggestions.")

# Collect user inputs
user_inputs = []
for col in feature_cols:
    if col == 'explicit':
        val = st.selectbox(f'{col}', [0, 1], index=0)
    elif col in ['key', 'mode', 'time_signature']:
        val = st.number_input(f'{col} (integer)', value=0, step=1)
    elif col == 'tempo':
        val = st.slider(f'{col} (BPM)', 40.0, 220.0, 120.0)
    elif col == 'loudness':
        val = st.slider(f'{col} (dB)', -60.0, 0.0, -10.0)
    else:
        val = st.slider(f'{col}', 0.0, 1.0, 0.5)
    
    # Ensure no NaN values
    if pd.isna(val):
        val = 0.0
    user_inputs.append(val)

# Recommend songs
if st.button("Recommend🎧"):
    try:
        user_scaled = scaler.transform([user_inputs])
        similarities = cosine_similarity(user_scaled, scaled_features)[0]
        indices = np.argsort(similarities)[::-1][:5]
        results = df.iloc[indices][['track_name', 'artists', 'track_genre', 'popularity']]
        st.subheader("🔥Top Recommendations")
        st.table(results)
    except Exception as e:
        st.error(f"Error: {e}")
