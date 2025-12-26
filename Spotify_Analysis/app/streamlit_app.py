import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="Song Hit Predictor", layout="wide")
st.title("🎵 Song Hit Predictor")
st.write("Predict whether a song is likely to be a **hit** based on audio features.")


with st.sidebar:
    
    st.header("Song features")
    
    danceability = st.slider("Danceability", 0.0, 1.0, 0.7)
    energy = st.slider("Energy", 0.0, 1.0, 0.7)
    loudness = st.number_input("Loudness (dB)", value=-6.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
    valence = st.slider("Valence", 0.0, 1.0, 0.6)
    tempo = st.number_input("Tempo (BPM)", value=120.0)
    duration_ms = st.number_input("Duration (ms)", value=210000.0)
    
    
    
payload = {
"danceability": danceability,
"energy": energy,
"loudness": loudness,
"speechiness": speechiness,
"acousticness": acousticness,
"instrumentalness": instrumentalness,
"liveness": liveness,
"valence": valence,
"tempo": tempo,
"duration_ms": duration_ms,
}


col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict"):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            
            st.subheader("Result")
            st.metric("Hit probability", f"{result['hit_probability']:.2f}")

            label = "✅ Likely HIT" if result["prediction"] == 1 else "❌ Likely NOT a hit"
            st.write(label)

        except requests.RequestException as e:
            st.error(f"API error: {e}")
            
with col2:
    st.subheader("What this app does")
    st.write(
        "- Sends your feature inputs to a FastAPI service\n"
        "- The API loads the trained model and returns a hit probability\n"
        "- Streamlit displays the result"
    )