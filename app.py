import streamlit as st
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import clean_text

model = joblib.load('src/fake_news_model.pkl')
tfidf = joblib.load('src/tfidf_vectorizer.pkl')

st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")

# --- Custom CSS ---
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title-text {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0;
}
.subtitle-text {
    text-align: center;
    color: #9ca3af;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.result-card {
    padding: 1.5rem;
    border-radius: 12px;
    margin-top: 1.5rem;
    text-align: center;
}
.result-fake {
    background-color: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.result-real {
    background-color: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.4);
}
.result-label {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<p class="title-text">📰 Fake News Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Paste a news article and get an instant, explainable prediction</p>', unsafe_allow_html=True)

# --- Input ---
user_input = st.text_area("", height=220, placeholder="Paste the full article text here...", label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_clicked = st.button("🔍 Classify Article")

# --- Prediction ---
if classify_clicked:
    if user_input.strip() == "":
        st.warning("Please paste some article text.")
    else:
        with st.spinner("Analyzing article..."):
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

        css_class = "result-fake" if prediction == "Fake News" else "result-real"
        emoji = "🚩" if prediction == "Fake News" else "✅"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-label">{emoji} {prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        m1, m2 = st.columns(2)
        for (label, prob), col in zip(zip(model.classes_, probabilities), [m1, m2]):
            col.metric(label=label, value=f"{prob:.1%}")

        st.write("")
        with st.expander("🔬 See cleaned text (what the model actually analyzed)"):
            st.code(cleaned if cleaned else "(empty after cleaning)", language=None)

        with st.expander("ℹ️ About this model"):
            st.write("""
            This model was trained on ~44,000 news articles using TF-IDF vectorization and 
            Logistic Regression. Three separate data leakage sources (subject metadata, 
            wire-service datelines, scraped media credit lines) were identified and corrected 
            during development to ensure the model learns genuine linguistic patterns rather 
            than dataset artifacts.
            """)
else:
    st.info("👆 Paste an article above and click **Classify Article** to get started.")