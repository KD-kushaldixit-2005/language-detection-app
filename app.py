
import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Language Detection App",
    page_icon="🌍",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
cv = joblib.load("vectorizer.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg,#081229,#0b1f45,#081229);
    color:white;
}

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#5DADE2;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#D6EAF8;
}

.card{
    background:#101d3d;
    padding:15px;
    border-radius:15px;
    margin-bottom:10px;
    border:1px solid #1f3b73;
}

.result-card{
    background:#0f3d2e;
    padding:20px;
    border-radius:15px;
    border:2px solid #22c55e;
}

.tech{
    background:#101d3d;
    padding:10px;
    border-radius:10px;
    margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("## 📌 Project Info")

    st.success("🎯 Accuracy: 95%")

    st.info("🧠 Model: Multinomial Naive Bayes")

    st.warning("🌍 Languages: 17+")

    st.markdown("---")

    st.markdown("## 🛠 Tech Stack")

    st.markdown("""
    - Python
    - Numpy
    - Panda                       
    - NLP
    - CountVectorizer
    - Scikit-Learn
    - Streamlit
    """)

    st.markdown("---")

    st.markdown("""
    ### ℹ About

    This app detects the language of text
    using Machine Learning and NLP.
    """)

# ---------------- HEADER ----------------
st.markdown(
    '<p class="main-title">🌍 Language Detection App</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Detect Any Language Using Machine Learning & NLP ✨</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- INPUT ----------------
st.subheader("✍ Enter Text")

text = st.text_area(
    "",
    height=200,
    placeholder="Type or paste your text here..."
)

st.markdown("### OR")

uploaded_file = st.file_uploader(
    "📂 Upload Text File",
    type=["txt"]
)

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

# ---------------- BUTTON ----------------
if st.button("🔍 Detect Language", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        data = cv.transform([text])

        prediction = model.predict(data)[0]

        st.markdown(
            f"""
            <div class="result-card">
            <h3>✅ Detected Language</h3>
            <h1>{prediction}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            prob = model.predict_proba(data)
            confidence = np.max(prob) * 100

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

        except:
            pass

# ---------------- HOW IT WORKS ----------------
st.markdown("---")

st.subheader("⚙ How It Works")

st.markdown("""
1. User enters text
2. CountVectorizer converts text into numerical features
3. Multinomial Naive Bayes predicts language
4. Result is displayed instantly
""")

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "❤️ Built using Python, NLP, Scikit-Learn and Streamlit"
)