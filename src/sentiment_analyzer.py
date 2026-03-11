import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# ---------- Page Setup ----------
st.set_page_config(page_title="Pro Sentiment AI", page_icon="📚", layout="wide")

# ---------- Download NLTK ----------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------- Load Models ----------
@st.cache_resource
def load_ml_pipeline():
    """Loads the 85% accuracy Logistic Regression pipeline"""
    return joblib.load("sentiment_pipeline.pkl")

@st.cache_resource
def load_bert_pipeline():
    """Loads DistilBERT for Deep Learning inference"""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ---------- Text Cleaning ----------
def clean_input(text):
    # Standardize and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()

    # Preserve negation words as they are critical for sentiment
    negation_words = {"not", "no", "never", "neither", "nor"}
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words or w in negation_words]

    return " ".join(cleaned)

# ---------- Sidebar / Model Config ----------
st.sidebar.title("⚙️ Model Configuration")
model_choice = st.sidebar.radio(
    "Select Architecture:",
    ("Logistic Regression (Fast/ML)", "BERT Transformer (Deep Learning)")
)

st.sidebar.divider()
st.sidebar.write("**Current Specs:**")
if model_choice == "Logistic Regression (Fast/ML)":
    st.sidebar.info("Accuracy: ~85%\n\nFeatures: TF-IDF + N-Grams")
else:
    st.sidebar.info("Architecture: Transformer\n\nMethod: Transfer Learning")

# ---------- UI Content ----------
st.title("🚀 Kindle Review Sentiment Analysis Suite")
st.write("Compare standard Machine Learning against Deep Learning Transformer models.")

# Example Reviews
examples = [
    "The character development was slow but the ending was breathtaking!",
    "Terrible formatting. I want my money back.",
    "Not as good as the first one, but still a decent read."
]
example_choice = st.selectbox("Try an example:", [""] + examples)

user_input = st.text_area("Review Input:", value=example_choice, height=150)

# ---------- Analysis Logic ----------
if st.button("Run Analysis"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Analyzing with {model_choice}..."):
            st.divider()
            
            # --- MODEL 1: Logistic Regression ---
            if model_choice == "Logistic Regression (Fast/ML)":
                model = load_ml_pipeline()
                cleaned = clean_input(user_input)
                prediction = model.predict([cleaned])[0]
                probs = model.predict_proba([cleaned])[0]
                
                pos_score, neg_score = probs[1] * 100, probs[0] * 100
                label = "POSITIVE 😊" if prediction == 1 else "NEGATIVE 😞"
            
            # --- MODEL 2: BERT Transformer ---
            else:
                bert = load_bert_pipeline()
                # BERT handles its own tokenization/cleaning internally
                result = bert(user_input[:512])[0]
                label_raw = result['label']
                conf = result['score'] * 100
                
                label = "POSITIVE 😊" if label_raw == "POSITIVE" else "NEGATIVE 😞"
                pos_score = conf if label_raw == "POSITIVE" else (100 - conf)
                neg_score = conf if label_raw == "NEGATIVE" else (100 - conf)

            # --- Display Metrics ---
            col_res, col_conf = st.columns(2)
            col_res.metric("Sentiment", label)
            col_conf.metric("Confidence", f"{max(pos_score, neg_score):.2f}%")

            c1, c2 = st.columns(2)
            c1.write(f"**Positivity:** {pos_score:.1f}%")
            c1.progress(int(pos_score))
            c2.write(f"**Negativity:** {neg_score:.1f}%")
            c2.progress(int(neg_score))

            # --- Visualization ---
            st.subheader("🔠 Key Terms (Word Cloud)")
            cleaned_cloud = clean_input(user_input)
            if len(cleaned_cloud.split()) > 0:
                wordcloud = WordCloud(background_color="white", colormap='viridis', width=800, height=400).generate(cleaned_cloud)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.write("Enter more text to generate a word cloud.")