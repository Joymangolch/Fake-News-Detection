import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK Setup ---
# We do this once
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

# --- Model and Vectorizer Loading ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open("fake_news_model.pkl", "rb") as f_model:
            model = pickle.load(f_model)
        with open("tfidf_vectorizer.pkl", "rb") as f_vec:
            vectorizer = pickle.load(f_vec)
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        st.stop()
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Pre-processing Function (same as before) ---
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Streamlit App Interface ---

st.title("Fake News Detection System")
st.write("Enter a news article text below to check if it's Real or Fake.")

# Text area for user input
input_text = st.text_area("News Article Text", height=200)

# Button to trigger prediction
if st.button("Check News"):
    if input_text:
        # 1. Preprocess the input text
        clean_text = preprocess_text(input_text)
        
        # 2. Vectorize the text
        text_vector = vectorizer.transform([clean_text])
        
        # 3. Predict
        prediction = model.predict(text_vector)[0]
        
        # 4. Display the result
        st.subheader("Result")
        if prediction == 1:
            st.error("This looks like FAKE news.")
        else:
            st.success("This looks like REAL news.")
    else:
        st.warning("Please enter some text to check.")