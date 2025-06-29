import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import pickle

# Load the trained model
model = tf.keras.models.load_model('sentiment_lstm_model_v2.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Clean the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Constants
max_length = 200

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.subheader("Predict whether a movie review is Positive or Negative")

user_input = st.text_area("Enter your movie review:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)[0][0]

    sentiment = "🌟 Positive" if prediction > 0.5 else "😠 Negative"
    st.markdown(f"**Sentiment:** {sentiment} ({prediction:.2f})")
