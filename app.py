import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load models and encoders
xgb_model = joblib.load('xgb_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
lstm_model = tf.keras.models.load_model('lstm_model.h5')  # type: ignore
target_encoder = joblib.load('target_encoder.pkl')

# Load tokenizer from JSON
with open('tokenizer.json') as f:
    tokenizer_json = json.load(f)
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
tokenizer = tokenizer_from_json(json.dumps(tokenizer_json))

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit UI
st.title("Customer Query Classification")
st.write("Enter a customer query and get the category prediction.")

query = st.text_input("Enter your query:")
if st.button("Classify Query"):
    if query:
        processed_query = preprocess_text(query)
        
        # XGBoost prediction
        tfidf_features = tfidf_vectorizer.transform([processed_query])
        xgb_pred = xgb_model.predict(tfidf_features)[0]
        
        # LSTM prediction
        lstm_seq = tokenizer.texts_to_sequences([processed_query])
        lstm_padded = pad_sequences(lstm_seq, maxlen=100)
        lstm_pred = np.argmax(lstm_model.predict(lstm_padded), axis=1)[0]
        
        # Ensemble Voting
        final_prediction = np.bincount([xgb_pred, lstm_pred]).argmax()
        category = target_encoder.inverse_transform([final_prediction])[0]
        
        st.success(f"Predicted Category: {category}")
    else:
        st.warning("Please enter a query.")
