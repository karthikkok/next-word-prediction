import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load tokenizer and model
@st.cache_resource
def load_resources():
    
    # Laod the tokenizer
    with open('model/tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)

    #Load the LSTM Model
    model=load_model('model/next_word_lstm.keras', compile=False)

    return tokenizer, model

tokenizer, model = load_resources()

MAX_SEQUENCE_LEN = 14  # match your training config

# Session state for managing text and suggestions
if "current_text" not in st.session_state:
    st.session_state.current_text = ""
if "suggested_word" not in st.session_state:
    st.session_state.suggested_word = ""

st.title("Smart Compose: Next Word Prediction (Gmail Style)")

# Input box
#user_text = st.text_input("Type your message:", value=st.session_state.current_text, key="input_box",  height=200)
# st.text_input() or st.text_area()
user_text = st.text_area("Enter your text here:", value=st.session_state.current_text, height=200)

# Predict next word if there's text
if user_text:
    token_list = tokenizer.texts_to_sequences([user_text])[0]
    token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)

    suggested_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            suggested_word = word
            break

    st.session_state.suggested_word = suggested_word
    st.markdown(f"*Suggestion:* `{suggested_word}`")

    # Accept Suggestion
    if st.button("➕ Accept Suggestion"):
        updated_text = user_text.strip() + " " + suggested_word + " "
        st.session_state.current_text = updated_text
        st.rerun()


# Clear text
if st.button("🧹 Clear"):
    st.session_state.current_text = ""
    st.session_state.suggested_word = ""
    #st.experimental_rerun()
    st.rerun()

