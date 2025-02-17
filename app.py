import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Define the local model file name
file = "poetry_model_6_epoch.keras"

# Check if the file exists before loading
if os.path.exists(file):
    st.write("Model file found. Loading...")
    try:
        model = tf.keras.models.load_model(file, compile=False)  # Load model
        st.write("Model loaded successfully!!!")
        st.write(f"Model architecture: {model.summary()}")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
else:
    st.error("Model file not found! Please check the file path.")
    
# Load the trained model
#model = tf.keras.models.load_model(file, compile=False)

# character mappings
char_vocab = sorted(set("\n !',-.?DHTabcdefghijklmnopqrstuvwxyzñāēġīū۔Ḍḳ"))
char_to_index = {char: idx for idx, char in enumerate(char_vocab)}
index_to_char = {idx: char for idx, char in enumerate(char_vocab)}

seq_length = 100

def generate_poetry(seed_text, next_chars=50):
    result = seed_text
    for _ in range(next_chars):
        encoded_input = [char_to_index.get(c, 0) for c in result[-seq_length:]]
        encoded_input = np.array(encoded_input).reshape(1, -1)
        prediction = model.predict(encoded_input, verbose=0)
        next_char = index_to_char[np.argmax(prediction)]
        result += next_char
    return result

# Streamlit UI
import streamlit as st

# Header Styling
st.markdown("<h1 style='text-align: center; '>Roman Urdu Poetry Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter a phrase, and AI will generate poetry for you!</p>", unsafe_allow_html=True)


# Centered Layout with Columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    seed_text = st.text_input("Enter a starting phrase:", placeholder="Type here...")
    seed_text = seed_text.lower() if seed_text else ""
    
    next_chars = st.slider("Number of Characters to Generate:", min_value=50, max_value=500, step=10, value=200)

    if st.button("Generate Poetry"):
        generated_poetry = generate_poetry(seed_text, next_chars)
        st.subheader("Generated Poetry:")
        st.write(generated_poetry.replace("\n", "  \n"))



def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function with your image path
#set_background("https://images.unsplash.com/photo-1473186505569-9c61870c11f9?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
set_background("https://images.unsplash.com/photo-1505682634904-d7c8d95cdc50?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
#set_background("https://images.unsplash.com/photo-1478641300939-0ec5188d3802?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
#set_background("https://images.unsplash.com/photo-1483546363825-7ebf25fb7513?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
