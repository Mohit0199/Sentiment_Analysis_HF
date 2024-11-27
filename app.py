import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_model():
    return pipeline(task="sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

model = load_model()


# Streamlit App Layout
st.title("Sentiment Analysis 🤖")
st.write("Enter a sentence, and I'll tell you the sentiment!")

# User input text
user_input = st.text_area("Your text here:", "")

# Button to trigger analysis
if st.button("Analyze"):
    if user_input:
        result = model(user_input)[0]
        sentiment = result['label']
        confidence = round(result['score'] * 100, 2)
        
        # Display results
        st.subheader("Prediction Result:")
        if sentiment == "POSITIVE":
            st.success(f"😊 Positive Sentiment with {confidence}% confidence!")
        else:
            st.error(f"😞 Negative Sentiment with {confidence}% confidence!")
    else:
        st.warning("Please enter some text to analyze.")


