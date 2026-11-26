
# Import required libraries
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import clean_text

# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="😊",
    layout="centered"
)

# ----------------------------------
# 2. LOAD TRAINED MODEL
# ----------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("model/sentiment_model.pkl")

    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()

# ----------------------------------
# 3. APPLICATION TITLE
# ----------------------------------

st.title("😊 Sentiment Analysis System")

st.write(
    "This application uses Natural Language Processing "
    "and Machine Learning to analyse text sentiment."
)

st.markdown("---")

# ----------------------------------
# 4. USER INPUT
# ----------------------------------

st.subheader("Enter your text")

user_text = st.text_area(
    "Type a tweet or sentence below:",
    placeholder="Example: I really love this product!",
    height=150
)

# ----------------------------------
# 5. PREDICTION BUTTON
# ----------------------------------

if st.button("Analyze Sentiment", use_container_width=True):

    if user_text.strip() == "":

        st.warning("Please enter some text first.")

    else:

        # Clean the input text
        cleaned_text = clean_text(user_text)

        if not cleaned_text:

            st.warning("Please enter meaningful text.")

        else:

            # Convert text into TF-IDF features
            text_vector = vectorizer.transform([cleaned_text])

            # Predict sentiment
            prediction = model.predict(text_vector)[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(text_vector)[0]

            # Identify the confidence score
            confidence = max(probabilities) * 100

            # Display prediction
            st.markdown("---")
            st.subheader("Sentiment Analysis Result")

            if prediction == 1:

                st.success("😊 Positive Sentiment")

            else:

                st.error("😞 Negative Sentiment")

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            # Display probability chart
            st.subheader("Sentiment Probability")

            probability_df = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability": probabilities * 100
            })

            fig, ax = plt.subplots()

            ax.bar(
                probability_df["Sentiment"],
                probability_df["Probability"]
            )

            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Sentiment Prediction Probabilities")

            st.pyplot(fig)

            plt.close(fig)

            # Display cleaned text
            with st.expander("View preprocessed text"):

                st.write(cleaned_text)

# ----------------------------------
# 6. ABOUT SECTION
# ----------------------------------

st.markdown("---")

st.subheader("About this project")

st.write("""
This project performs sentiment analysis using:

- Python
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Scikit-learn
- Streamlit

Dataset: Sentiment140 Twitter Dataset.

Classification categories:
- 0: Negative
- 1: Positive
""")

st.caption("Sentiment Analysis Project | NLP Assignment")
