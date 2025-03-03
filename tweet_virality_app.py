
import joblib
import streamlit as st

# Load trained model and vectorizer
model = joblib.load("viral_tweet_model_v2.pkl")
vectorizer = joblib.load("tfidf_vectorizer_v2.pkl")

st.title("📝 Will Your Tweet Go Viral? 🚀")

st.write("Type your tweet below and click 'Predict' to see if it has viral potential!")

tweet_input = st.text_area("Enter your tweet:", max_chars=280)

if st.button("Predict Virality"):
    if tweet_input:
        tweet_features = vectorizer.transform([tweet_input]).toarray()
        prediction = model.predict(tweet_features)

        if prediction[0] == 1:
            st.success("✅ This tweet has a **HIGH chance** of going viral! 🎉")
        else:
            st.warning("❌ This tweet is **less likely** to go viral. Try tweaking it!")
    else:
        st.error("Please enter a tweet before predicting!")
