import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- Load and Train Model ---
@st.cache_resource
def train_model():
    train_df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])
    val_df = pd.read_csv("val.txt", sep=";", header=None, names=["text", "emotion"])
    train_df = pd.concat([train_df, val_df])

    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(train_df['text'], train_df['emotion'])
    return model

model = train_model()

# --- Correction Logic ---
def correct_prediction(text, predicted_emotion):
    text = text.lower()
    correction_map = {
        "overthinking": "fear", "exhausting": "sadness", "anxious": "fear",
        "hopeless": "sadness", "lonely": "sadness", "worthless": "sadness",
        "panic": "fear", "panicking": "fear", "burned out": "sadness",
        "tired": "sadness", "explode": "anger", "pressure": "fear",
        "breaking down": "sadness", "frustrated": "anger", "screaming": "anger",
        "useless": "sadness", "empty": "sadness", "numb": "sadness",
        "unloved": "sadness", "disappear": "sadness", "racing thoughts": "fear",
        "breathless": "fear", "nervous": "fear", "worried": "fear",
        "tense": "fear", "paranoid": "fear",

        # 🔥 Added strong anger cues
        "furious": "anger", "rage": "anger", "hate": "anger",
        "punch": "anger", "mad": "anger", "irritated": "anger",
        "yelling": "anger", "annoyed": "anger"
    }
    for keyword, corrected_emotion in correction_map.items():
        if keyword in text:
            return corrected_emotion
    return predicted_emotion

# --- Mental Health Detection ---
def detect_mental_health_condition(text, predicted_emotion):
    text = text.lower()
    depression_keywords = ["hopeless", "worthless", "empty", "tired", "useless", "broken", "numb", "unloved", "disappear"]
    anxiety_keywords = ["panic", "panicking", "anxious", "anxiety", "nervous", "worried", "tense", "paranoid", "racing thoughts", "overthinking", "breathless"]
    stress_keywords = ["overwhelmed", "burned out", "exhausted", "under pressure", "frustrated", "irritated", "tension"]

    if predicted_emotion == "sadness" and any(word in text for word in depression_keywords):
        return "⚠️ Possible sign of Depression"
    if predicted_emotion == "fear" and any(word in text for word in anxiety_keywords):
        return "⚠️ Possible sign of Anxiety"
    if predicted_emotion in ["sadness", "fear"] and any(word in text for word in stress_keywords):
        return "⚠️ Possible sign of Stress"
    
    return "✅ No strong indicators of mental health concern"

# --- Streamlit UI ---
st.title("🧠 Emotion & Mental Health Detector")
st.markdown("Enter one or more sentences below to detect emotional tone and check for possible mental health indicators.")

user_input = st.text_area("Enter your sentence(s):", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        lines = user_input.strip().split('\n')
        for i, line in enumerate(lines, 1):
            raw_prediction = model.predict([line])[0]
            corrected = correct_prediction(line, raw_prediction)
            mental_health = detect_mental_health_condition(line, corrected)

            st.markdown(f"### ✏️ Sentence {i}:")
            
            # Emotion display with Streamlit styling
            if corrected == "sadness":
                st.error("😢 Predicted Emotion: Sadness")
            elif corrected == "joy":
                st.success("😊 Predicted Emotion: Joy")
            elif corrected == "anger":
                st.warning("😠 Predicted Emotion: Anger")
            elif corrected == "fear":
                st.info("😨 Predicted Emotion: Fear")
            else:
                st.info(f"🎯 Predicted Emotion: {corrected.capitalize()}")

            st.markdown(f"- 🧠 **Mental Health Check:** {mental_health}")
            st.markdown("---")
