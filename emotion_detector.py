import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Load dataset
def load_data(path):
    df = pd.read_csv(path, sep=';', header=None, names=["text", "emotion"])
    return df

print("📥 Loading training and validation data...")
train_df = load_data("train.txt")
val_df = load_data("val.txt")
test_df = load_data("test.txt")

train_df = pd.concat([train_df, val_df])

print("🧠 Training model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
model.fit(train_df['text'], train_df['emotion'])

print("✅ Training complete. Evaluating on test set...")
preds = model.predict(test_df['text'])

print("🔍 Accuracy:", accuracy_score(test_df['emotion'], preds))
print("\n📊 Classification Report:\n", classification_report(test_df['emotion'], preds))

#new
def detect_mental_health_condition(text, predicted_emotion):
    text = text.lower()

    depression_keywords = ["hopeless", "worthless", "empty", "tired", "useless", "broken", "numb", "unloved", "disappear"]
    anxiety_keywords = ["panic", "panicking", "anxious", "anxiety", "nervous", "worried", "tense", "paranoid", "racing", "racing thoughts", "overthinking", "breathless"]
    stress_keywords = ["overwhelmed", "burned out", "exhausted", "under pressure", "frustrated", "irritated", "tension"]

    # Check depression
    if predicted_emotion == "sadness" and any(keyword in text for keyword in depression_keywords):
        return "⚠️ Possible sign of Depression"
    
    # Check anxiety
    if predicted_emotion == "fear" and any(keyword in text for keyword in anxiety_keywords):
        return "⚠️ Possible sign of Anxiety"
    
    # Check stress
    if predicted_emotion in ["sadness", "fear"] and any(keyword in text for keyword in stress_keywords):
        return "⚠️ Possible sign of Stress"
    
    return "✅ No strong indicators of mental health concern"

def correct_prediction(text, predicted_emotion):
    text = text.lower()
    correction_map = {
        "overthinking": "fear",
        "exhausting": "sadness",
        "anxious": "fear",
        "hopeless": "sadness",
        "lonely": "sadness",
        "worthless": "sadness",
        "panic": "fear",
        "panicking": "fear",
        "burned out": "sadness",
        "tired": "sadness",
        "explode": "anger",
        "pressure": "fear",
        "breaking down": "sadness",
        "frustrated": "anger",
        "screaming": "anger",
        "useless": "sadness",
        "empty": "sadness",
        "numb": "sadness",
        "unloved": "sadness",
        "disappear": "sadness",
        "racing thoughts": "fear",
        "breathless": "fear",
        "nervous": "fear",
        "worried": "fear",
        "tense": "fear",
        "paranoid": "fear"
    }

    for keyword, corrected_emotion in correction_map.items():
        if keyword in text:
            return corrected_emotion

    return predicted_emotion


#new loop
while True:
    user_input = input("\n💬 Enter a sentence (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    # First: get prediction from model
    raw_prediction = model.predict([user_input])[0]

    # Second: correct it if needed
    corrected_prediction = correct_prediction(user_input, raw_prediction)

    # Third: print corrected prediction
    print("🎯 Predicted Emotion:", corrected_prediction)

    # Fourth: run mental health check
    mh_signal = detect_mental_health_condition(user_input, corrected_prediction)
    print("🧠 Mental Health Check:", mh_signal)

