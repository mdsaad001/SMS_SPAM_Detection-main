# ==========================
# Import Required Libraries
# ==========================

import joblib
from preprocess import preprocess_text


# ==========================
# Load Saved Model & Vectorizer
# ==========================

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# ==========================
# Take Input from User
# ==========================

message = input("Enter an SMS message:\n")


# ==========================
# Preprocess the Input
# ==========================

processed_message = preprocess_text(message)


# ==========================
# Convert Text to TF-IDF
# ==========================

vector = vectorizer.transform([processed_message])


# ==========================
# Predict
# ==========================

prediction = model.predict(vector)


# ==========================
# Display Result
# ==========================

print("\nPrediction:")

if prediction[0] == 1:
    print("🚨 SPAM MESSAGE")
else:
    print("✅ HAM (Not Spam)")