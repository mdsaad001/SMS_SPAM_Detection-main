# ==========================
# Import Required Libraries
# ==========================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import preprocess_text


# ==========================
# Load Dataset
# ==========================

print("Loading dataset...")

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']


# ==========================
# Display Basic Information
# ==========================

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================
# Remove Duplicate Records
# ==========================

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)


# ==========================
# Convert Labels
# ham = 0
# spam = 1
# ==========================

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})


# ==========================
# Preprocess SMS Messages
# ==========================

print("\nCleaning Messages...")

df['processed_message'] = df['message'].apply(preprocess_text)


# Display sample cleaned messages
print("\nSample Processed Messages:")

print(df[['message', 'processed_message']].head())


# ==========================
# Convert Text to Numbers
# Using TF-IDF
# ==========================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['processed_message'])

y = df['label']


# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================
# Train Model
# ==========================

print("\nTraining Model...")

model = MultinomialNB()

model.fit(X_train, y_train)


# ==========================
# Make Predictions
# ==========================

predictions = model.predict(X_test)


# ==========================
# Evaluate Model
# ==========================

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")

print(f"{accuracy*100:.2f}%")

print("\nClassification Report:\n")

print(classification_report(y_test, predictions))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, predictions))


# ==========================
# Save Model
# ==========================

joblib.dump(model, "model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")

print("Files Created:")

print("model.pkl")

print("vectorizer.pkl")