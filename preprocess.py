import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text):
    text = text.lower()                       # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)       # Remove punctuation
    text = re.sub(r"\d+", "", text)           # Remove numbers
    words = text.split()                      # Split into words
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)                    # Join words back
print(preprocess_text("Congratulations!!! You have won ₹50000. Click NOW!!!"))