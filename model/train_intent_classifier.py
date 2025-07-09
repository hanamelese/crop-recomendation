import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example dataset
training_data = [
    ("hello", "greeting"),
    ("hi there", "greeting"),
    ("what crop is best for my land?", "crop_query"),
    ("I have a field with low nitrogen", "crop_query"),
    ("thank you", "thanks"),
    ("thanks a lot", "thanks"),
    ("goodbye", "goodbye"),
    ("see you later", "goodbye")
]

X_texts, y_labels = zip(*training_data)

# Train vectorizer and model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_texts)
model = LogisticRegression()
model.fit(X, y_labels)

# Save them
import os
joblib.dump(model, os.path.join(os.path.dirname(__file__), "intent_classifier.pkl"))
joblib.dump(vectorizer, os.path.join(os.path.dirname(__file__), "intent_vectorizer.pkl"))
