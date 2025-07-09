# model/crop_model_trainer.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv("data/crop_recommendation.csv")

# Separate features and label
X = data.drop("label", axis=1)
y = data["label"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to file
joblib.dump(model, "model/crop_recommendation_model.pkl")

print("✅ Model trained and saved as model/crop_recommendation_model.pkl")



