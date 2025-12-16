import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("data.csv")

X = df.drop("donation_need", axis=1)
y = df["donation_need"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")
