# Quick sanity check
# Run this in Python terminal

import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('model/fraud_model.pkl')
print("✅ Model loaded")

# Load one sample transaction
df = pd.read_csv('data/creditcard.csv')
sample = df.drop('Class', axis=1).iloc[0:1]

# Predict
proba = model.predict_proba(sample)[0][1]
print(f"✅ Prediction works")
print(f"   Sample fraud probability: {proba:.4f}")
print(f"   Decision: {'🚨 FRAUD' if proba > 0.5 else '✅ LEGITIMATE'}")