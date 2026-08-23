import joblib
scaler = joblib.load('feature_scaler.pkl')
print("Mean:", scaler.mean_)
print("Scale:", scaler.scale_)
