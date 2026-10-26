import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import shap

np.random.seed(42)
N_SAMPLES = 8000

is_malicious = np.random.binomial(1, 0.22, N_SAMPLES)

data = pd.DataFrame({
    "whois_age_days": np.where(is_malicious, np.random.exponential(18, N_SAMPLES), np.random.exponential(1400, N_SAMPLES)).clip(0, 5000),
    "shannon_entropy": np.where(is_malicious, np.random.normal(4.45, 0.35, N_SAMPLES), np.random.normal(3.35, 0.30, N_SAMPLES)).clip(2.0, 5.5),
    "brand_similarity": np.where(is_malicious, np.random.beta(5, 2, N_SAMPLES) * 100, np.random.beta(1.2, 6, N_SAMPLES) * 100).clip(0, 100),
    "subdomain_depth": np.where(is_malicious, np.random.poisson(2.6, N_SAMPLES), np.random.poisson(0.8, N_SAMPLES)).clip(0, 6),
    "credential_tokens": np.where(is_malicious, np.random.poisson(2.2, N_SAMPLES), np.random.poisson(0.2, N_SAMPLES)).clip(0, 6),
    "suspicious_tld": np.where(is_malicious, np.random.binomial(1, 0.65, N_SAMPLES), np.random.binomial(1, 0.04, N_SAMPLES)),
    "has_raw_ip_or_at": np.where(is_malicious, np.random.binomial(1, 0.35, N_SAMPLES), np.random.binomial(1, 0.005, N_SAMPLES)),
    "https_missing": np.where(is_malicious, np.random.binomial(1, 0.45, N_SAMPLES), np.random.binomial(1, 0.05, N_SAMPLES)),
})

X = data
y = is_malicious

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
print(f"\n✅ Holdout ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
print(classification_report(y_test, model.predict(X_test), target_names=["Benign", "Malicious"]))

explainer = shap.TreeExplainer(model, X_train.sample(200, random_state=42), model_output="probability")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

print("✅ Saved model.pkl and explainer.pkl inside backend/!")
