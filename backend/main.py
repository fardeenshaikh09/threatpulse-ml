import math
import pickle
import re
import time
from urllib.parse import urlparse
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ThreatPulse ML API", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

SUSPICIOUS_TLDS = {"xyz", "top", "zip", "tk", "ml", "cf", "gq", "cn", "ru", "buzz", "click", "loan", "work"}
LURE_KEYWORDS = ["login", "verify", "secure", "update", "account", "auth", "wallet", "confirm", "bank", "signin", "mfa", "security"]
BRANDS = ["axisbank", "hdfcbank", "icicibank", "sbi", "paytm", "paypal", "microsoft", "apple", "google", "amazon"]

class URLScanRequest(BaseModel):
    url: str

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return round(-sum((c / len(s)) * math.log2(c / len(s)) for c in freq.values()), 2)

def extract_features(raw_url: str) -> dict:
    url_str = raw_url.strip()
    if not url_str.startswith(("http://", "https://")):
        url_str = "http://" + url_str
    parsed = urlparse(url_str)
    host = (parsed.hostname or "").lower()
    parts = host.split(".")
    tld = parts[-1] if len(parts) > 1 else ""
    sub_depth = max(0, len(parts) - 2)

    norm_host = host.replace("0", "o").replace("1", "l").replace("3", "e").replace("-", "")
    brand_sim = 88.0 if any(b in norm_host and not host.endswith(f"{b}.com") for b in BRANDS) else 8.0

    lure_hits = sum(1 for kw in LURE_KEYWORDS if kw in url_str.lower())
    has_ip_or_at = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host)) or ("@" in raw_url))
    susp_tld = int(tld in SUSPICIOUS_TLDS)
    https_missing = int(not raw_url.lower().startswith("https://"))
    age_days = 3 if (susp_tld or brand_sim > 50 or has_ip_or_at) else 1420

    return {
        "whois_age_days": age_days,
        "shannon_entropy": shannon_entropy(url_str),
        "brand_similarity": brand_sim,
        "subdomain_depth": sub_depth,
        "credential_tokens": lure_hits,
        "suspicious_tld": susp_tld,
        "has_raw_ip_or_at": has_ip_or_at,
        "https_missing": https_missing,
    }

@app.post("/api/v1/triage")
def triage_url(req: URLScanRequest):
    t0 = time.perf_counter()
    feats = extract_features(req.url)
    X_df = pd.DataFrame([feats])

    prob = float(model.predict_proba(X_df)[0, 1])
    shap_vals = explainer.shap_values(X_df)[0]
    base_val = float(explainer.expected_value)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "url": req.url,
        "malicious_probability": round(prob, 4),
        "base_value": round(base_val, 4),
        "verdict": "MALICIOUS" if prob >= 0.70 else ("SUSPICIOUS" if prob >= 0.38 else "BENIGN"),
        "features": feats,
        "shap_contributions": dict(zip(X_df.columns, [round(float(v), 4) for v in shap_vals])),
        "inference_latency_ms": latency_ms
    }
