import os
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict

app = FastAPI(title="ESG / Safety Incident Intelligence Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "safety_esg_incidents.csv"
FRONTEND_PATH = BASE / "frontend" / "index.html"


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Load dataset and initialize NLP/ML pipeline
df = pd.read_csv(DATA_PATH)
df["clean_narrative"] = df["narrative"].apply(clean_text)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
X = vectorizer.fit_transform(df["clean_narrative"])

cat_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
cat_clf.fit(X, df["target_category"])

sev_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
sev_clf.fit(X, df["target_severity"])


def compute_validation_metrics() -> Dict[str, Any]:
    """Computes stratified cross-validation metrics on the bundled dataset."""
    min_class = min(
        df["target_category"].value_counts().min(),
        df["target_severity"].value_counts().min(),
    )
    n_splits = max(2, min(3, int(min_class)))
    if len(df) < 4 or min_class < 2:
        cat_preds = cat_clf.predict(X)
        sev_preds = sev_clf.predict(X)
        mode = "in-sample (small dataset fallback)"
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cat_preds = cross_val_predict(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            X,
            df["target_category"],
            cv=cv,
        )
        sev_preds = cross_val_predict(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            X,
            df["target_severity"],
            cv=cv,
        )
        mode = f"{n_splits}-fold stratified cross-validation"

    return {
        "validation_strategy": mode,
        "dataset_size": int(len(df)),
        "category_accuracy": round(float(accuracy_score(df["target_category"], cat_preds)), 3),
        "category_macro_f1": round(float(f1_score(df["target_category"], cat_preds, average="macro")), 3),
        "severity_accuracy": round(float(accuracy_score(df["target_severity"], sev_preds)), 3),
        "severity_macro_f1": round(float(f1_score(df["target_severity"], sev_preds, average="macro")), 3),
    }


EVAL_METRICS = compute_validation_metrics()


def extract_trends() -> Dict[str, Any]:
    """Extracts recurring themes and category/severity distributions from historical records."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[::-1][:8]
    top_terms = [feature_names[i] for i in top_idx]

    return {
        "total_incidents": int(len(df)),
        "category_distribution": df["target_category"].value_counts().to_dict(),
        "severity_distribution": df["target_severity"].value_counts().to_dict(),
        "recurring_themes": top_terms,
        "recent_incidents": df[["incident_id", "narrative", "target_category", "target_severity"]]
        .to_dict(orient="records"),
    }


def generate_summary(narrative: str, category: str, severity: str) -> str:
    clean_sent = narrative.strip().rstrip(".")
    return (
        f"[{severity.upper()} | {category.upper()}] Incident reported: {clean_sent}. "
        f"Requires immediate supervisor review and {category}-protocol compliance check."
    )


def recommend_actions(category: str, severity: str, narrative: str) -> List[str]:
    text = narrative.lower()
    actions = []

    if severity == "high":
        actions.append("Escalate immediately to Site EHS Head and halt affected equipment operation.")
    elif severity == "medium":
        actions.append("Log corrective action ticket within 24 hours and brief shift supervisor.")
    else:
        actions.append("Record in routine EHS audit log and review during weekly toolbox talk.")

    if "environment" in category or any(w in text for w in ["oil", "spill", "leak", "seepage", "waste", "emission"]):
        actions.append("Deploy secondary containment/spill kit and inspect seals, valves, and drainage lines.")
    elif "governance" in category or any(w in text for w in ["permit", "audit", "compliance", "log", "training"]):
        actions.append("Audit permit-to-work documentation and verify contractor compliance records.")
    else:
        actions.append("Inspect site housekeeping/PPE compliance and conduct targeted hazard refresher.")

    actions.append("Verify root-cause closure within 7 days to prevent recurrence across renewable assets.")
    return actions


class Query(BaseModel):
    text: str = ""


@app.get("/")
def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH)
    return {"message": "Frontend not found. Use /api/demo or /api/run."}


@app.get("/api/demo")
def demo():
    return {
        "case_study": "ESG / Safety Incident Intelligence Dashboard",
        "status": "ML + TF-IDF Retrieval Pipeline Active",
        "trends": extract_trends(),
        "evaluation_metrics": EVAL_METRICS,
    }


@app.post("/api/run")
def run_case(query: Query):
    if not query.text or not query.text.strip():
        raise HTTPException(status_code=400, detail="Please provide an incident narrative.")

    cleaned = clean_text(query.text)
    vec = vectorizer.transform([cleaned])

    # 1. Classification & Risk Tagging
    pred_cat = str(cat_clf.predict(vec)[0])
    cat_conf = round(float(np.max(cat_clf.predict_proba(vec))), 3)

    pred_sev = str(sev_clf.predict(vec)[0])
    sev_conf = round(float(np.max(sev_clf.predict_proba(vec))), 3)

    # Rule-based safety guardrail for critical high-risk keywords
    if any(w in cleaned for w in ["fatal", "fire", "explosion", "major spill", "electrocution"]):
        pred_sev = "high"

    # 2. Retrieval Evidence (TF-IDF Cosine Similarity)
    sims = cosine_similarity(vec, X).ravel()
    top_indices = sims.argsort()[::-1][:3]
    evidence = []
    for idx in top_indices:
        if sims[idx] > 0:
            row = df.iloc[idx]
            evidence.append({
                "incident_id": row["incident_id"],
                "narrative": row["narrative"],
                "category": row["target_category"],
                "severity": row["target_severity"],
                "similarity_score": round(float(sims[idx]), 3),
            })

    # 3. Summary & Preventive Actions
    summary = generate_summary(query.text, pred_cat, pred_sev)
    actions = recommend_actions(pred_cat, pred_sev, query.text)

    return {
        "input": query.text,
        "predictions": {
            "category": pred_cat,
            "category_confidence": cat_conf,
            "severity": pred_sev,
            "severity_confidence": sev_conf,
        },
        "summary": summary,
        "preventive_actions": actions,
        "retrieval_evidence": evidence,
        "trends": extract_trends(),
        "evaluation_metrics": EVAL_METRICS,
        "limitations": [
            "Trained on a compact starter dataset; rare edge-case phrasing may reduce confidence.",
            "TF-IDF captures lexical overlap rather than deep semantic synonyms without dense embeddings.",
        ],
    }