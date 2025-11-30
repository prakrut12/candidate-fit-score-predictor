import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# --- Download NLTK data (if not already downloaded) ---
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


def preprocess_text(text):
    """Cleans and preprocesses text data."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


def load_data(candidates_path="candidates.json", jobs_path="jobs.json"):
    """Loads candidate and job data from JSON files."""
    with open(candidates_path, "r") as f:
        candidates = json.load(f)
    with open(jobs_path, "r") as f:
        jobs = json.load(f)
    return candidates, jobs


def create_vectorizer(candidates, jobs):
    """Creates a synthetic dataset for training the model."""
    data = []
    vectorizer = TfidfVectorizer()

    for job in jobs:
        job["processed_description"] = preprocess_text(job["description"])

    for candidate in candidates:
        candidate["processed_resume"] = preprocess_text(candidate["resume"])

    all_docs = [job["processed_description"] for job in jobs] + [
        cand["processed_resume"] for cand in candidates
    ]
    vectorizer.fit(all_docs)

    return vectorizer


def predict_fit(candidate_resume, job_description, vectorizer):
    """Predicts the fit score for a new candidate and job."""
    processed_resume = preprocess_text(candidate_resume)
    processed_job = preprocess_text(job_description)

    resume_vec = vectorizer.transform([processed_resume])
    job_vec = vectorizer.transform([processed_job])
    
    # Use cosine similarity as the score
    score = np.dot(job_vec.toarray(), resume_vec.toarray().T)[0, 0]

    # Get matching factors
    feature_names = vectorizer.get_feature_names_out()
    resume_tfidf = pd.Series(resume_vec.toarray().flatten(), index=feature_names)
    job_tfidf = pd.Series(job_vec.toarray().flatten(), index=feature_names)
    matching_keywords = (resume_tfidf * job_tfidf).sort_values(ascending=False)

    return score * 100, matching_keywords[matching_keywords > 0]


if __name__ == "__main__":
    candidates, jobs = load_data()
    vectorizer = create_vectorizer(candidates, jobs)
 
    # --- Example Prediction ---
    test_candidate = candidates[0]  # Alice
    test_job = jobs[0]  # Data Scientist
 
    score, factors = predict_fit(
        test_candidate["resume"], test_job["description"], vectorizer
    )
 
    print(f"\n--- Prediction for {test_candidate['name']} and {test_job['title']} ---")
    print(f"Predicted Fit Score: {score:.2f}%")
    print("\nTop Matching Factors:")
    print(factors.head())
    print("----------------------------------------------------")