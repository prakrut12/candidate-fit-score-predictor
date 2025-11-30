from flask import Flask, request, jsonify

# Import the functions from your original script
from candidate_fit_predictor import (
    load_data,
    create_vectorizer,
    predict_fit,
)

app = Flask(__name__)

# --- Load data and train vectorizer once at startup ---
print("Loading data and initializing vectorizer...")
candidates, jobs = load_data()
vectorizer = create_vectorizer(candidates, jobs)
print("Initialization complete. Server is ready.")
# ----------------------------------------------------


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    resume = data.get("resume")
    job_description = data.get("job_description")

    if not resume or not job_description:
        return jsonify({"error": "Missing 'resume' or 'job_description'"}), 400

    score, factors = predict_fit(resume, job_description, vectorizer)
    return jsonify({"fit_score": score, "top_factors": factors.to_dict()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)