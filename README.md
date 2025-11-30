# Candidate Fit Score Predictor

This project provides a machine learning solution to predict the compatibility score between a candidate's resume and a job description.

## Objective

The main goal is to create a Python-based tool that analyzes candidate profiles and job requirements to generate a fit score (from 0 to 100%). This helps in automating the initial screening process for recruitment.

## Technologies Used

- Python
- scikit-learn
- Pandas
- NumPy
- NLTK

## Project Structure

```
.
├── candidate_fit_predictor.py  # Main script for the prediction logic
├── candidates.json             # Sample candidate data
├── jobs.json                   # Sample job descriptions
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Setup and Usage

1.  **Install Dependencies:**
    Make sure you have Python 3 installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Script:**
    Execute the main script to see an example prediction:
    ```bash
    python candidate_fit_predictor.py
    ```
    The script will train a model on the sample data and then predict the fit score for a sample candidate and job, displaying the top matching keywords.