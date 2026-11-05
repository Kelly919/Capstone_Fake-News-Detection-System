# FakeNewsChecker AI
### AI-Based Fake News Detection System for Social Media Platforms

## Project Overview

FakeNewsChecker AI is a web-based application that uses natural language processing (NLP) and machine learning to classify English news text as Fake or Genuine. Users can submit news headlines, articles, or social media posts and view the predicted classification and confidence score.

## Features

- Fake and genuine news classification.
- Text preprocessing using NLTK.
- TF-IDF feature extraction.
- Machine learning models: Logistic Regression, Naive Bayes, and Random Forest.
- Prediction history stored using SQLite.
- Model performance evaluation using accuracy, precision, recall, and F1-score.
- Input validation with a maximum text length of 5,000 characters.
- Interactive character counter and Clear Text button.

## Technologies Used

- Python
- Flask
- NLTK
- Scikit-learn
- Pandas and NumPy
- SQLite
- HTML, CSS, and JavaScript

## Project Structure

- `app.py` – Flask application and prediction functionality.
- `preprocess.py` – Text preprocessing functions.
- `prepare_data.py` – Dataset preparation and preprocessing.
- `train_model.py` – Model training and evaluation.
- `templates/` – HTML pages.
- `static/` – CSS stylesheets.
- `data/` – Dataset files used for model training (not included in the repository).
- `models/` – Contains the trained Random Forest model and TF-IDF vectoriser.
- `database/` – Stores the local SQLite prediction history database (created by the application).

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required packages using `pip install -r requirements.txt`.
4. Download the NLTK resources required by the preprocessing script.
5.  Verify that the trained Random Forest model and TF-IDF vectoriser files are available in the `models/` directory.
6. Start the Flask application with `python app.py`.
7. Open the local address displayed in the terminal in your browser.

## Important Note

## Important Note

The repository includes the trained Random Forest model and TF-IDF vectoriser required to run the application. These files must remain in the `models/` directory for the classification functionality to work.

The original dataset files are not included in the repository. Therefore, users can run the application using the supplied trained model without having to download the datasets or retrain the model. However, the datasets are required if users wish to prepare the data and train the machine learning models again.

## Project Scope

The prototype focuses on English text-based news content. It does not currently perform image or video deepfake detection or real-time analysis through social media APIs.
