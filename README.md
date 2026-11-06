# FakeNewsChecker AI
### AI-Based Fake News Detection System for Social Media Platforms

## Project Overview

FakeNewsChecker AI is a web-based application developed as an Honours capstone project in Artificial Intelligence and Data Science.

The system uses Natural Language Processing (NLP) and machine learning techniques to classify English news content as Fake or Genuine. Users can submit news headlines, articles, or social media posts and receive a predicted classification with a confidence score.

The project aims to demonstrate how AI and machine learning can support the identification of potentially misleading information shared online.

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

Prerequisites

Python installed on your computer.

Git installed on your computer.

Step 1: Clone the repository

`git clone https://github.com/Kelly919/Capstone_Fake-News-Detection-System.git`

Step 2: Navigate to the project folder

`cd Capstone_Fake-News-Detection-System`

Step 3: Create a virtual environment

`python -m venv venv`

Step 4: Activate the virtual environment

For Windows PowerShell:

`.\venv\Scripts\Activate.ps1`

Step 5: Install the required dependencies

`pip install -r requirements.txt`

Step 6: Verify the trained model files

Ensure that the following files are available inside the models/ directory:

- random_forest_model.pkl

- tfidf_vectorizer.pkl

Step 7: Run the application

`python app.py`

Step 8: Open the application

Open the local address displayed in your terminal in your web browser. For example:

`http://127.0.0.1:5000/`

## Dataset Information

The datasets used for training and evaluating the machine learning models are not included in this GitHub repository because of their large file sizes.

The following dataset files are retained locally:

- Fake.csv

- True.csv

- cleaned_news.csv

- preprocessed_news.csv

The trained Random Forest model and TF-IDF vectorizer are included in the repository. Therefore, users can run the application without downloading the original datasets or retraining the models.

However, the datasets are required if users wish to repeat the data preparation, preprocessing, training, and evaluation processes.

## Important Note

The repository includes the trained Random Forest model and TF-IDF vectoriser required to run the application. These files must remain in the `models/` directory for the classification functionality to work.

The original dataset files are not included in the repository. Therefore, users can run the application using the supplied trained model without having to download the datasets or retrain the model. However, the datasets are required if users wish to prepare the data and train the machine learning models again.

## Project Scope

The prototype focuses on English text-based news content. It does not currently perform image or video deepfake detection or real-time analysis through social media APIs.
