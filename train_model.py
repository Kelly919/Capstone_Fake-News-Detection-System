import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. LOAD PREPROCESSED DATASET
# ==========================================

df = pd.read_csv("data/preprocessed_news.csv")

# Remove missing processed text
df = df.dropna(subset=["processed_text"])

# Remove empty processed text
df = df[df["processed_text"].str.strip() != ""]

# Input features and labels
X = df["processed_text"]
y = df["label"]

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL EVALUATION")
print("=" * 60)

print("\nDataset size:", len(df))
print("\nClass distribution:")
print(y.value_counts().sort_index())


# ==========================================
# 2. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 3. TF-IDF FEATURE EXTRACTION
# ==========================================

tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ==========================================
# 4. DEFINE THE THREE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 5. TRAIN AND EVALUATE EACH MODEL
# ==========================================

results = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 60)

    # Train model
    model.fit(X_train_tfidf, y_train)

    # Generate predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Calculate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    # Save metrics for comparison
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Confusion Matrix": cm
    }

    # Display results
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:  {f1:.4f} ({f1 * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nConfusion Matrix Labels:")
    print("Rows    = Actual classes")
    print("Columns = Predicted classes")
    print("0 = Fake")
    print("1 = Genuine")


# ==========================================
# 6. COMPARE ALL THREE MODELS
# ==========================================

print("\n" + "=" * 75)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 75)

comparison = pd.DataFrame({
    model_name: {
        "Accuracy (%)": metrics["Accuracy"] * 100,
        "Precision (%)": metrics["Precision"] * 100,
        "Recall (%)": metrics["Recall"] * 100,
        "F1-score (%)": metrics["F1-score"] * 100
    }
    for model_name, metrics in results.items()
}).T

print(comparison.round(2).to_string())


# ==========================================
# 7. SAVE RANDOM FOREST FOR FLASK
# ==========================================

random_forest_model = models["Random Forest"]

joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("\n" + "=" * 60)
print("SAVED DEPLOYMENT FILES")
print("=" * 60)

print("Random Forest model saved successfully.")
print("TF-IDF vectorizer saved successfully.")
print("Your Flask application can continue using Random Forest.")


# ==========================================
# 8. IDENTIFY THE BEST MODEL BY F1-SCORE
# ==========================================

best_model_name = max(
    results,
    key=lambda name: results[name]["F1-score"]
)

print("\nBest model by F1-score:", best_model_name)
print(
    "F1-score:",
    f"{results[best_model_name]['F1-score'] * 100:.2f}%"
)

print("\nEVALUATION COMPLETED SUCCESSFULLY.")