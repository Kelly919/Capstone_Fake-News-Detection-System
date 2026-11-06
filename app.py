from flask import Flask, render_template, request, redirect, url_for
import joblib
import sqlite3
from datetime import datetime
from preprocess import preprocess_text

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load("models/random_forest_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Initialise the SQLite database
def init_db():
    conn = sqlite3.connect("database/predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# Home page and news prediction
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        news_text = request.form.get("news_text", "").strip()

        # Validate input length
        if len(news_text) > 5000:
            return render_template(
                "index.html",
                prediction="Error: Text must not exceed 5,000 characters.",
                confidence=None,
                news_text=news_text
            )

        # Validate empty input
        if not news_text:
            return render_template(
                "index.html",
                prediction=None,
                confidence=None,
                news_text=""
            )

        # Preprocess the submitted text
        processed_text = preprocess_text(news_text)

        # Convert the text into TF-IDF features
        text_tfidf = tfidf.transform([processed_text])

        # Predict whether the news is fake or genuine
        prediction_value = model.predict(text_tfidf)[0]

        # Obtain prediction probabilities
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities) * 100

        # Convert the prediction into a readable label
        if prediction_value == 0:
            prediction = "Fake"
        else:
            prediction = "Genuine"

        # Save the prediction to the database
        conn = sqlite3.connect("database/predictions.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions
            (news_text, prediction, confidence, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            news_text,
            prediction,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        # Display the result on a separate page
        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            news_text=news_text
        )

    # Display the home page
    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        news_text=""
    )


# Display prediction history
@app.route("/history")
def history():
    search = request.args.get("search", "").strip()
    prediction_filter = request.args.get("filter", "").strip()

    conn = sqlite3.connect("database/predictions.db")
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    parameters = []

    # Search news content
    if search:
        query += " AND news_text LIKE ?"
        parameters.append(f"%{search}%")

    # Filter by classification
    if prediction_filter in ["Fake", "Genuine"]:
        query += " AND prediction = ?"
        parameters.append(prediction_filter)

    query += " ORDER BY id DESC"

    cursor.execute(query, parameters)
    predictions = cursor.fetchall()
    conn.close()

    return render_template(
        "history.html",
        predictions=predictions,
        search=search,
        prediction_filter=prediction_filter
    )


# Display model performance dashboard
@app.route("/performance")
def performance():
    return render_template("performance.html")



@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):
    conn = sqlite3.connect("database/predictions.db")
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions WHERE id = ?",
        (prediction_id,)
    )

    conn.commit()
    conn.close()

    search = request.form.get("search", "").strip()

    if search:
        return redirect(url_for("history", search=search))

    return redirect(url_for("history"))



# Initialise the database and start the application
if __name__ == "__main__":
    init_db()
    app.run(debug=True)

