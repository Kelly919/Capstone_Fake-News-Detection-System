import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load English stop words
stop_words = set(stopwords.words("english"))

# Create lemmatiser
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Tokenise text
    tokens = word_tokenize(text)

    # Keep alphabetic words and remove stop words
    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    # Lemmatise words
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Return cleaned text
    return " ".join(tokens)


# Simple test
if __name__ == "__main__":
    sample_text = "This is a sample news article about people running quickly."

    cleaned_text = preprocess_text(sample_text)

    print("Original:")
    print(sample_text)

    print("\nProcessed:")
    print(cleaned_text)