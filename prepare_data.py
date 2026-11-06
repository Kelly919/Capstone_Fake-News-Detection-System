import pandas as pd
from preprocess import preprocess_text

# Load cleaned dataset
df = pd.read_csv("data/cleaned_news.csv")

# Combine the headline and article body
df["combined_text"] = (
    df["title"].fillna("") + " " + df["text"].fillna("")
)

# Apply NLP preprocessing to the combined text
df["processed_text"] = df["combined_text"].apply(preprocess_text)

# Save preprocessed dataset
df.to_csv("data/preprocessed_news.csv", index=False)

print("Preprocessed dataset shape:", df.shape)

print("\nExample:")
print("Original:")
print(df["text"].iloc[0][:500])

print("\nProcessed:")
print(df["processed_text"].iloc[0][:500])