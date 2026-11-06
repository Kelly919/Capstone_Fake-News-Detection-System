import pandas as pd

# Load the original datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Select one fake article
fake_row = fake_df.dropna(subset=["text"]).iloc[0]

# Select one genuine article
true_row = true_df.dropna(subset=["text"]).iloc[0]

print("\n========== FAKE NEWS ARTICLE ==========\n")
print("Headline:", fake_row["title"])
print("\nArticle:")
print(fake_row["text"])

print("\n========== GENUINE NEWS ARTICLE ==========\n")
print("Headline:", true_row["title"])
print("\nArticle:")
print(true_row["text"])