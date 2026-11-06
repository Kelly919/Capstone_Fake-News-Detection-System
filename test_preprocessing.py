from preprocess import preprocess_text

sample_text = (
    "WASHINGTON (Reuters) - The U.S. Department of Agriculture "
    "announced new measures to support farmers and improve agricultural"
    " production. The initiative aims to provide additional resources"
    " to farming communities and strengthen the country's food supply. "
    "Officials said the programme would be implemented through existing "
    "agricultural support services, with further details to be released as the initiative progresses."
   
)

processed_text = preprocess_text(sample_text)

print("Original Text:")
print(sample_text)

print("\nPreprocessed Text:")
print(processed_text)