import re

# Function to clean a tweet
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove usernames (e.g. @john)
    text = re.sub(r"@\w+", "", text)

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Test the preprocessing function
tweet = "@John I LOVE this phone!!! Visit https://example.com"

cleaned_tweet = clean_text(tweet)

print("Original Tweet:")
print(tweet)

print("\nCleaned Tweet:")
print(cleaned_tweet)
