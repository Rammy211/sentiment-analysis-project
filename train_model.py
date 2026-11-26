
# Import libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocess import clean_text

# 1. Load the dataset
print("Loading dataset...")

df = pd.read_csv(
    "dataset/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None,
    usecols=[0, 5],
    names=["sentiment", "text"]
)

# 2. Select a balanced sample
print("Selecting 100,000 tweets...")

df = df.groupby("sentiment", group_keys=False).sample(
    n=50000,
    random_state=42
)

# 3. Convert sentiment labels
# 0 = Negative
# 4 = Positive
df["sentiment"] = df["sentiment"].map({
    0: 0,
    4: 1
})

# 4. Clean the tweets
print("Cleaning tweets...")

df["text"] = df["text"].apply(clean_text)

# Remove empty tweets
df = df[df["text"].str.strip() != ""]

# 5. Separate features and labels
X = df["text"]
y = df["sentiment"]

# 6. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training tweets:", len(X_train))
print("Testing tweets:", len(X_test))

# 7. Convert text into numerical features using TF-IDF
print("Applying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 8. Create the Logistic Regression model
print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

model.fit(X_train_tfidf, y_train)

# 9. Make predictions
print("Testing the model...")

y_pred = model.predict(X_test_tfidf)

# 10. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-----------------")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Save the trained model
print("\nSaving model...")

joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("Model saved successfully!")
print("Training completed!")
