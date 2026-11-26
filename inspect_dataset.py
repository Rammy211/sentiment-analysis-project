
# Import pandas
import pandas as pd

# Step 1: Define the dataset path
DATASET_PATH = "dataset/training.1600000.processed.noemoticon.csv"

# Step 2: Define column names
columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]

# Step 3: Load the CSV dataset
df = pd.read_csv(
    DATASET_PATH,
    encoding="latin-1",
    header=None,
    names=columns
)

# Step 4: Display dataset information
print("DATASET INFORMATION")
print("-------------------")

# Number of rows and columns
print("Dataset shape:", df.shape)

# Display first five records
print("\nFirst five tweets:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check sentiment distribution
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# Display dataset information
print("\nDataset details:")
df.info()
