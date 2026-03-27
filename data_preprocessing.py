import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys

# Set encoding for Windows console
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Define dataset paths
ENRON_PATH = "C:/Users/hp/Documents/Projects/spam_mail_prediction/data_sets/enron_spam_data.csv"
SPAM_PATH = "C:/Users/hp/Documents/Projects/spam_mail_prediction/data_sets/spam_original.csv"


def load_and_combine_data():
    """Load and combine both datasets"""

    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    # Load Enron dataset
    enron_data = pd.DataFrame()
    if os.path.exists(ENRON_PATH):
        print(f"\n[OK] Loading Enron dataset from: {ENRON_PATH}")
        enron_data = pd.read_csv(ENRON_PATH)
        print(f"  Shape: {enron_data.shape}")

        # Rename columns to standard format
        enron_data = enron_data.rename(columns={
            'Spam/Ham': 'label',
            'Message': 'text'
        })

        # Convert labels to binary (spam=0, ham=1)
        enron_data['label'] = enron_data['label'].map({'spam': 0, 'ham': 1})

        # Handle missing values
        enron_data = enron_data.where((pd.notnull(enron_data)), ' ')

        # Display distribution
        spam_count = len(enron_data[enron_data['label'] == 0])
        ham_count = len(enron_data[enron_data['label'] == 1])
        print(f"  Spam: {spam_count}, Ham: {ham_count}")

    else:
        print(f"\n[WARNING] Enron dataset not found at: {ENRON_PATH}")

    # Load SMS spam dataset
    sms_data = pd.DataFrame()
    if os.path.exists(SPAM_PATH):
        print(f"\n[OK] Loading SMS dataset from: {SPAM_PATH}")
        sms_data = pd.read_csv(SPAM_PATH, encoding='latin-1')
        print(f"  Shape: {sms_data.shape}")

        # Extract relevant columns (v1 is label, v2 is message)
        sms_data = sms_data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

        # Convert labels to binary (spam=0, ham=1)
        sms_data['label'] = sms_data['label'].map({'spam': 0, 'ham': 1})

        # Handle missing values
        sms_data = sms_data.where((pd.notnull(sms_data)), ' ')

        # Display distribution
        spam_count = len(sms_data[sms_data['label'] == 0])
        ham_count = len(sms_data[sms_data['label'] == 1])
        print(f"  Spam: {spam_count}, Ham: {ham_count}")

    else:
        print(f"\n[WARNING] SMS dataset not found at: {SPAM_PATH}")

    # Combine datasets
    if not enron_data.empty and not sms_data.empty:
        combined_data = pd.concat([enron_data, sms_data], ignore_index=True)
        print(f"\n[OK] Combined dataset created!")
        print(f"  Total samples: {len(combined_data)}")
        print(f"  Spam samples: {len(combined_data[combined_data['label'] == 0])}")
        print(f"  Ham samples: {len(combined_data[combined_data['label'] == 1])}")
    elif not enron_data.empty:
        combined_data = enron_data
        print(f"\n[OK] Using only Enron dataset")
    elif not sms_data.empty:
        combined_data = sms_data
        print(f"\n[OK] Using only SMS dataset")
    else:
        raise FileNotFoundError("No dataset files found! Please check the file paths.")

    return combined_data


def preprocess_data(data):
    """Preprocess the data for training"""

    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)

    # Separate features and labels
    x = data['text']
    y = data['label']

    print(f"\nSplitting data into train (80%) and test (20%)...")
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Feature extraction using TF-IDF
    print(f"\nExtracting features using TF-IDF...")
    feature_extraction = TfidfVectorizer(
        min_df=1,
        stop_words='english',
        lowercase=True,
        max_features=5000  # Limit features to avoid memory issues
    )

    # Transform text to feature vectors
    print("Transforming training data...")
    x_train_features = feature_extraction.fit_transform(x_train)
    print("Transforming test data...")
    x_test_features = feature_extraction.transform(x_test)

    print(f"\nFeature matrix shape:")
    print(f"Training features: {x_train_features.shape}")
    print(f"Test features: {x_test_features.shape}")

    # Convert to integers
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    return x_train_features, x_test_features, y_train, y_test, feature_extraction


def save_model_data(feature_extraction, model):
    """Save the model and feature extractor for later use"""
    import pickle

    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    # Save feature extraction
    with open('feature_extraction.pkl', 'wb') as f:
        pickle.dump(feature_extraction, f)
    print("[OK] Feature extraction saved to: feature_extraction.pkl")

    # Save model
    with open('spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("[OK] Model saved to: spam_model.pkl")

    print("\nModel and feature extraction saved successfully!")


if __name__ == "__main__":
    # Test the preprocessing
    try:
        data = load_and_combine_data()
        print(f"\nSample data (first 5 rows):")
        print(data.head())

        # Show data distribution
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(data)}")
        print(f"Spam (0): {len(data[data['label'] == 0])}")
        print(f"Ham (1): {len(data[data['label'] == 1])}")

        # Preprocess
        x_train, x_test, y_train, y_test, feat_ext = preprocess_data(data)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
