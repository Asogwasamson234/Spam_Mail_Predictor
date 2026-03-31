import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import warnings

warnings.filterwarnings('ignore')

# Set encoding for Windows console
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Define dataset paths
ENRON_PATH = "C:/Users/hp/Documents/Projects/spam_mail_prediction/data_sets/enron_spam_data.csv"
SPAM_PATH = "C:/Users/hp/Documents/Projects/spam_mail_prediction/data_sets/spam_original.csv"


class AdvancedTextPreprocessor:
    """
    Advanced text preprocessor with stemming and lemmatization capabilities
    """

    def __init__(self,
                 use_stemming=False,
                 use_lemmatization=True,
                 remove_stopwords=True,
                 remove_numbers=True,
                 remove_punctuation=True,
                 lowercase=True,
                 min_word_length=2):
        """
        Initialize the preprocessor with specified options.

        Parameters:
        -----------
        use_stemming : bool, default=False
            Whether to apply stemming to tokens
        use_lemmatization : bool, default=True
            Whether to apply lemmatization to tokens
        remove_stopwords : bool, default=True
            Whether to remove stopwords
        remove_numbers : bool, default=True
            Whether to remove numeric values
        remove_punctuation : bool, default=True
            Whether to remove punctuation
        lowercase : bool, default=True
            Whether to convert text to lowercase
        min_word_length : int, default=2
            Minimum word length to keep
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.min_word_length = min_word_length

        # Initialize stemmer and lemmatizer
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None

        # Initialize stopwords
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
            # Add custom spam-related stopwords
            custom_stops = {'subject', 're', 'fw', 'fwd', 'hi', 'hello', 'dear'}
            self.stop_words.update(custom_stops)
        else:
            self.stop_words = set()

    def clean_text(self, text):
        """
        Basic text cleaning operations.
        """
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()

        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)

        # Replace URLs with placeholder
        text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)

        # Replace email addresses with placeholder
        text = re.sub(r'\S+@\S+', 'EMAIL', text)

        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation if specified
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_and_process(self, text):
        """
        Tokenize text and apply stemming/lemmatization.
        """
        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Remove short words
        tokens = [token for token in tokens if len(token) >= self.min_word_length]

        # Apply lemmatization (if enabled)
        if self.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Apply stemming (if enabled, usually after lemmatization)
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def transform(self, X):
        """
        Transform text data using the preprocessing pipeline.

        Parameters:
        -----------
        X : array-like
            Input text data

        Returns:
        --------
        list : Preprocessed text data
        """
        processed_texts = []

        for text in X:
            try:
                # Clean text
                cleaned = self.clean_text(text)

                # Skip empty text
                if not cleaned.strip():
                    processed_texts.append('')
                    continue

                # Tokenize and process
                tokens = self.tokenize_and_process(cleaned)

                # Join tokens back into string
                processed_texts.append(' '.join(tokens))
            except Exception as e:
                print(f"Error processing text: {e}")
                processed_texts.append('')

        return processed_texts

    def get_info(self):
        """
        Get preprocessing configuration info.
        """
        return {
            'use_stemming': self.use_stemming,
            'use_lemmatization': self.use_lemmatization,
            'remove_stopwords': self.remove_stopwords,
            'remove_numbers': self.remove_numbers,
            'remove_punctuation': self.remove_punctuation,
            'lowercase': self.lowercase,
            'min_word_length': self.min_word_length
        }


def load_and_combine_data(use_advanced_preprocessing=False, preprocessor=None):
    """
    Load and combine both datasets

    Parameters:
    -----------
    use_advanced_preprocessing : bool, default=False
        Whether to use advanced preprocessing (stemming/lemmatization)
    preprocessor : AdvancedTextPreprocessor, optional
        Custom preprocessor instance
    """

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

    # Apply advanced preprocessing if requested
    if use_advanced_preprocessing and preprocessor is None:
        print(f"\n[INFO] Applying advanced preprocessing...")
        preprocessor = AdvancedTextPreprocessor(
            use_stemming=False,
            use_lemmatization=True,
            remove_stopwords=True,
            remove_numbers=True,
            remove_punctuation=True,
            lowercase=True
        )
        print(f"  Configuration: {preprocessor.get_info()}")

    return combined_data, preprocessor


def preprocess_data(data, use_advanced_preprocessing=False, preprocessor=None):
    """
    Preprocess the data for training

    Parameters:
    -----------
    data : DataFrame
        The dataset containing 'text' and 'label' columns
    use_advanced_preprocessing : bool, default=False
        Whether to use advanced preprocessing (stemming/lemmatization)
    preprocessor : AdvancedTextPreprocessor, optional
        Custom preprocessor instance

    Returns:
    --------
    tuple: (x_train_features, x_test_features, y_train, y_test, feature_extraction, preprocessor)
    """

    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)

    # Separate features and labels
    x = data['text']
    y = data['label']

    print(f"\nSplitting data into train (80%) and test (20%)...")
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Apply advanced preprocessing if requested
    if use_advanced_preprocessing:
        print(f"\n[INFO] Applying advanced text preprocessing...")
        if preprocessor is None:
            preprocessor = AdvancedTextPreprocessor(
                use_stemming=False,
                use_lemmatization=True,
                remove_stopwords=True,
                remove_numbers=True,
                remove_punctuation=True,
                lowercase=True
            )

        print(f"  Preprocessing configuration:")
        for key, value in preprocessor.get_info().items():
            print(f"    {key}: {value}")

        # Transform text data
        print("  Processing training data...")
        x_train_processed = preprocessor.transform(x_train)
        print("  Processing test data...")
        x_test_processed = preprocessor.transform(x_test)

        # Use processed text for TF-IDF
        train_text = x_train_processed
        test_text = x_test_processed
    else:
        print(f"\n[INFO] Using basic preprocessing (no stemming/lemmatization)")
        train_text = x_train
        test_text = x_test

    # Feature extraction using TF-IDF
    print(f"\nExtracting features using TF-IDF...")
    feature_extraction = TfidfVectorizer(
        min_df=1,
        stop_words='english' if not use_advanced_preprocessing else None,  # Stopwords already removed if advanced
        lowercase=True if not use_advanced_preprocessing else False,  # Already lowercased if advanced
        max_features=5000  # Limit features to avoid memory issues
    )

    # Transform text to feature vectors
    print("Transforming training data...")
    x_train_features = feature_extraction.fit_transform(train_text)
    print("Transforming test data...")
    x_test_features = feature_extraction.transform(test_text)

    print(f"\nFeature matrix shape:")
    print(f"Training features: {x_train_features.shape}")
    print(f"Test features: {x_test_features.shape}")

    # Convert to integers
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    return x_train_features, x_test_features, y_train, y_test, feature_extraction, preprocessor


def save_model_data(feature_extraction, model, preprocessor=None):
    """
    Save the model, feature extractor, and preprocessor for later use
    """
    import pickle

    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    # Save feature extraction
    with open('models/feature_extraction.pkl', 'wb') as f:
        pickle.dump(feature_extraction, f)
    print("[OK] Feature extraction saved to: models/feature_extraction.pkl")

    # Save model
    with open('models/spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("[OK] Model saved to: models/spam_model.pkl")

    # Save preprocessor if provided
    if preprocessor:
        with open('models/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        print("[OK] Preprocessor saved to: models/preprocessor.pkl")

        # Save preprocessing configuration
        with open('models/preprocessing_config.txt', 'w') as f:
            f.write("Preprocessing Configuration:\n")
            f.write("=" * 40 + "\n")
            for key, value in preprocessor.get_info().items():
                f.write(f"{key}: {value}\n")
        print("[OK] Preprocessing config saved to: models/preprocessing_config.txt")

    print("\nModel, feature extraction, and preprocessor saved successfully!")


def compare_preprocessing_methods():
    """
    Compare different preprocessing methods on the dataset
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Load data
    data, _ = load_and_combine_data(use_advanced_preprocessing=False)

    # Preprocessing configurations to test
    configs = [
        {'name': 'Basic (No preprocessing)', 'use_stemming': False, 'use_lemmatization': False,
         'remove_stopwords': False, 'remove_numbers': False, 'remove_punctuation': False},
        {'name': 'Stopwords Removal Only', 'use_stemming': False, 'use_lemmatization': False,
         'remove_stopwords': True, 'remove_numbers': False, 'remove_punctuation': False},
        {'name': 'Basic Cleaning', 'use_stemming': False, 'use_lemmatization': False,
         'remove_stopwords': True, 'remove_numbers': True, 'remove_punctuation': True},
        {'name': 'Lemmatization', 'use_stemming': False, 'use_lemmatization': True,
         'remove_stopwords': True, 'remove_numbers': True, 'remove_punctuation': True},
        {'name': 'Stemming', 'use_stemming': True, 'use_lemmatization': False,
         'remove_stopwords': True, 'remove_numbers': True, 'remove_punctuation': True},
        {'name': 'Combined (Lemmatization + Stemming)', 'use_stemming': True, 'use_lemmatization': True,
         'remove_stopwords': True, 'remove_numbers': True, 'remove_punctuation': True},
    ]

    results = {}

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {config['name']}")
        print('=' * 60)

        # Create preprocessor
        preprocessor = AdvancedTextPreprocessor(
            use_stemming=config['use_stemming'],
            use_lemmatization=config['use_lemmatization'],
            remove_stopwords=config['remove_stopwords'],
            remove_numbers=config['remove_numbers'],
            remove_punctuation=config['remove_punctuation'],
            lowercase=True
        )

        # Preprocess
        x = data['text']
        y = data['label']

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply preprocessing
        x_train_processed = preprocessor.transform(x_train)
        x_test_processed = preprocessor.transform(x_test)

        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        x_train_tfidf = vectorizer.fit_transform(x_train_processed)
        x_test_tfidf = vectorizer.transform(x_test_processed)

        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_train_tfidf, y_train)

        # Predict
        y_pred = model.predict(x_test_tfidf)

        # Evaluate
        results[config['name']] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        print(f"Accuracy: {results[config['name']]['accuracy']:.4f}")
        print(f"Precision: {results[config['name']]['precision']:.4f}")
        print(f"Recall: {results[config['name']]['recall']:.4f}")
        print(f"F1-Score: {results[config['name']]['f1_score']:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Best Preprocessing Methods")
    print("=" * 60)
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Accuracy: {best_accuracy[0]} - {best_accuracy[1]['accuracy']:.4f}")

    return results


if __name__ == "__main__":
    # Test the preprocessing
    try:
        print("Testing Advanced Text Preprocessing...\n")

        # Load data with advanced preprocessing
        data, preprocessor = load_and_combine_data(use_advanced_preprocessing=True)

        print(f"\nSample data (first 3 rows):")
        print(data[['text', 'label']].head(3))

        # Show data distribution
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(data)}")
        print(f"Spam (0): {len(data[data['label'] == 0])}")
        print(f"Ham (1): {len(data[data['label'] == 1])}")

        # Show preprocessing example
        if preprocessor:
            print(f"\nPreprocessing Example:")
            sample_text = data['text'].iloc[0]
            print(f"Original: {sample_text[:100]}...")
            processed = preprocessor.transform([sample_text])
            print(f"Processed: {processed[0][:100]}...")

        # Preprocess data
        x_train, x_test, y_train, y_test, feat_ext, prep = preprocess_data(
            data, use_advanced_preprocessing=True, preprocessor=preprocessor
        )

        print(f"\n[SUCCESS] Data preprocessing completed!")

        # Uncomment to compare preprocessing methods
        # results = compare_preprocessing_methods()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
