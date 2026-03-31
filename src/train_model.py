import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
import os
import sys
import warnings

warnings.filterwarnings('ignore')

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
        try:
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

        except Exception as e:
            print(f"\n[ERROR] Failed to load Enron dataset: {e}")
            enron_data = pd.DataFrame()
    else:
        print(f"\n[WARNING] Enron dataset not found at: {ENRON_PATH}")

    # Load SMS spam dataset
    sms_data = pd.DataFrame()
    if os.path.exists(SPAM_PATH):
        try:
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

        except Exception as e:
            print(f"\n[ERROR] Failed to load SMS dataset: {e}")
            sms_data = pd.DataFrame()
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
        print(f"\n[ERROR] No datasets found!")
        return None

    return combined_data


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

    # Check if data is valid
    if data is None or len(data) == 0:
        print("\n[ERROR] No data to preprocess!")
        return None, None, None, None, None, None

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
        from data_preprocessing import AdvancedTextPreprocessor
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
        preprocessor = None

    # Feature extraction using TF-IDF
    print(f"\nExtracting features using TF-IDF...")
    feature_extraction = TfidfVectorizer(
        min_df=1,
        stop_words='english' if not use_advanced_preprocessing else None,
        lowercase=True if not use_advanced_preprocessing else False,
        max_features=5000
    )

    # Transform text to feature vectors
    try:
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

    except Exception as e:
        print(f"\n[ERROR] Feature extraction failed: {e}")
        return None, None, None, None, None, None


def perform_cross_validation(feature_extraction, x_train_text, y_train, cv_folds=5):
    """
    Perform k-fold cross-validation to evaluate model performance.

    Parameters:
    -----------
    feature_extraction : TfidfVectorizer
        Fitted TF-IDF vectorizer
    x_train_text : array-like
        Training text data
    y_train : array-like
        Training labels
    cv_folds : int, default=5
        Number of cross-validation folds

    Returns:
    --------
    dict : Cross-validation scores
    """
    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION ({cv_folds}-FOLD)")
    print("=" * 60)

    # Use StratifiedKFold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Models to evaluate with cross-validation
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB()
    }

    cv_results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        # Store fold scores
        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        # Perform cross-validation manually to get multiple metrics
        fold = 1
        for train_idx, val_idx in skf.split(x_train_text, y_train):
            # Split data
            X_train_fold = x_train_text[train_idx]
            X_val_fold = x_train_text[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            # Transform text to features
            X_train_features = feature_extraction.transform(X_train_fold)
            X_val_features = feature_extraction.transform(X_val_fold)

            # Train model
            model.fit(X_train_features, y_train_fold)

            # Predict
            y_pred = model.predict(X_val_features)

            # Calculate metrics
            fold_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            fold_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            fold_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            fold_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))

            print(f"  Fold {fold}: Acc={fold_scores['accuracy'][-1]:.4f}, "
                  f"P={fold_scores['precision'][-1]:.4f}, "
                  f"R={fold_scores['recall'][-1]:.4f}, "
                  f"F1={fold_scores['f1'][-1]:.4f}")
            fold += 1

        # Calculate mean and std for each metric
        cv_results[model_name] = {
            'accuracy': {'mean': np.mean(fold_scores['accuracy']), 'std': np.std(fold_scores['accuracy'])},
            'precision': {'mean': np.mean(fold_scores['precision']), 'std': np.std(fold_scores['precision'])},
            'recall': {'mean': np.mean(fold_scores['recall']), 'std': np.std(fold_scores['recall'])},
            'f1': {'mean': np.mean(fold_scores['f1']), 'std': np.std(fold_scores['f1'])},
            'fold_scores': fold_scores
        }

        print(f"\n{model_name} Summary:")
        print(
            f"  Accuracy: {cv_results[model_name]['accuracy']['mean']:.4f} (+/- {cv_results[model_name]['accuracy']['std']:.4f})")
        print(
            f"  Precision: {cv_results[model_name]['precision']['mean']:.4f} (+/- {cv_results[model_name]['precision']['std']:.4f})")
        print(
            f"  Recall: {cv_results[model_name]['recall']['mean']:.4f} (+/- {cv_results[model_name]['recall']['std']:.4f})")
        print(f"  F1-Score: {cv_results[model_name]['f1']['mean']:.4f} (+/- {cv_results[model_name]['f1']['std']:.4f})")

    return cv_results


def train_model_with_cv(data, model_type='logistic_regression', use_advanced_preprocessing=False):
    """
    Train model with cross-validation and final evaluation.

    Parameters:
    -----------
    data : DataFrame
        The dataset
    model_type : str
        Type of model to train
    use_advanced_preprocessing : bool
        Whether to use advanced preprocessing

    Returns:
    --------
    dict : Training results
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB

    # Preprocess data
    x_train_features, x_test_features, y_train, y_test, feature_extraction, preprocessor = preprocess_data(
        data, use_advanced_preprocessing=use_advanced_preprocessing
    )

    if x_train_features is None:
        return None

    # Perform cross-validation on training data
    cv_results = perform_cross_validation(feature_extraction, x_train_features, y_train, cv_folds=5)

    # Select best model based on cross-validation F1 score
    best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['f1']['mean'])
    print(f"\n{'=' * 60}")
    print(f"Best model based on cross-validation: {best_model_name}")
    print(f"Mean F1-Score: {cv_results[best_model_name]['f1']['mean']:.4f}")
    print('=' * 60)

    # Train final model
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'naive_bayes': MultinomialNB()
    }

    if model_type not in models:
        model_type = 'logistic_regression'

    final_model = models[model_type]
    final_model.fit(x_train_features, y_train)

    # Evaluate on test set
    y_pred = final_model.predict(x_test_features)

    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    print(f"\nFinal Model Evaluation on Test Set:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")

    return {
        'model': final_model,
        'feature_extraction': feature_extraction,
        'preprocessor': preprocessor,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'y_test': y_test,
        'y_pred': y_pred
    }


def save_model_data(feature_extraction, model, preprocessor=None):
    """Save the model and feature extractor for later use"""
    import pickle

    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    try:
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

        print("\nModel and feature extraction saved successfully!")

    except Exception as e:
        print(f"\n[ERROR] Failed to save model: {e}")
        raise


if __name__ == "__main__":
    # Test the preprocessing
    try:
        data = load_and_combine_data()

        if data is not None:
            print(f"\nSample data (first 5 rows):")
            print(data.head())

            # Show data distribution
            print(f"\nDataset Statistics:")
            print(f"Total samples: {len(data)}")
            print(f"Spam (0): {len(data[data['label'] == 0])}")
            print(f"Ham (1): {len(data[data['label'] == 1])}")

            # Train model with cross-validation
            results = train_model_with_cv(
                data,
                model_type='logistic_regression',
                use_advanced_preprocessing=True  # Enable advanced preprocessing
            )

            if results:
                # Save model
                save_model_data(
                    results['feature_extraction'],
                    results['model'],
                    results.get('preprocessor')
                )
                print("\n[SUCCESS] Training completed successfully!")
            else:
                print("\n[ERROR] Training failed!")
        else:
            print("\n[ERROR] No data loaded!")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
