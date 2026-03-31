from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import numpy as np
import pandas as pd


def evaluate_model():
    """Evaluate the trained model with cross-validation metrics"""

    # Load model and feature extraction
    try:
        with open('models/feature_extraction.pkl', 'rb') as f:
            feature_extraction = pickle.load(f)

        with open('models/spam_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Try to load preprocessor if exists
        try:
            with open('models/preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            print("Preprocessor loaded successfully!")
        except:
            preprocessor = None
            print("No preprocessor found, using basic preprocessing.")

        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model files not found! Please run train_model.py first.")
        return

    # Load the data for evaluation
    from data_preprocessing import load_and_combine_data, preprocess_data

    data = load_and_combine_data()

    if data is None:
        print("Failed to load data!")
        return

    # Preprocess data with same method used during training
    use_advanced = preprocessor is not None
    _, x_test_features, _, y_test, _, _ = preprocess_data(
        data,
        use_advanced_preprocessing=use_advanced,
        preprocessor=preprocessor
    )

    if x_test_features is None:
        print("Failed to preprocess data!")
        return

    # Make predictions
    y_pred = model.predict(x_test_features)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print("MODEL EVALUATION")
    print('=' * 60)
    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Spam', 'Ham']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Spam correctly identified): {cm[0][0]}")
    print(f"False Positives (Ham incorrectly as Spam): {cm[0][1]}")
    print(f"False Negatives (Spam incorrectly as Ham): {cm[1][0]}")
    print(f"True Positives (Ham correctly identified): {cm[1][1]}")

    # Additional statistics
    total = len(y_test)
    correct = (y_pred == y_test).sum()
    print(f"\nSummary:")
    print(f"Total test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")

    # Calculate cross-validation on full dataset to get stability metrics
    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION STABILITY CHECK")
    print('=' * 60)

    # Prepare full dataset for cross-validation
    x_all = data['text']
    y_all = data['label']

    # Preprocess full dataset
    if use_advanced and preprocessor:
        x_all_processed = preprocessor.transform(x_all)
        x_all_features = feature_extraction.transform(x_all_processed)
    else:
        x_all_features = feature_extraction.transform(x_all)

    # Perform cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x_all_features, y_all, cv=skf, scoring='f1')

    print(f"5-Fold Cross-Validation F1-Scores: {cv_scores}")
    print(f"Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Feature importance for logistic regression
    if hasattr(model, 'coef_'):
        print(f"\n{'=' * 60}")
        print("TOP FEATURES (Most Important Words)")
        print('=' * 60)

        feature_names = feature_extraction.get_feature_names_out()
        coefficients = model.coef_[0]

        # Get top 10 spam indicators (positive coefficients for spam class)
        # Note: In our encoding, spam=0, ham=1, so negative coefficients indicate spam
        spam_features = []
        ham_features = []

        for name, coef in zip(feature_names, coefficients):
            if coef < -0.5:  # Strong spam indicator
                spam_features.append((name, abs(coef)))
            elif coef > 0.5:  # Strong ham indicator
                ham_features.append((name, coef))

        spam_features.sort(key=lambda x: x[1], reverse=True)
        ham_features.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 10 Spam Indicators:")
        for word, importance in spam_features[:10]:
            print(f"  {word}: {importance:.4f}")

        print("\nTop 10 Ham Indicators:")
        for word, importance in ham_features[:10]:
            print(f"  {word}: {importance:.4f}")


def compare_models():
    """Compare multiple models with cross-validation"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from data_preprocessing import load_and_combine_data, preprocess_data

    print("\n" + "=" * 60)
    print("MODEL COMPARISON WITH CROSS-VALIDATION")
    print("=" * 60)

    # Load data
    data = load_and_combine_data()
    if data is None:
        return

    # Preprocess data
    x_train_features, x_test_features, y_train, y_test, feature_extraction, _ = preprocess_data(data)

    if x_train_features is None:
        return

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB()
    }

    # Perform cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, x_train_features, y_train, cv=skf, scoring='f1')
        results[model_name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        print(f"\n{model_name}:")
        print(f"  Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Fold scores: {cv_scores}")

    # Best model
    best_model = max(results.keys(), key=lambda x: results[x]['mean'])
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_model} with F1-Score: {results[best_model]['mean']:.4f}")
    print('=' * 60)


if __name__ == "__main__":
    evaluate_model()
