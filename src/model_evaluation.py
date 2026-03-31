from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


def evaluate_model():
    """Evaluate the trained model"""

    # Load model and feature extraction
    try:
        with open('feature_extraction.pkl', 'rb') as f:
            feature_extraction = pickle.load(f)

        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)

        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model files not found! Please run train_model.py first.")
        return

    # Load the data for evaluation
    from data_preprocessing import load_and_combine_data, preprocess_data

    data = load_and_combine_data()
    _, x_test_features, _, y_test, _ = preprocess_data(data)

    # Make predictions
    prediction_on_test_data = model.predict(x_test_features)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, prediction_on_test_data)
    print(f"\nModel Accuracy on Test Data: {accuracy:.4f}")

    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, prediction_on_test_data,
                                target_names=['Spam', 'Ham']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, prediction_on_test_data)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Spam correctly identified): {cm[0][0]}")
    print(f"False Positives (Ham incorrectly as Spam): {cm[0][1]}")
    print(f"False Negatives (Spam incorrectly as Ham): {cm[1][0]}")
    print(f"True Positives (Ham correctly identified): {cm[1][1]}")


if __name__ == "__main__":
    evaluate_model()
