from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import load_and_combine_data, preprocess_data, save_model_data
import time
import sys
import io

# Set encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def train_and_evaluate():
    """Train the spam detection model"""

    print("\n" + "=" * 60)
    print("SPAM MAIL DETECTION - MODEL TRAINING")
    print("=" * 60)

    start_time = time.time()

    try:
        # Load and combine data
        print("\n[1/4] Loading and combining datasets...")
        data = load_and_combine_data()

        # Preprocess data
        print("\n[2/4] Preprocessing data...")
        x_train_features, x_test_features, y_train, y_test, feature_extraction = preprocess_data(data)

        # Train the model
        print("\n[3/4] Training Logistic Regression model...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_train_features, y_train)
        print("[OK] Model training completed!")

        # Evaluate on training data
        print("\n[4/4] Evaluating model performance...")
        print("\n" + "-" * 60)
        print("PERFORMANCE METRICS")
        print("-" * 60)

        # Training accuracy
        y_train_pred = model.predict(x_train_features)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"\n[OK] Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")

        # Test accuracy
        y_test_pred = model.predict(x_test_features)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"[OK] Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        # Detailed classification report
        print("\n" + "-" * 60)
        print("CLASSIFICATION REPORT (Test Data)")
        print("-" * 60)
        print(classification_report(y_test, y_test_pred, target_names=['Spam', 'Ham']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print("\n" + "-" * 60)
        print("CONFUSION MATRIX")
        print("-" * 60)
        print(f"True Negatives (Spam correctly identified): {cm[0][0]}")
        print(f"False Positives (Ham incorrectly as Spam): {cm[0][1]}")
        print(f"False Negatives (Spam incorrectly as Ham): {cm[1][0]}")
        print(f"True Positives (Ham correctly identified): {cm[1][1]}")

        # Calculate additional metrics
        precision_spam = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        recall_spam = cm[0][0] / (cm[0][0] + cm[1][0]) if (cm[0][0] + cm[1][0]) > 0 else 0
        precision_ham = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        recall_ham = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0

        print(f"\nAdditional Metrics:")
        print(f"Precision (Spam): {precision_spam:.4f}")
        print(f"Recall (Spam): {recall_spam:.4f}")
        print(f"Precision (Ham): {precision_ham:.4f}")
        print(f"Recall (Ham): {recall_ham:.4f}")

        # Save model and feature extraction
        save_model_data(feature_extraction, model)

        # Training time
        elapsed_time = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print("=" * 60)

        return model, feature_extraction

    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    train_and_evaluate()
