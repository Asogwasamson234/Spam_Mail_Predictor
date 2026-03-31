import pickle
import sys
import os
import io

# Set encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_model(models_dir='models'):
    """
    Load the trained model, feature extractor, and preprocessor.

    Parameters:
    -----------
    models_dir : str
        Directory where models are stored

    Returns:
    --------
    tuple : (model, feature_extraction, preprocessor)
    """
    try:
        # Check for model files
        feature_path = os.path.join(models_dir, 'feature_extraction.pkl')
        model_path = os.path.join(models_dir, 'spam_model.pkl')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')

        if not os.path.exists(feature_path):
            print(f"[ERROR] feature_extraction.pkl not found in {models_dir}!")
            return None, None, None

        if not os.path.exists(model_path):
            print(f"[ERROR] spam_model.pkl not found in {models_dir}!")
            return None, None, None

        # Load feature extraction
        with open(feature_path, 'rb') as f:
            feature_extraction = pickle.load(f)

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load preprocessor if exists
        preprocessor = None
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print("[OK] Advanced preprocessor loaded!")
        else:
            print("[OK] Using basic preprocessing (no stemming/lemmatization)")

        print("[OK] Model loaded successfully!")
        return model, feature_extraction, preprocessor

    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print("Please run train_model.py first to train and save the model.")
        return None, None, None


def preprocess_input_text(text, preprocessor=None):
    """
    Preprocess input text using the same method as training.

    Parameters:
    -----------
    text : str
        Input text to preprocess
    preprocessor : AdvancedTextPreprocessor, optional
        Preprocessor instance from training

    Returns:
    --------
    str : Preprocessed text
    """
    if not text:
        return text

    if preprocessor:
        # Use advanced preprocessing
        processed = preprocessor.transform([text])
        return processed[0] if processed else text
    else:
        # Basic preprocessing (lowercase and strip)
        return text.lower().strip()


def predict_mail(model, feature_extraction, mail_text, preprocessor=None):
    """
    Predict whether a mail is spam or ham.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    feature_extraction : TfidfVectorizer
        Fitted TF-IDF vectorizer
    mail_text : str
        Input text to predict
    preprocessor : AdvancedTextPreprocessor, optional
        Preprocessor instance from training

    Returns:
    --------
    tuple : (result, confidence, indicator)
    """
    try:
        # Preprocess the text
        processed_text = preprocess_input_text(mail_text, preprocessor)

        # Convert text to feature vector
        mail_features = feature_extraction.transform([processed_text])

        # Make prediction
        prediction = model.predict(mail_features)[0]
        probability = model.predict_proba(mail_features)[0]

        # Interpret result
        # Note: In our encoding, spam=0, ham=1
        if prediction == 1:
            result = "HAM (Not Spam)"
            confidence = probability[1] * 100
            indicator = "[OK]"
        else:
            result = "SPAM"
            confidence = probability[0] * 100
            indicator = "[!]"

        return result, confidence, indicator, probability

    except Exception as e:
        return f"Error: {e}", 0, "[X]", None


def analyze_text_statistics(text):
    """
    Analyze text statistics for additional insights.

    Parameters:
    -----------
    text : str
        Input text to analyze

    Returns:
    --------
    dict : Text statistics
    """
    words = text.split()

    # Calculate spam indicators
    has_url = 'http' in text.lower() or 'www.' in text.lower()
    has_exclamation = text.count('!')
    has_all_caps = any(word.isupper() and len(word) > 1 for word in words)
    has_numbers = any(char.isdigit() for char in text)

    # Calculate percentages
    caps_percentage = sum(1 for word in words if word.isupper() and len(word) > 1) / len(words) * 100 if words else 0

    return {
        'char_count': len(text),
        'word_count': len(words),
        'has_url': has_url,
        'has_exclamation': has_exclamation,
        'has_all_caps': has_all_caps,
        'caps_percentage': round(caps_percentage, 2),
        'has_numbers': has_numbers,
        'exclamation_count': has_exclamation
    }


def display_analysis(text, result, confidence, probability, statistics):
    """
    Display detailed analysis of the prediction.
    """
    print("\n" + "=" * 60)
    print(f"{result[0]} RESULT: {result[1]}")
    print(f"Confidence: {confidence:.2f}%")

    # Show probability distribution
    if probability is not None:
        print(f"Probability breakdown:")
        print(f"  Spam: {probability[0] * 100:.2f}%")
        print(f"  Ham:  {probability[1] * 100:.2f}%")

    print("=" * 60)

    # Show text statistics
    print("\n[INFO] Text Analysis:")
    print(f"  Character count: {statistics['char_count']}")
    print(f"  Word count: {statistics['word_count']}")
    print(f"  Contains URL: {'Yes' if statistics['has_url'] else 'No'}")
    print(f"  Contains numbers: {'Yes' if statistics['has_numbers'] else 'No'}")
    print(f"  ALL CAPS words: {statistics['caps_percentage']}% of words")
    print(f"  Exclamation marks: {statistics['exclamation_count']}")

    # Spam indicators warning
    spam_indicators = []
    if statistics['has_url']:
        spam_indicators.append("URLs")
    if statistics['has_exclamation'] > 2:
        spam_indicators.append("multiple exclamation marks")
    if statistics['caps_percentage'] > 30:
        spam_indicators.append("excessive capitalization")
    if statistics['has_numbers']:
        spam_indicators.append("numbers")

    if spam_indicators and result[1] == "SPAM":
        print(f"\n[!] Spam indicators detected: {', '.join(spam_indicators)}")


def show_help():
    """Display help information."""
    print("\n" + "=" * 60)
    print("HELP")
    print("=" * 60)
    print("Simply type or paste your message and press Enter.")
    print("The system will analyze it and tell you if it's spam or ham.")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'sample' - Show sample messages to test")
    print("  'stats' - Show current session statistics")
    print("  'help' - Show this help message")
    print("\nThe system uses advanced text preprocessing including:")
    print("  - Stemming/Lemmatization (if available)")
    print("  - Stopword removal")
    print("  - URL and email detection")
    print("  - TF-IDF feature extraction")


def show_samples(samples):
    """Display sample messages."""
    print("\n" + "=" * 60)
    print("SAMPLE MESSAGES")
    print("=" * 60)
    for i, (msg, label) in enumerate(samples, 1):
        preview = msg[:80] + '...' if len(msg) > 80 else msg
        print(f"\n{i}. [{label.upper()}] {preview}")


def main():
    """Main function for user input"""
    print("\n" + "=" * 60)
    print("SPAM MAIL DETECTION SYSTEM (Enhanced)")
    print("=" * 60)
    print("\nThis system detects whether an email or message is spam or ham.")
    print("It uses advanced preprocessing (stemming/lemmatization) for better accuracy.")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'sample' - Show sample messages")
    print("  'stats' - Show session statistics")
    print("  'help' - Show this help message")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, feature_extraction, preprocessor = load_model()

    if model is None:
        return

    # Show preprocessing info
    if preprocessor:
        print("\n[INFO] Using advanced preprocessing with:")
        preprocess_info = preprocessor.get_info()
        for key, value in preprocess_info.items():
            print(f"  {key}: {value}")
    else:
        print("\n[INFO] Using basic preprocessing")

    # Statistics counter
    predictions = {'spam': 0, 'ham': 0, 'total': 0}
    confidence_scores = []

    # Sample messages for testing
    samples = [
        ("Congratulations! You've won a $1000 gift card! Click here to claim now!", "spam"),
        ("Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.", "ham"),
        ("URGENT: Your account has been compromised! Verify your details immediately at http://fake.com", "spam"),
        ("I had a great time at the party last night. We should do it again soon!", "ham"),
        ("FREE entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121 now!", "spam"),
        ("Can you please send me the report by 5 PM today? Thanks!", "ham"),
        ("WINNER!! As a valued customer you have been selected to receive a 900 pound prize!", "spam"),
        ("Don't forget to bring snacks for the movie night.", "ham")
    ]

    while True:
        print("\n" + "-" * 60)
        user_input = input("\nEnter your message (or type 'sample', 'quit', 'help', 'stats'):\n> ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            print(f"Total predictions made: {predictions['total']}")
            print(f"Spam detected: {predictions['spam']}")
            print(f"Ham detected: {predictions['ham']}")
            if confidence_scores:
                print(f"Average confidence: {sum(confidence_scores) / len(confidence_scores):.2f}%")
            print("\nThank you for using Spam Detection System. Goodbye!")
            break

        if user_input.lower() == 'help':
            show_help()
            continue

        if user_input.lower() == 'stats':
            print("\n" + "=" * 60)
            print("SESSION STATISTICS")
            print("=" * 60)
            print(f"Total predictions: {predictions['total']}")
            print(
                f"Spam: {predictions['spam']} ({predictions['spam'] / predictions['total'] * 100:.1f}%)" if predictions[
                                                                                                                'total'] > 0 else "Spam: 0")
            print(f"Ham: {predictions['ham']} ({predictions['ham'] / predictions['total'] * 100:.1f}%)" if predictions[
                                                                                                               'total'] > 0 else "Ham: 0")
            if confidence_scores:
                print(f"Average confidence: {sum(confidence_scores) / len(confidence_scores):.2f}%")
            continue

        if user_input.lower() == 'sample':
            show_samples(samples)
            print("\nEnter the number (1-8) to test a sample, or type your own message:")
            sample_choice = input("> ").strip()

            if sample_choice.isdigit() and 1 <= int(sample_choice) <= len(samples):
                sample_text, expected = samples[int(sample_choice) - 1]
                user_input = sample_text
                print(f"\nTesting sample message: {sample_text[:100]}...")
            else:
                continue

        if not user_input:
            print("[WARNING] Please enter a valid message!")
            continue

        # Make prediction
        result, confidence, indicator, probability = predict_mail(
            model, feature_extraction, user_input, preprocessor
        )

        # Analyze text statistics
        statistics = analyze_text_statistics(user_input)

        # Display analysis
        display_analysis(user_input, (indicator, result), confidence, probability, statistics)

        # Update statistics
        if result == "SPAM":
            predictions['spam'] += 1
            predictions['total'] += 1
            confidence_scores.append(confidence)
            print("\n[!] WARNING: This message appears to be SPAM! Be cautious!")
            print("    Do not click on suspicious links or share personal information.")
        elif result == "HAM (Not Spam)":
            predictions['ham'] += 1
            predictions['total'] += 1
            confidence_scores.append(confidence)
            print("\n[OK] This message appears to be HAM (legitimate).")
        else:
            print(f"\n[X] {result}")


if __name__ == "__main__":
    main()
