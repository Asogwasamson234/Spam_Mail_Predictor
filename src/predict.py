import pickle
import sys
import os
import io

# Set encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_model():
    """Load the trained model and feature extractor"""
    try:
        if not os.path.exists('feature_extraction.pkl'):
            print("[ERROR] feature_extraction.pkl not found!")
            return None, None

        if not os.path.exists('spam_model.pkl'):
            print("[ERROR] spam_model.pkl not found!")
            return None, None

        with open('feature_extraction.pkl', 'rb') as f:
            feature_extraction = pickle.load(f)

        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)

        print("[OK] Model loaded successfully!")
        return model, feature_extraction
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print("Please run train_model.py first to train and save the model.")
        return None, None


def predict_mail(model, feature_extraction, mail_text):
    """Predict whether a mail is spam or ham"""
    try:
        # Convert text to feature vector
        mail_features = feature_extraction.transform([mail_text])

        # Make prediction
        prediction = model.predict(mail_features)[0]
        probability = model.predict_proba(mail_features)[0]

        # Interpret result
        if prediction == 1:
            result = "HAM (Not Spam)"
            confidence = probability[1] * 100
            indicator = "[OK]"
        else:
            result = "SPAM"
            confidence = probability[0] * 100
            indicator = "[!]"

        return result, confidence, indicator
    except Exception as e:
        return f"Error: {e}", 0, "[X]"


def main():
    """Main function for user input"""

    print("\n" + "=" * 60)
    print("SPAM MAIL DETECTION SYSTEM")
    print("=" * 60)
    print("\nThis system detects whether an email or message is spam or ham.")
    print("You can type your message below or use the sample options.")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'sample' - Show sample messages")
    print("  'help' - Show this help message")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, feature_extraction = load_model()

    if model is None:
        return

    # Statistics counter
    predictions = {'spam': 0, 'ham': 0}

    # Sample messages for testing
    samples = [
        ("Congratulations! You've won a $1000 gift card! Click here to claim now!", "spam"),
        ("Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.", "ham"),
        ("URGENT: Your account has been compromised. Verify your details immediately.", "spam"),
        ("I had a great time at the party last night. We should do it again soon!", "ham"),
        ("FREE entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121 now!", "spam"),
        ("Can you please send me the report by 5 PM today? Thanks!", "ham"),
        ("WINNER!! As a valued customer you have been selected to receive a 900 pound prize!", "spam"),
        ("Don't forget to bring snacks for the movie night.", "ham")
    ]

    while True:
        print("\n" + "-" * 60)
        user_input = input("\nEnter your message (or type 'sample', 'quit', 'help'):\n> ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            print(f"Total predictions made: {sum(predictions.values())}")
            print(f"Spam detected: {predictions['spam']}")
            print(f"Ham detected: {predictions['ham']}")
            print("\nThank you for using Spam Detection System. Goodbye!")
            break

        if user_input.lower() == 'help':
            print("\n" + "=" * 60)
            print("HELP")
            print("=" * 60)
            print("Simply type or paste your message and press Enter.")
            print("The system will analyze it and tell you if it's spam or ham.")
            print("\nCommands:")
            print("  'quit' or 'exit' - Exit the program")
            print("  'sample' - Show sample messages to test")
            print("  'help' - Show this help message")
            continue

        if user_input.lower() == 'sample':
            print("\n" + "=" * 60)
            print("SAMPLE MESSAGES")
            print("=" * 60)
            for i, (msg, label) in enumerate(samples, 1):
                print(f"\n{i}. [{label.upper()}] {msg[:100]}{'...' if len(msg) > 100 else ''}")

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
        result, confidence, indicator = predict_mail(model, feature_extraction, user_input)

        # Display result
        print("\n" + "=" * 60)
        print(f"{indicator} RESULT: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print("=" * 60)

        # Update statistics
        if result == "SPAM":
            predictions['spam'] += 1
            print("\n[!] WARNING: This message appears to be SPAM! Be cautious!")
            print("    Do not click on suspicious links or share personal information.")
        else:
            predictions['ham'] += 1
            print("\n[OK] This message appears to be HAM (legitimate).")

        # Show additional details
        print(f"\nMessage length: {len(user_input)} characters")
        print(f"Word count: {len(user_input.split())} words")


if __name__ == "__main__":
    main()
