import os
import subprocess
import sys
import io

# Set encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'-' * 60}")
    print(f">>> {description}")
    print(f"{'-' * 60}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings:\n{result.stderr}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("SPAM MAIL DETECTION SYSTEM - SETUP AND RUN")
    print("=" * 60)

    print("\nThis script will help you:")
    print("1. Install required packages")
    print("2. Train the model")
    print("3. Run the GUI application")

    # Check if packages are installed
    print("\n" + "=" * 60)
    print("CHECKING PACKAGES")
    print("=" * 60)

    try:
        import pandas
        import sklearn
        print("[OK] Required packages are installed")
    except ImportError:
        print("[WARNING] Required packages are missing!")
        print("\nInstalling required packages...")
        run_command("pip install -r requirements.txt", "Installing dependencies")

    # Train the model
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    print("\nThis will train the spam detection model...")

    if run_command("python train_model.py", "Training model"):
        print("\n[OK] Model training completed!")
    else:
        print("\n[ERROR] Model training failed!")
        sys.exit(1)

    # Ask user which interface to run
    print("\n" + "=" * 60)
    print("SELECT INTERFACE")
    print("=" * 60)
    print("\n1. GUI Application (Recommended - User friendly)")
    print("2. Command-line Interface (Lightweight)")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == '1':
        print("\nLaunching GUI Application...")
        run_command("python main.py", "Starting GUI")
    elif choice == '2':
        print("\nLaunching Command-line Interface...")
        run_command("python predict.py", "Starting CLI")
    else:
        print("\nExiting...")

    print("\n" + "=" * 60)
    print("Thank you for using Spam Detection System!")
    print("=" * 60)


if __name__ == "__main__":
    main()
