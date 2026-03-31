"""
SPAM MAIL DETECTION SYSTEM - Main Launcher
=============================================
This script serves as the main entry point for the spam detection system.
It handles:
- Package installation and dependency management
- Model training with cross-validation
- Model evaluation and performance metrics
- Launching GUI or CLI interfaces

Author: Spam Detection Team
Version: 2.0 (Enhanced with Advanced Preprocessing & Cross-validation)
"""

import os
import sys
import io
import subprocess
import platform
from datetime import datetime

# Set encoding for Windows console to handle Unicode characters properly
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def print_header(title, char='=', length=70):
    """
    Print a formatted header with the given title.

    Parameters:
    -----------
    title : str
        The title to display
    char : str
        Character used for the border
    length : int
        Length of the border line
    """
    print("\n" + char * length)
    print(f"{title.upper():^{length}}")
    print(char * length)


def print_section(title, char='-', length=60):
    """
    Print a formatted section header.

    Parameters:
    -----------
    title : str
        The section title
    char : str
        Character used for the border
    length : int
        Length of the border line
    """
    print(f"\n{char * length}")
    print(f">>> {title}")
    print(char * length)


def run_command(command, description, cwd=None, capture_output=True):
    """
    Execute a system command with proper error handling.

    Parameters:
    -----------
    command : str
        The command to execute
    description : str
        Description of what the command does (for display)
    cwd : str, optional
        Working directory for the command
    capture_output : bool
        Whether to capture and display output

    Returns:
    --------
    bool : True if command succeeded, False otherwise
    """
    print_section(description)

    try:
        if capture_output:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd
            )
            if result.stdout:
                # Limit output to avoid flooding the console
                output_lines = result.stdout.split('\n')
                if len(output_lines) > 20:
                    print('\n'.join(output_lines[:15]))
                    print(f"... (and {len(output_lines) - 15} more lines)")
                    print(f"\n[INFO] Full output available in the console")
                else:
                    print(result.stdout)

            if result.stderr:
                print(f"\n[WARNING] Issues encountered:\n{result.stderr}")

            return result.returncode == 0
        else:
            # For interactive commands, don't capture output
            result = subprocess.run(command, shell=True, cwd=cwd)
            return result.returncode == 0

    except Exception as e:
        print(f"[ERROR] Failed to execute command: {e}")
        return False


def check_python_version():
    """
    Verify that Python version is 3.7 or higher.

    Returns:
    --------
    bool : True if version is compatible
    """
    print_header("Checking Python Environment")

    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("[ERROR] Python 3.7 or higher is required!")
        print("Please upgrade Python before continuing.")
        return False

    print("[OK] Python version is compatible")
    return True


def check_required_packages():
    """
    Check if all required packages are installed.

    Returns:
    --------
    tuple : (all_installed, missing_packages)
    """
    print_header("Checking Required Packages")

    required_packages = {
        'pandas': 'pd',
        'numpy': 'np',
        'sklearn': 'sklearn',
        'nltk': 'nltk',
        'matplotlib': 'matplotlib',
        'seaborn': 'sns'
    }

    installed = []
    missing = []

    for package, import_name in required_packages.items():
        try:
            if import_name == 'pd':
                __import__('pandas')
            elif import_name == 'np':
                __import__('numpy')
            elif import_name == 'sklearn':
                __import__('sklearn')
            elif import_name == 'nltk':
                __import__('nltk')
            elif import_name == 'matplotlib':
                __import__('matplotlib')
            elif import_name == 'sns':
                __import__('seaborn')

            installed.append(package)
            print(f"[OK] {package} - installed")
        except ImportError:
            missing.append(package)
            print(f"[X] {package} - MISSING")

    if missing:
        print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
        return False, missing

    print("\n[OK] All required packages are installed!")
    return True, []


def install_packages(packages):
    """
    Install the specified packages using pip.

    Parameters:
    -----------
    packages : list
        List of package names to install

    Returns:
    --------
    bool : True if installation succeeded
    """
    print_header("Installing Required Packages")

    for package in packages:
        print(f"\nInstalling {package}...")
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"[ERROR] Failed to install {package}")
            return False

    # Download NLTK data after installation
    print("\n" + "-" * 60)
    print(">>> Downloading NLTK Data")
    print("-" * 60)

    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("[OK] NLTK data downloaded successfully")
    except Exception as e:
        print(f"[WARNING] Could not download NLTK data: {e}")
        print("You may need to download it manually later.")

    return True


def check_model_files():
    """
    Check if trained model files exist in the expected locations.

    Returns:
    --------
    tuple : (exists, missing_files, has_preprocessor)
    """
    print_header("Checking Model Files")

    # Check multiple locations for model files
    locations = ['models/', './', 'src/models/']
    required_files = ['feature_extraction.pkl', 'spam_model.pkl']
    preprocessor_file = 'preprocessor.pkl'

    found_files = []
    missing_files = []
    has_preprocessor = False

    for file in required_files:
        found = False
        for loc in locations:
            path = os.path.join(loc, file)
            if os.path.exists(path):
                found_files.append((file, loc))
                print(f"[OK] {file} found in {loc}")
                found = True
                break

        if not found:
            missing_files.append(file)
            print(f"[X] {file} - NOT FOUND")

    # Check for preprocessor (advanced preprocessing)
    for loc in locations:
        preprocessor_path = os.path.join(loc, preprocessor_file)
        if os.path.exists(preprocessor_path):
            has_preprocessor = True
            print(f"[OK] Advanced preprocessor found in {loc}")
            break

    if not has_preprocessor:
        print("[INFO] No advanced preprocessor found - using basic preprocessing")

    return len(missing_files) == 0, missing_files, has_preprocessor


def train_model(use_advanced=True):
    """
    Train the spam detection model.

    Parameters:
    -----------
    use_advanced : bool
        Whether to use advanced preprocessing (stemming/lemmatization)

    Returns:
    --------
    bool : True if training succeeded
    """
    print_header("Model Training Phase")

    if use_advanced:
        print("\n[INFO] Using ADVANCED preprocessing with:")
        print("  ✓ Lemmatization (WordNet)")
        print("  ✓ Stopword removal")
        print("  ✓ URL/Email masking")
        print("  ✓ 5-fold cross-validation")
    else:
        print("\n[INFO] Using BASIC preprocessing (no stemming/lemmatization)")

    # Check if training script exists
    if os.path.exists('src/train_model.py'):
        training_script = 'src/train_model.py'
    elif os.path.exists('train_model.py'):
        training_script = 'train_model.py'
    else:
        print("[ERROR] train_model.py not found!")
        return False

    print(f"\nRunning training script: {training_script}")
    print("This may take a few minutes depending on dataset size...")

    # Run training script
    if run_command(f"python {training_script}", "Training Model"):
        print("\n[SUCCESS] Model training completed!")

        # Verify model was created
        exists, missing, _ = check_model_files()
        if exists:
            print("[OK] Model files verified")
            return True
        else:
            print(f"[WARNING] Model training completed but some files are missing: {missing}")
            return True  # Still return True as training completed
    else:
        print("\n[ERROR] Model training failed!")
        return False


def evaluate_model():
    """
    Run model evaluation to display performance metrics.

    Returns:
    --------
    bool : True if evaluation succeeded
    """
    print_header("Model Evaluation")

    # Check if evaluation script exists
    if os.path.exists('src/model_evaluation.py'):
        eval_script = 'src/model_evaluation.py'
    elif os.path.exists('model_evaluation.py'):
        eval_script = 'model_evaluation.py'
    else:
        print("[ERROR] model_evaluation.py not found!")
        return False

    print("\nRunning model evaluation...")
    print("This will display:")
    print("  - Accuracy, Precision, Recall, F1-Score")
    print("  - Confusion Matrix")
    print("  - Cross-validation scores")
    print("  - Top spam/ham indicators")

    return run_command(f"python {eval_script}", "Evaluating Model Performance")


def show_model_info():
    """
    Display detailed information about the trained model.
    """
    print_header("Model Information")

    # Check models directory
    models_dir = 'models'
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    config_path = os.path.join(models_dir, 'preprocessing_config.txt')

    # Try to find model files
    model_locations = []
    for loc in ['models/', './', 'src/models/']:
        if os.path.exists(os.path.join(loc, 'spam_model.pkl')):
            model_locations.append(loc)

    if model_locations:
        print(f"\n[INFO] Model found in: {', '.join(model_locations)}")
    else:
        print("\n[INFO] No trained model found")
        return

    # Check preprocessing type
    if os.path.exists(preprocessor_path):
        print("\n[✓] Preprocessing Type: ADVANCED")
        print("    Features:")
        print("    - Lemmatization (WordNet)")
        print("    - Stopword removal")
        print("    - URL/Email masking")
        print("    - Number and punctuation removal")

        # Read config if exists
        if os.path.exists(config_path):
            print("\n[✓] Preprocessing Configuration:")
            with open(config_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line and not line.startswith('='):
                        print(f"    {line.strip()}")
    else:
        print("\n[✓] Preprocessing Type: BASIC")
        print("    Features:")
        print("    - Lowercase conversion")
        print("    - Stopword removal (English)")
        print("    - TF-IDF vectorization")

    # Get model file size
    model_path = os.path.join(models_dir, 'spam_model.pkl')
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"\n[✓] Model Size: {size_mb:.2f} MB")

    # Get last modified time
    if os.path.exists(model_path):
        modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"[✓] Last Trained: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")


def run_gui():
    """
    Launch the Graphical User Interface (GUI) application.
    """
    print_header("Launching GUI Application")

    # Check for GUI file (renamed from main.py to GUI.py)
    gui_files = ['GUI.py', 'gui.py', 'main.py', 'src/gui_app.py', 'src/main.py']
    gui_script = None

    for file in gui_files:
        if os.path.exists(file):
            gui_script = file
            print(f"[OK] Found GUI file: {file}")
            break

    if gui_script is None:
        print("[ERROR] GUI application file not found!")
        print("Expected files: GUI.py, main.py, or src/gui_app.py")
        return False

    print("\nStarting Spam Detection GUI...")
    print("The application window will open shortly.")
    print("If the window doesn't appear, check the console for errors.")

    return run_command(f"python {gui_script}", "Starting GUI Application", capture_output=False)


def run_cli():
    """
    Launch the Command-Line Interface (CLI) application.
    """
    print_header("Launching Command-Line Interface")

    # Check for CLI file
    cli_files = ['src/predict.py', 'predict.py']
    cli_script = None

    for file in cli_files:
        if os.path.exists(file):
            cli_script = file
            print(f"[OK] Found CLI file: {file}")
            break

    if cli_script is None:
        print("[ERROR] CLI application file not found!")
        print("Expected: predict.py or src/predict.py")
        return False

    print("\nStarting Spam Detection CLI...")
    print("You can type messages directly in the terminal.")

    return run_command(f"python {cli_script}", "Starting CLI Application", capture_output=False)


def show_welcome_message():
    """
    Display welcome message and system information.
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 15 + "SPAM MAIL DETECTION SYSTEM" + " " * 26 + "█")
    print("█" + " " * 68 + "█")
    print("█" + " " * 10 + "Enhanced with Advanced Preprocessing & Cross-validation" + " " * 8 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    print(f"\nSystem Information:")
    print(f"  Operating System: {platform.system()} {platform.release()}")
    print(f"  Python Version: {platform.python_version()}")
    print(f"  Working Directory: {os.getcwd()}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def show_menu():
    """
    Display the main menu and get user choice.

    Returns:
    --------
    int : User's menu choice
    """
    print_header("Main Menu", char='=', length=60)
    print("\nPlease select an option:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ 1. GUI Application    - User-friendly graphical interface │")
    print("  │ 2. CLI Application    - Lightweight command-line tool    │")
    print("  │ 3. Train Model        - Train/retrain the model          │")
    print("  │ 4. Evaluate Model     - View performance metrics         │")
    print("  │ 5. Model Information  - Show model details               │")
    print("  │ 6. Install Packages   - Install required dependencies    │")
    print("  │ 7. Exit               - Close the program                │")
    print("  └─────────────────────────────────────────────────────────┘")

    while True:
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7']:
                return int(choice)
            else:
                print("[WARNING] Invalid choice. Please enter a number between 1 and 7.")
        except KeyboardInterrupt:
            print("\n\n[INFO] Exiting...")
            return 7
        except Exception as e:
            print(f"[ERROR] Invalid input: {e}")


def main():
    """
    Main entry point for the spam detection system.
    """
    # Display welcome message
    show_welcome_message()

    # Check Python version first
    if not check_python_version():
        sys.exit(1)

    # Check required packages
    packages_installed, missing_packages = check_required_packages()

    # Handle missing packages
    if not packages_installed:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        response = input("\nDo you want to install missing packages? (y/n): ").strip().lower()

        if response == 'y':
            if not install_packages(missing_packages):
                print("[ERROR] Package installation failed!")
                sys.exit(1)
        else:
            print("\n[ERROR] Cannot proceed without required packages!")
            sys.exit(1)

    # Check for existing model
    model_exists, missing_files, has_preprocessor = check_model_files()

    if not model_exists:
        print(f"\n[INFO] Model files missing: {missing_files}")
        response = input("\nDo you want to train a new model? (y/n): ").strip().lower()

        if response == 'y':
            print("\nSelect preprocessing method:")
            print("  1. Advanced (Recommended) - Uses lemmatization and cross-validation")
            print("  2. Basic - Simple preprocessing without stemming/lemmatization")

            preprocess_choice = input("\nChoose (1-2): ").strip()
            use_advanced = preprocess_choice != '2'

            if not train_model(use_advanced):
                print("\n[ERROR] Model training failed!")
                sys.exit(1)
        else:
            print("\n[ERROR] Cannot proceed without trained model!")
            print("Please run option 3 to train the model.")

            # Ask if user wants to continue to menu anyway
            response = input("\nDo you want to continue to the main menu? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(1)

    # Main application loop
    while True:
        choice = show_menu()

        if choice == 1:  # GUI Application
            run_gui()
            break  # Exit after GUI closes

        elif choice == 2:  # CLI Application
            run_cli()
            break  # Exit after CLI closes

        elif choice == 3:  # Train Model
            print("\nSelect preprocessing method:")
            print("  1. Advanced (Recommended) - Uses lemmatization and cross-validation")
            print("  2. Basic - Simple preprocessing without stemming/lemmatization")

            preprocess_choice = input("\nChoose (1-2): ").strip()
            use_advanced = preprocess_choice != '2'

            train_model(use_advanced)

            # Re-check model files after training
            model_exists, _, _ = check_model_files()
            input("\nPress Enter to continue...")

        elif choice == 4:  # Evaluate Model
            if model_exists or check_model_files()[0]:
                evaluate_model()
            else:
                print("[ERROR] No trained model found! Please train the model first (option 3).")
            input("\nPress Enter to continue...")

        elif choice == 5:  # Model Information
            show_model_info()
            input("\nPress Enter to continue...")

        elif choice == 6:  # Install Packages
            install_packages(list(missing_packages) if missing_packages else
                             ['pandas', 'numpy', 'scikit-learn', 'nltk', 'matplotlib', 'seaborn'])
            input("\nPress Enter to continue...")

        elif choice == 7:  # Exit
            print_header("Goodbye!", char='=', length=60)
            print("\nThank you for using the Spam Detection System!")
            print("If you found this tool helpful, please consider sharing it.")
            print("\n" + "=" * 60)
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Program interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
