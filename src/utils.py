"""
Utility functions for the spam detection system.
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np


def setup_logging(log_dir='logs', log_file='spam_detector.log'):
    """
    Set up logging configuration.

    Parameters:
    -----------
    log_dir : str
        Directory to store log files
    log_file : str
        Name of the log file

    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('spam_detector')
    logger.setLevel(logging.INFO)

    # Create file handler
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_prediction_history(predictions, file_path='logs/predictions.json'):
    """
    Save prediction history to a JSON file.

    Parameters:
    -----------
    predictions : list
        List of prediction dictionaries
    file_path : str
        Path to save predictions
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load existing predictions if file exists
    existing = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            pass

    # Append new predictions
    all_predictions = existing + predictions

    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    return len(all_predictions)


def load_prediction_history(file_path='logs/predictions.json'):
    """
    Load prediction history from JSON file.

    Returns:
    --------
    list : List of prediction dictionaries
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def get_model_info(models_dir='models'):
    """
    Get information about the trained model.

    Parameters:
    -----------
    models_dir : str
        Directory containing model files

    Returns:
    --------
    dict : Model information
    """
    info = {
        'model_exists': False,
        'preprocessor_exists': False,
        'config_exists': False,
        'config': {}
    }

    # Check for model files
    model_path = os.path.join(models_dir, 'spam_model.pkl')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    config_path = os.path.join(models_dir, 'preprocessing_config.txt')

    info['model_exists'] = os.path.exists(model_path)
    info['preprocessor_exists'] = os.path.exists(preprocessor_path)
    info['config_exists'] = os.path.exists(config_path)

    # Load config if exists
    if info['config_exists']:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()
                for line in config_text.split('\n'):
                    if ':' in line and not line.startswith('='):
                        key, value = line.split(':', 1)
                        info['config'][key.strip()] = value.strip()
        except:
            pass

    return info


def calculate_metrics(y_true, y_pred):
    """
    Calculate various classification metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    dict : Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }


def print_section(title, char='=', length=60):
    """
    Print a formatted section header.

    Parameters:
    -----------
    title : str
        Section title
    char : str
        Character to use for the line
    length : int
        Length of the line
    """
    print("\n" + char * length)
    print(title.upper())
    print(char * length)


def validate_file_path(file_path):
    """
    Validate that a file path exists and is readable.

    Parameters:
    -----------
    file_path : str
        Path to the file

    Returns:
    --------
    bool : True if file exists and readable, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False

    if not os.access(file_path, os.R_OK):
        print(f"[ERROR] File not readable: {file_path}")
        return False

    return True


def format_time(seconds):
    """
    Format seconds into a readable time string.

    Parameters:
    -----------
    seconds : float
        Number of seconds

    Returns:
    --------
    str : Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def get_dataset_stats(data):
    """
    Get statistics about the dataset.

    Parameters:
    -----------
    data : DataFrame
        Dataset with 'text' and 'label' columns

    Returns:
    --------
    dict : Dataset statistics
    """
    stats = {
        'total_samples': len(data),
        'spam_count': len(data[data['label'] == 0]),
        'ham_count': len(data[data['label'] == 1]),
        'spam_ratio': len(data[data['label'] == 0]) / len(data) * 100,
        'ham_ratio': len(data[data['label'] == 1]) / len(data) * 100
    }

    # Text statistics
    data['text_length'] = data['text'].str.len()
    data['word_count'] = data['text'].str.split().str.len()

    stats['avg_text_length'] = data['text_length'].mean()
    stats['avg_word_count'] = data['word_count'].mean()
    stats['max_text_length'] = data['text_length'].max()
    stats['min_text_length'] = data['text_length'].min()

    return stats


def create_visualization(y_test, y_pred, save_path=None):
    """
    Create and save visualization of model performance.

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    save_path : str, optional
        Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticklabels(['Spam', 'Ham'])
    axes[0].set_yticklabels(['Spam', 'Ham'])

    # Class distribution
    labels = ['Spam', 'Ham']
    true_counts = [sum(y_test == 0), sum(y_test == 1)]
    pred_counts = [sum(y_pred == 0), sum(y_pred == 1)]

    x = np.arange(len(labels))
    width = 0.35

    axes[1].bar(x - width / 2, true_counts, width, label='True', alpha=0.8)
    axes[1].bar(x + width / 2, pred_counts, width, label='Predicted', alpha=0.8)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].set_title('True vs Predicted Distribution')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[OK] Visualization saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test logging
    logger = setup_logging()
    logger.info("Testing logging functionality")

    # Test model info
    info = get_model_info()
    print(f"\nModel Info: {info}")

    print("\n[OK] Utility functions loaded successfully!")
