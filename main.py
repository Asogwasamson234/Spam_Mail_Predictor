import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import pickle
import sys
import os


class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Mail Detector")
        self.root.geometry("700x600")
        self.root.configure(bg='#f0f0f0')

        # Statistics
        self.predictions = {'spam': 0, 'ham': 0}

        # Load model
        self.load_model()

        # Create GUI elements
        self.create_widgets()

    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists('feature_extraction.pkl'):
                messagebox.showerror("Error",
                                     "feature_extraction.pkl not found!\nPlease run train_model.py first.")
                sys.exit(1)

            if not os.path.exists('spam_model.pkl'):
                messagebox.showerror("Error",
                                     "spam_model.pkl not found!\nPlease run train_model.py first.")
                sys.exit(1)

            with open('feature_extraction.pkl', 'rb') as f:
                self.feature_extraction = pickle.load(f)

            with open('spam_model.pkl', 'rb') as f:
                self.model = pickle.load(f)

            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            sys.exit(1)

    def create_widgets(self):
        """Create GUI widgets"""

        # Title
        title_label = tk.Label(self.root, text="SPAM MAIL DETECTION SYSTEM",
                               font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=10)

        # Subtitle
        subtitle_label = tk.Label(self.root,
                                  text="Powered by Machine Learning | Trained on 5000+ messages",
                                  font=("Arial", 9), bg='#f0f0f0', fg='#666')
        subtitle_label.pack(pady=5)

        # Instruction
        instruction_label = tk.Label(self.root,
                                     text="Enter the email/message content below:",
                                     font=("Arial", 11), bg='#f0f0f0')
        instruction_label.pack(pady=10)

        # Text input area
        text_frame = tk.Frame(self.root, bg='white', relief=tk.GROOVE, bd=2)
        text_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.text_area = scrolledtext.ScrolledText(text_frame,
                                                   width=70, height=12, font=("Arial", 10), wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Buttons frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=15)

        # Detect button
        detect_btn = tk.Button(button_frame, text="DETECT SPAM",
                               command=self.detect_spam, bg='#4CAF50', fg='white',
                               font=("Arial", 12, "bold"), padx=25, pady=8, cursor='hand2')
        detect_btn.pack(side=tk.LEFT, padx=10)

        # Clear button
        clear_btn = tk.Button(button_frame, text="CLEAR",
                              command=self.clear_text, bg='#f44336', fg='white',
                              font=("Arial", 12, "bold"), padx=25, pady=8, cursor='hand2')
        clear_btn.pack(side=tk.LEFT, padx=10)

        # Reset stats button
        reset_btn = tk.Button(button_frame, text="RESET STATS",
                              command=self.reset_stats, bg='#FF9800', fg='white',
                              font=("Arial", 10, "bold"), padx=15, pady=8, cursor='hand2')
        reset_btn.pack(side=tk.LEFT, padx=10)

        # Result frame
        result_frame = tk.Frame(self.root, bg='#f0f0f0', relief=tk.GROOVE, bd=2)
        result_frame.pack(pady=10, padx=20, fill=tk.X)

        # Result label
        self.result_label = tk.Label(result_frame, text="",
                                     font=("Arial", 16, "bold"), bg='#f0f0f0', pady=10)
        self.result_label.pack()

        # Confidence label
        self.confidence_label = tk.Label(result_frame, text="",
                                         font=("Arial", 11), bg='#f0f0f0')
        self.confidence_label.pack()

        # Statistics frame
        stats_frame = tk.Frame(self.root, bg='#f0f0f0')
        stats_frame.pack(pady=10)

        self.stats_label = tk.Label(stats_frame, text="Predictions made: 0 | Spam: 0 | Ham: 0",
                                    font=("Arial", 9), bg='#f0f0f0', fg='#666')
        self.stats_label.pack()

        # Sample messages button
        sample_btn = tk.Button(self.root, text="Load Sample Messages",
                               command=self.load_samples, bg='#2196F3', fg='white',
                               font=("Arial", 10), padx=15, pady=5, cursor='hand2')
        sample_btn.pack(pady=5)

        # Footer
        footer_label = tk.Label(self.root,
                                text="Note: This is an AI-powered tool. Always use your judgment for important communications.",
                                font=("Arial", 8), bg='#f0f0f0', fg='#999')
        footer_label.pack(side=tk.BOTTOM, pady=5)

    def detect_spam(self):
        """Detect spam from input text"""
        text = self.text_area.get("1.0", tk.END).strip()

        if not text:
            messagebox.showwarning("Warning", "Please enter a message to analyze!")
            return

        try:
            # Transform text to features
            text_features = self.feature_extraction.transform([text])

            # Make prediction
            prediction = self.model.predict(text_features)[0]
            probability = self.model.predict_proba(text_features)[0]

            # Display result
            if prediction == 1:
                result_text = "HAM - NOT SPAM"
                confidence = probability[1] * 100
                color = '#4CAF50'
                self.predictions['ham'] += 1
                detail_msg = "This message appears to be legitimate (HAM)."
            else:
                result_text = "SPAM DETECTED"
                confidence = probability[0] * 100
                color = '#f44336'
                self.predictions['spam'] += 1
                detail_msg = "WARNING: This message appears to be SPAM! Be cautious!\nDo not click on suspicious links or share personal information."

            self.result_label.config(text=result_text, fg=color)
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

            # Update statistics
            total = self.predictions['spam'] + self.predictions['ham']
            self.stats_label.config(
                text=f"Predictions made: {total} | Spam: {self.predictions['spam']} | Ham: {self.predictions['ham']}"
            )

            # Show detail message
            messagebox.showinfo("Analysis Result",
                                f"{result_text}\n\nConfidence: {confidence:.2f}%\n\n{detail_msg}")

            # Print to console
            print(f"\n[Analysis]")
            print(f"Message: {text[:100]}...")
            print(f"Result: {result_text}")
            print(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze message: {e}")

    def clear_text(self):
        """Clear the text area"""
        self.text_area.delete("1.0", tk.END)
        self.result_label.config(text="")
        self.confidence_label.config(text="")

    def reset_stats(self):
        """Reset prediction statistics"""
        self.predictions = {'spam': 0, 'ham': 0}
        self.stats_label.config(text="Predictions made: 0 | Spam: 0 | Ham: 0")
        messagebox.showinfo("Statistics Reset", "Prediction statistics have been reset!")

    def load_samples(self):
        """Load sample messages for testing"""
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

        # Create sample selection window
        sample_window = tk.Toplevel(self.root)
        sample_window.title("Sample Messages")
        sample_window.geometry("500x450")
        sample_window.configure(bg='#f0f0f0')

        tk.Label(sample_window, text="Select a sample message to test:",
                 font=("Arial", 12, "bold"), bg='#f0f0f0').pack(pady=10)

        # Create frame for samples
        samples_frame = tk.Frame(sample_window, bg='#f0f0f0')
        samples_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Add scrollbar
        scrollbar = tk.Scrollbar(samples_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(samples_frame, width=60, height=12,
                             font=("Arial", 9), yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        for i, (msg, label) in enumerate(samples, 1):
            display_text = f"[{label.upper()}] {msg[:80]}{'...' if len(msg) > 80 else ''}"
            listbox.insert(tk.END, display_text)

        def load_selected():
            selection = listbox.curselection()
            if selection:
                selected_text = samples[selection[0]][0]
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert("1.0", selected_text)
                sample_window.destroy()
                messagebox.showinfo("Sample Loaded",
                                    f"Sample message loaded!\n\nClick 'DETECT SPAM' to analyze it.")

        load_btn = tk.Button(sample_window, text="Load Selected Message",
                             command=load_selected, bg='#4CAF50', fg='white',
                             font=("Arial", 10, "bold"), padx=15, pady=5)
        load_btn.pack(pady=10)


def main():
    root = tk.Tk()
    app = SpamDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
