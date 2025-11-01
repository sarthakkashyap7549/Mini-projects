import customtkinter as ctk
import torch
import os
import sys
# Path to the trained model directory
MODEL_PATH = "model" 

# Use GPU if available (recommended for performance)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for model and tokenizer
tokenizer = None
model = None

class NewsSnifferApp(ctk.CTk):
    
    # Define placeholder text as a class attribute
    PLACEHOLDER_TEXT = "Enter news text here..."

    def __init__(self):
        super().__init__()

        # --- Basic App Setup ---
        self.title("Fake News Detection")
        self.geometry("800x650")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Set appearance theme (Light, Dark, System)
        ctk.set_appearance_mode("System")  
        ctk.set_default_color_theme("blue")

        # --- Load Model (Immediately on App Startup) ---
        self.load_model()
        
        # --- Create Main UI Frame ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # --- Title and Instructions ---
        self.title_label = ctk.CTkLabel(self.main_frame, text="Fake News Detection", 
                                        font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, pady=(20, 10), sticky="n")

        self.instruction_label = ctk.CTkLabel(self.main_frame, text="Paste the news headline and body text below for analysis:", 
                                            font=ctk.CTkFont(size=14))
        self.instruction_label.grid(row=1, column=0, pady=(0, 10), sticky="n")

        # --- Text Input Area ---
        self.text_input = ctk.CTkTextbox(self.main_frame, width=750, height=350, wrap="word",
        font=ctk.CTkFont(size=16))
        self.text_input.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # Insert placeholder and bind focus events
        self.text_input.insert("0.0", self.PLACEHOLDER_TEXT)
        self.text_input.bind("<FocusIn>", self._remove_placeholder)
        self.text_input.bind("<FocusOut>", self._add_placeholder)


        # --- Create Button Frame to hold both buttons ---
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.grid(row=3, column=0, padx=20, pady=20, sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1), weight=1) # Makes columns equally sized

        # --- Predict Button ---
        self.predict_button = ctk.CTkButton(self.button_frame, text="ANALYZE NEWS", 
                                            command=self.predict_click,
                                            font=ctk.CTkFont(size=18, weight="bold"),
                                            height=50)
        self.predict_button.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        # --- Clear Button ---
        self.clear_button = ctk.CTkButton(self.button_frame, text="CLEAR INPUT", command=self.clear_input,font=ctk.CTkFont(size=18),height=50,fg_color="gray50")
        self.clear_button.grid(row=0, column=1, padx=(10, 0), sticky="ew")


        # --- Result Display Label ---
        self.result_label = ctk.CTkLabel(self.main_frame, text="Awaiting input...", font=ctk.CTkFont(size=28, weight="bold"),text_color="gray")
        self.result_label.grid(row=4, column=0, pady=(10, 20), sticky="n")

    
    def _remove_placeholder(self, event=None):
        """Clears the placeholder text when the textbox receives focus."""
        current_content = self.text_input.get("1.0", "end-1c").strip()
        if current_content == self.PLACEHOLDER_TEXT:
            self.text_input.delete("1.0", "end")

    def _add_placeholder(self, event=None):
        """Adds the placeholder text back if the textbox is empty when losing focus."""
        current_content = self.text_input.get("1.0", "end-1c").strip()
        if not current_content:
            self.text_input.insert("1.0", self.PLACEHOLDER_TEXT)


    def load_model(self):
        """Loads the trained DistilBERT model and tokenizer."""
        global tokenizer, model
        
        if not os.path.exists(MODEL_PATH):
            self.title("Fake News Detection - ERROR")
            # If path is wrong, display error in console and close
            print(f"❌ FATAL ERROR: Model folder not found at '{MODEL_PATH}'")
            print("Please ensure the 'model' directory is in the same folder as app_gui.py.")
            sys.exit(1)

        try:
            # We assume DistilBERT was used, matching your fast training session
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            
            print("⏳ Loading trained model components...")
            tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
            model.to(device)
            model.eval()
            print(f"✅ Model successfully loaded on device: {device}")
            
        except Exception as e:
            self.title("Fake News Detection - MODEL LOAD FAILED")
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def clear_input(self):
        """Clears the text input area and resets the result label."""
        self.text_input.delete("1.0", "end")
        self.text_input.insert("0.0", self.PLACEHOLDER_TEXT)
        self.result_label.configure(text="Awaiting input...", text_color="gray")

    def predict_click(self):
        """Handles the button click event and runs the prediction."""
        
        input_text = self.text_input.get("1.0", "end-1c").strip()
        
        # Check if input is empty or just contains the placeholder text
        if not input_text or input_text == self.PLACEHOLDER_TEXT:
            self.result_label.configure(text="Please paste news text for analysis.", text_color="orange")
            return

        self.result_label.configure(text="Analyzing...", text_color="yellow")
        self.update_idletasks() # Force UI update immediately

        try:
            prediction = self._run_prediction(input_text)
            
            if prediction == 0:
                result_text = "FAKE NEWS ❌"
                color = "#E74C3C"  # Red
            else:
                result_text = "REAL NEWS ✅"
                color = "#2ECC71"  # Green
            
            self.result_label.configure(text=result_text, text_color=color)

        except Exception as e:
            self.result_label.configure(text="Prediction Error!", text_color="red")
            print(f"An unexpected error occurred during prediction: {e}")
            
            
    def _run_prediction(self, text):
        """Executes the model inference."""
        global tokenizer, model
        
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Move tensors to the selected device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        return predicted_class

if __name__ == "__main__":
    app = NewsSnifferApp()
    app.mainloop()
