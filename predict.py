# from transformers import BertTokenizer, BertForSequenceClassification
# import torch

# # Load the trained model and tokenizer
# model_path = "D:/Myprograms/fake_news_sniffer/model"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path)

# # Function to predict fake/real news
# def predict_news(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class = torch.argmax(logits, dim=1).item()

#     if predicted_class == 0:
#         print("📰 Prediction: FAKE NEWS ❌")
#     else:
#         print("🗞️ Prediction: REAL NEWS ✅")

# # Try a few examples
# while True:
#     user_input = input("\nEnter a news headline or paragraph (or 'exit' to quit): ")
#     if user_input.lower() == "exit":
#         break
#     predict_news(user_input)
import torch
# Use the DistilBERT classes, as this is the model you successfully trained/saved.
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# --- 🎯 SET THE CORRECT LOCAL MODEL PATH ---
# Assuming the trained model folder is named 'model' and is in the same directory as this script.
# If you placed it elsewhere, update the path accordingly, e.g., "D:/Myprograms/fake_news_sniffer/model"
model_path = "./model" 

# Check if the model path exists
if not os.path.exists(model_path):
    print(f"❌ Error: Model path not found at {model_path}")
    print("Please ensure you have downloaded and placed the 'model' folder in this directory.")
    exit()

# Load the trained model and tokenizer
print("⏳ Loading trained DistilBERT model...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    print(f"❌ Error loading model components: {e}")
    print("Ensure all files (config.json, pytorch_model.bin, vocab.txt, etc.) are in the 'model' folder.")
    exit()

# Use GPU if available (recommended for faster inference)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"🔥 Model successfully loaded and running on device: {device}")

# Function to predict fake/real news
def predict_news(text):
    # Set model to evaluation mode
    model.eval()
    
    # Process input text
    # Ensure truncation and max_length match training setup (max_length=128)
    inputs = tokenizer(
        text, 
        return_tensors="pt",  # Return PyTorch tensors
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    # Move tensors to the selected device (CPU or GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad(): # Disable gradient calculation for efficient inference
        outputs = model(**inputs)
    
    logits = outputs.logits
    # Get the class with the highest probability (0 or 1)
    predicted_class = torch.argmax(logits, dim=1).item()

    # Interpretation: 0=Fake, 1=True
    if predicted_class == 0:
        print("📰 Prediction: FAKE NEWS ❌")
    else:
        print("🗞️ Prediction: REAL NEWS ✅")

# Run the interactive prediction loop
print("\n--- Fake News Sniffer Ready ---")
while True:
    user_input = input("\nEnter a news headline or paragraph (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    
    if not user_input.strip():
        print("Please enter some text.")
        continue

    predict_news(user_input)

print("\n--- Program finished ---")