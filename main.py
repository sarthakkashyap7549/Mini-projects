# import pandas as pd

# # Load both datasets
# fake = pd.read_csv("D:/Myprograms/fake_news_sniffer/Fake.csv")
# true = pd.read_csv("D:/Myprograms/fake_news_sniffer/True.csv")

# # Add labels
# fake["label"] = 0  # fake news
# true["label"] = 1  # true news

# # Combine them
# df = pd.concat([fake, true], ignore_index=True)

# # Drop missing text and merge title+text
# df = df.dropna(subset=["text"])
# df["content"] = df["title"] + " " + df["text"]
# df = df[["content", "label"]]

# print(df.head())
# df.to_csv("D:/Myprograms/fake_news_sniffer/cleaned_data.csv", index=False)
# print("✅ Cleaned dataset saved with both fake & true news!")

# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Load your cleaned dataset
# df = pd.read_csv("D:/Myprograms/fake_news_sniffer/cleaned_data.csv")

# # Split into training and testing data (80% train, 20% test)
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     df["content"].tolist(),
#     df["label"].tolist(),
#     test_size=0.2,
#     random_state=42
# )

# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Convert text into BERT tokens
# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# import torch

# class NewsDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = NewsDataset(train_encodings, train_labels)
# test_dataset = NewsDataset(test_encodings, test_labels)

# from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# from transformers import Trainer, TrainingArguments

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=1,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=50,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     eval_strategy="epoch"   # ✅ changed from evaluation_strategy
# )


# # Create Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

# # Train the model
# trainer.train()

# # Evaluate model
# print("📊 Evaluating model...")
# results = trainer.evaluate()
# print(results)

# # Save trained model and tokenizer

# model.save_pretrained("D:/Myprograms/fake_news_sniffer/model")
# tokenizer.save_pretrained("D:/Myprograms/fake_news_sniffer/model")
# print("✅ Model trained and saved successfully!")
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# --- 🎯 Configuration ---
# Set the model path (replace with your desired local path if different)
MODEL_SAVE_PATH = "D:/Myprograms/fake_news_sniffer/model"
DATA_BASE_PATH = "D:/Myprograms/fake_news_sniffer"
CLEANED_DATA_PATH = os.path.join(DATA_BASE_PATH, "cleaned_data.csv")

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# --- 🔄 Model Training / Loading Logic ---
if os.path.exists(MODEL_SAVE_PATH):
    # --- LOAD EXISTING MODEL (SKIP TRAINING) ---
    print(f"✅ Trained model found at: {MODEL_SAVE_PATH}. Loading for prediction...")
    try:
        # NOTE: If your successfully trained model was DISTILBERT, you must change 
        # BertTokenizer and BertForSequenceClassification to DistilBertTokenizer and 
        # DistilBertForSequenceClassification here.
        tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
        model.to(device)
    except Exception as e:
        print(f"❌ Error loading saved model: {e}")
        print("Please check if the saved model files are complete and compatible.")
        exit()

else:
    # --- TRAIN NEW MODEL (IF MODEL PATH DOES NOT EXIST) ---
    print("🚀 Trained model not found. Starting data preparation and training...")

    # Load and combine data
    fake = pd.read_csv(os.path.join(DATA_BASE_PATH, "Fake.csv"))
    true = pd.read_csv(os.path.join(DATA_BASE_PATH, "True.csv"))

    # Add labels and combine
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    
    # Drop missing text and merge title+text
    df = df.dropna(subset=["text", "title"])
    df["content"] = df["title"] + " " + df["text"]
    df = df[["content", "label"]]

    # Shuffle and save (optional but good practice)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print("✅ Cleaned dataset prepared and saved!")

    # Train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["content"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    # Dataset class (re-used from original code)
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = NewsDataset(train_encodings, train_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)

    # Define model and training arguments
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        report_to="none" # Ensure W&B logging is off
    )

    # Create and run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("🚀 Training model...")
    trainer.train()

    # Evaluate and Save
    print("📊 Evaluating model...")
    results = trainer.evaluate()
    print(results)

    # Save trained model and tokenizer
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("✅ Model trained and saved successfully!")

# --- 🔎 Prediction Function (Runs regardless of training/loading) ---

def predict_news(text):
    # Set model to evaluation mode
    model.eval()
    
    # Process input text
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

    # Interpretation: 0=Fake, 1=True
    if predicted_class == 0:
        print("📰 Prediction: FAKE NEWS ❌")
    else:
        print("🗞️ Prediction: REAL NEWS ✅")

# --- 🏃 Interactive Loop ---
print("\n--- Fake News Sniffer Ready for Prediction ---")
while True:
    user_input = input("\nEnter a news headline or paragraph (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    
    if not user_input.strip():
        print("Please enter some text.")
        continue

    predict_news(user_input)

print("\n--- Program finished ---")