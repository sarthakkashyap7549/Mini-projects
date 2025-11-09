# 📰 Fake News Sniffer  
### AI-Based Fake News Detection using DistilBERT and CustomTkinter  

---

## 📖 Overview
The **Fake News Sniffer** is an AI-powered system that classifies news articles as *real* or *fake* using Natural Language Processing (NLP) and deep learning.  
It leverages **DistilBERT**, a lightweight Transformer model, to analyze text contextually and provides an easy-to-use **CustomTkinter GUI** for real-time predictions.

---

## ⚙️ Features
- Transformer-based text classification (DistilBERT)  
- High accuracy (~95%)  
- Fast and lightweight inference  
- Real-time detection via GUI  
- Confidence score displayed for each prediction  

---

## 🧠 Tech Stack
- **Language:** Python 3.9+  
- **Libraries:** PyTorch, Transformers, Pandas, NumPy, Scikit-learn  
- **GUI:** CustomTkinter  
- **Dataset:** Kaggle Fake and Real News Dataset  

---

## 🧩 Workflow
1. **Dataset Collection:** Fake and real news datasets are gathered.  
2. **Data Preprocessing:** Cleaning, removing null values, and merging title and text.  
3. **Tokenization:** Converting text into tokens using DistilBERT tokenizer.  
4. **Model Training:** Fine-tuning DistilBERT for binary classification.  
5. **Evaluation:** Testing model accuracy and F1 score.  
6. **Prediction:** Classifying new unseen news articles as real or fake.

---
