# Spam SMS Detection System

A machine learning project that classifies SMS messages as **spam** or **ham**. Includes data preprocessing and feature engineering in a Jupyter notebook, plus a Streamlit UI for live predictions.

---

## 🔧 Features
- Data cleaning: removes duplicates and unnecessary columns  
- Text preprocessing: tokenization, stopword removal, stemming  
- Feature engineering: message length, word count, sentence count  
- TF-IDF vectorization and Naive Bayes classifier  
- Model evaluation: accuracy, confusion matrix, classification report  
- Interactive Streamlit app (`app.py`) for user input  

---

## 📁 Project Structure
```text
spam-sms-detector/
├── data/  
│   └── spam.csv             # SMS Spam Collection dataset  
├── spam_sms.ipynb           # Notebook: preprocessing, EDA, model training  
├── app.py                   # Streamlit UI script  
├── model.pkl                # Pickled trained classifier  
├── vectorizer.pkl           # Pickled TF-IDF vectorizer  
└── README.md                # This file  
