#  FraudGuard AI Chatbot

An interactive Gen-AI powered fraud detection assistant that predicts and explains risky financial transactions. Built with **Streamlit**, **KNN**, and **Hugging Face Transformers**.

---

##  Project Overview

FraudGuard is designed for **three user roles**:
- 👤 Customers: Get fraud risk explanations on their transactions
- 🕵️ Fraud Analysts: Investigate suspicious activities interactively
- 👔 Bank Managers: Understand risk trends and summaries

---

##  Tech Stack

| Component        | Tech Used                    |
|------------------|------------------------------|
| ML Model         | KNeighborsClassifier (KNN)   |
| Feature Engineering | Custom rule-based + derived |
| LLM (GenAI)      | `google/flan-t5-base` (Hugging Face) |
| Interface        | Streamlit + Plotly           |
| Model Storage    | Pickle (`.pkl`)              |
| Deployment       | Local / Streamlit sharing    |

---

---

##  Running the App

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app/main.py


##  Running the App

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app/main.py

##  Folder Structure
fraudguard-ai-chatbot/
│
├── app/
│ └── main.py # Streamlit chatbot UI
│
├── ml_model/
│ ├── model_class.py # FraudDetectionModel definition
│ ├── model_predict.py # Prediction interface for chatbot
│ └── retrain_model.py # CLI retraining utility
│
├── tests/
│ └── test_model_predict.py # Unit tests for model
│
├── data/ # (optional) Sample input CSVs
├── assets/ # (optional) Images, icons
├── requirements.txt # All dependencies
├── .gitignore
└── README.md

Features:

Upload your own transaction CSV

Predict fraud with KNN model

Risk filtering, sorting, CSV download

Role-based question prompts

Natural Language QA using Hugging Face Flan-T5

Risk distribution charts via Plotly

Example Prompt (Fraud Analyst):  Show all transactions flagged as fraudulent

Future Improvements
LLM plug-in mode (OpenAI/GPT ready)
Better transaction timeline visualizations
Role-based authentication (planned)
Model explainability (SHAP values)




### LLM Modes

| Mode | How to Activate | Cost |
|------|-----------------|------|


| **Hugging Face (default)** | No action needed – free `microsoft/DialoGPT-small` model is downloaded on first run. | Free |


| **OpenAI GPT** | Set `USE_OPENAI=True` and add `OPENAI_API_KEY` to `.env`. | Pay‑as‑you‑go |
| **Rule‑based** | Set both `USE_HF_LLM=False` and `USE_OPENAI=False`. | Free |

> The project is LLM‑ready but incurs **no charges by default** because it runs an open‑source small model locally.


Contact:
Created by TANVI_VISHWANATH | For resume & learning purposes only.

