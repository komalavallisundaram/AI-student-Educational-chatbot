# AI-Powered Educational Chatbot
An interactive educational chatbot built with **Streamlit**, **Sentence Transformers**, and **Google Gemini API**.  
It answers questions from a curated knowledge base (CSV dataset) using semantic similarity search, with Gemini providing fallback explanations when no dataset match is found.

# Overview
This project provides a modern, mobile-friendly chatbot interface designed for educational purposes.  
The chatbot retrieves answers from a dataset using **pre-computed embeddings** and **cosine similarity**, ensuring accurate and context-aware responses.  
If no relevant answer is found, the chatbot uses **Google Gemini (`gemini-1.5-flash`)** to generate a simple explanation.

# Features
- Modern chat interface with user and bot message bubbles  
- Typing indicator simulates real-time responses  
- Sidebar chat history for quick reference  
- Dataset-driven answers (no hallucinations when matched)  
- Fallback to **Gemini AI** for unmatched queries  
- Secure API key configuration via environment variables  

# Project Structure
├── app.py                  # Main Streamlit application 
├── content.py              # Functional logic for Sentence Transformer
├── QA_final_cleaned.csv    # Dataset with questions and answers
├── question_embeddings.csv # Pre-computed embeddings from Sentence Transformer
├── requirements.txt        # Dependencies
├── README.md               # Project overview
└── project_documentation.tex # Overall project documentation

# InstallationClone the repository and install dependencies:
```bash
git clone https://github.com/komalavallisundaram/AI-student-Educational-chatbot.git
cd AI-student-Educational-chatbot
pip install -r requirements.txt
