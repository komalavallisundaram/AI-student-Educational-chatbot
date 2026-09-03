# AI-Powered Educational Chatbot
An interactive educational chatbot built with **Streamlit** and **Sentence Transformers**.  
It answers questions from a curated knowledge base (CSV dataset) using semantic similarity search.

##  Overview
This project provides a modern, mobile-friendly chatbot interface designed for educational purposes.  
The chatbot retrieves answers from a dataset using pre-computed embeddings and cosine similarity, ensuring accurate and context-aware responses.

## Features
- Modern chat interface with user and bot message bubbles  
- Typing indicator simulates real-time responses  
- Sidebar chat history for quick reference  
- Suggested questions for guided exploration  
- Fallback message: *“Sorry, I do not have information related to this question.”* when no match is found  
- Dataset-driven answers (no hallucinations, only CSV-based responses)  

##  Project Structure
├── app.py                   # Main Streamlit application (UI + chatbot interface)
├── content.py               # Functional logic for Sentence Transformer (embedding generation, saving, loading)
├── QA_final_cleaned.csv     # Dataset with questions and answers
├── question_embeddings.csv  # Pre-computed embeddings from Sentence Transformer
├── requirements.txt         # Dependencies
├── README.md                # Project overview
└── project_documentation.tex # Overall project documentation

## Installation
Clone the repository and install dependencies:

