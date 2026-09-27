import os
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.genai as genai   # ✅ updated import

# Configure Google Gemini API
genai.configure(api_key=os.getenv("AQ.Ab8RN6KE5erkkL1hUFIW_fcWyPKZ8cUefvu_gIOP1tirxq2RkQ"))
client = genai.Client()

# Load dataset and embeddings
qa_data = pd.read_csv("QA_final_cleaned.csv")
embeddings = np.loadtxt("question_embeddings.csv", delimiter=",")

# Load Sentence Transformer model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Streamlit UI
st.title("AI-Powered Educational Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat input
user_question = st.text_input("Ask a question:")

if user_question:
    # Step 1: Encode query
    query_embedding = model.encode([user_question], convert_to_numpy=True)

    # Step 2: Compare with stored embeddings
    similarities = cosine_similarity(query_embedding, embeddings)
    best_idx = np.argmax(similarities)
    best_score = similarities[0][best_idx]

    # Step 3: Decision logic
    if best_score >= 0.5:
        answer = qa_data.iloc[best_idx]["Answer"]
        response = f"Matched Question: {qa_data.iloc[best_idx]['Question']}\nSimilarity: {best_score:.2f}\nAnswer: {answer}"
    else:
        # Fallback → use Gemini for explanation
        gemini_response = client.models.generate_content(
            model="gemini-1.5-flash",   # ✅ updated model name
            contents=f"Explain this question in simple words: {user_question}"
        )
        response = gemini_response.text

    # Save to chat history
    st.session_state["messages"].append({"user": user_question, "bot": response})

# Display chat history
for msg in st.session_state["messages"]:
    st.write(f"**You:** {msg['user']}")
    st.write(f"**Bot:** {msg['bot']}")
