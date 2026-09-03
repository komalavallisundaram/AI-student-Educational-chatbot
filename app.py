import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

data_file = r"C:/Users/shenbagam/videos/apanaa/QA_final_cleaned.csv"
embeddings_file = r"C:/Users/shenbagam/videos/apanaa/question_embeddings.csv"
df = pd.read_csv(data_file)

embeddings_df = pd.read_csv(embeddings_file)
questions = embeddings_df['Question'].tolist()
embeddings = embeddings_df.drop(columns=['Question']).to_numpy()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2', use_auth_token=hf_token)
else:
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

def answer_question(user_question, threshold=0.5):
    query_embedding = model.encode([user_question], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_idx = np.argmax(similarities)

    if similarities[top_idx] < threshold:
        return None

    return {
        "Question": df.iloc[top_idx]['Question'],
        "Answer": df.iloc[top_idx]['Answer'],
        "Similarity": similarities[top_idx]
    }


st.set_page_config(page_title="AI-Powered Educational Chatbot", layout="centered")

st.title("🤖 AI-Powered Educational Chatbot")
st.write("Ask me a question based on the knowledge base!")
st.sidebar.title("📝 Chat History")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.sidebar.write(f"👤 {msg['content']}")

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

if user_q := st.chat_input("Type your question here..."):
    st.session_state["messages"].append({"role": "user", "content": user_q})
    st.chat_message("user").write(user_q)

    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            time.sleep(1)  # simulate typing delay
            result = answer_question(user_q, threshold=0.5)
            if result:
                answer_text = f"**Answer:** {result['Answer']}\n\n"
                answer_text += f"_Matched Question:_ {result['Question']} (Similarity: {result['Similarity']:.4f})"
                st.session_state["messages"].append({"role": "assistant", "content": answer_text})
                st.write(answer_text)
            else:
                fallback = "⚠️ Sorry, I do not have information related to this question."
                st.session_state["messages"].append({"role": "assistant", "content": fallback})
                st.warning(fallback)

st.markdown("### 💡 Suggested Questions")
suggestions = [
    "What is Artificial Intelligence?",
    "Explain Data Science.",
    "What is Machine Learning?",
    "What is Python?"
]
cols = st.columns(len(suggestions))
for i, q in enumerate(suggestions):
    if cols[i].button(q):
        st.session_state["messages"].append({"role": "user", "content": q})
        st.chat_message("user").write(q)

        with st.chat_message("assistant"):
            with st.spinner("💭 Thinking..."):
                time.sleep(1)
                result = answer_question(q, threshold=0.5)
                if result:
                    answer_text = f"**Answer:** {result['Answer']}\n\n"
                    answer_text += f"_Matched Question:_ {result['Question']} (Similarity: {result['Similarity']:.4f})"
                    st.session_state["messages"].append({"role": "assistant", "content": answer_text})
                    st.write(answer_text)
                else:
                    fallback = "⚠️ Sorry, I do not have information related to this question."
                    st.session_state["messages"].append({"role": "assistant", "content": fallback})
                    st.warning(fallback)

st.markdown("---")
st.caption("🤖 Powered by Sentence Transformers + Streamlit | Built by Komalavalli Sundaram")
