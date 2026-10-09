import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import google.genai as genai
import time

# 🔑 Configure Gemini API
client = genai.Client(api_key="API_KEY")  # Replace with your valid Gemini API key

# 📂 Load dataset and embeddings
master_csv = r"C:/Users/shenbagam/videos/apanaa/QA_cleaned_unique.csv"
embeddings_file = r"C:/Users/shenbagam/videos/apanaa/question_embeddings.csv"

df_master = pd.read_csv(master_csv)
df_embed = pd.read_csv(embeddings_file)

# 🧹 Clean column names
df_master.columns = df_master.columns.str.strip().str.lower().str.replace("?", "").str.replace(" ", "_")
df_embed.columns = df_embed.columns.str.strip()

# 🎯 Prepare embedding matrix
embedding_matrix = df_embed.drop(columns=["Question", "Answer"]).values

# 📌 Sidebar navigation
st.sidebar.title("📘 AI Q&A Chatbot")
page = st.sidebar.radio("Choose a mode:", ["Dataset Q&A", "LLM Q&A"])

# ==========================
# 📊 Dataset Q&A Page
# ==========================
if page == "Dataset Q&A":
    st.title("📊 Dataset Q&A (Interactive Chat Mode)")

    # ✅ Initialize history (only questions)
    if "dataset_history" not in st.session_state:
        st.session_state.dataset_history = []

    # ✅ Search box with suggestions
    search_query = st.text_input("🔍 Search or type a question:")
    suggestions = []
    selected_question = None

    if search_query.strip():
        suggestions = [q for q in df_embed['Question'].tolist() if search_query.lower() in q.lower()]
        if suggestions:
            selected_question = st.selectbox("📌 Matching questions:", suggestions)

            # 🔽 Show suggestion questions below dropdown as clickable buttons
            st.markdown("### 🔽 Suggested Questions")
            for s in suggestions:
                if st.button(s, key=f"suggestion_{s}"):
                    selected_question = s
        else:
            st.warning("❌ No matching questions found.")

    threshold = st.slider("📊 Set similarity threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    top_n = st.radio("📌 Show answers:", ["Best Answer", "Top 3 Answers"])

    if st.button("Get Answer") and selected_question:
        # ✅ Find the answer directly from df_master
        row = df_master[df_master['question'] == selected_question]
        if not row.empty:
            answer_text = row.iloc[0]['answer']

            # ✅ Save only the question to history
            st.session_state.dataset_history.append(selected_question)

            # Typing indicator
            with st.spinner("🤖 Assistant is typing..."):
                time.sleep(1)

            # Show the answer directly
            st.subheader("💡 Answer")
            with st.chat_message("assistant"):
                st.write(answer_text)
        else:
            # Fall back to similarity search if exact match not found
            idx = df_embed.index[df_embed['Question'] == selected_question][0]
            selected_embedding = embedding_matrix[idx].reshape(1, -1)
            similarities = cosine_similarity(selected_embedding, embedding_matrix)[0]

            results = []
            for i, sim in enumerate(similarities):
                if sim >= threshold:
                    results.append({
                        "Question": df_embed.iloc[i]['Question'],
                        "Answer": df_embed.iloc[i]['Answer'],
                        "Similarity": sim
                    })

            results = sorted(results, key=lambda x: x['Similarity'], reverse=True)

            if results:
                st.session_state.dataset_history.append(selected_question)
                with st.spinner("🤖 Assistant is typing..."):
                    time.sleep(1)

                st.subheader("💡 Answer(s)")
                if top_n == "Best Answer":
                    best_answer = results[0]
                    with st.chat_message("assistant"):
                        st.write(best_answer['Answer'])
                    st.caption(f"📊 Similarity Score: {best_answer['Similarity']:.2f}")
                else:
                    for r in results[:3]:
                        with st.expander(f"Q: {r['Question']} (Similarity: {r['Similarity']:.2f})"):
                            st.write(r['Answer'])
            else:
                st.warning("❌ No matches found above threshold.")

    # ✅ Display history (only questions)
    if st.session_state.dataset_history:
        st.subheader("📝 Questions Asked")
        for q in st.session_state.dataset_history:
            with st.chat_message("user"):
                st.write(q)

        # Download history
        st.download_button(
            "⬇️ Download History",
            "\n".join(st.session_state.dataset_history),
            file_name="dataset_question_history.txt"
        )

        if st.button("Clear History"):
            st.session_state.dataset_history = []
            st.success("🧹 History cleared!")

# ==========================
# 🤖 LLM Q&A Page
# ==========================
elif page == "LLM Q&A":
    st.title("🤖 LLM Q&A (Gemini 3.5 Flash)")

    # ✅ Initialize history (only questions)
    if "llm_history" not in st.session_state:
        st.session_state.llm_history = []

    user_q = st.text_input("💬 Type your question:")

    if st.button("Get Answer"):
        if user_q.strip():
            try:
                with st.spinner("🤖 Assistant is typing..."):
                    time.sleep(1)

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=f"Answer this question in two concise lines: {user_q}"
                )
                answer = response.text.strip()

                # ✅ Save only the question to history
                st.session_state.llm_history.append(user_q)

                # Show answer immediately
                with st.chat_message("assistant"):
                    st.write(answer)

            except Exception as e:
                st.error(f"⚠️ Gemini API error: {e}")
        else:
            st.warning("Please enter a question.")

    # ✅ Display history (only questions)
    if st.session_state.llm_history:
        st.subheader("📝 Questions Asked")
        for q in st.session_state.llm_history:
            with st.chat_message("user"):
                st.write(q)

        # Download history
        st.download_button(
            "⬇️ Download History",
            "\n".join(st.session_state.llm_history),
            file_name="llm_question_history.txt"
        )

        if st.button("Clear History"):
            st.session_state.llm_history = []
            st.success("🧹 History cleared!")
