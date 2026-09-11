import streamlit as st
import pandas as pd
import google.generativeai as genai
import time

# --- Configure Gemini API key (must start with AIza...) ---
genai.configure(api_key="AQ.Ab8RN6LMr1M-52QfqOlGJZULKkTT66SqET-G0-TscaARaGjxYQ")

# Load dataset
data_file = r"C:/Users/shenbagam/videos/apanaa/QAfinalcleaned_expanded.csv"
df = pd.read_csv(data_file)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

def answer_from_gemini(q):
    try:
        response = model.generate_content(
            f"Answer this question clearly in a short explanatory paragraph: {q}"
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def get_answer_from_csv(q):
    match = df[df['Question'].str.lower() == q.lower()]
    if not match.empty:
        return match.iloc[0]['Answer']
    return None

def main():
    st.title("📘 Dual Q&A Streamlit App")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Split into two sections
    tab1, tab2 = st.tabs(["Dataset Q&A", "ChatGPT Q&A"])

    # --- Section 1: Dataset Q&A ---
    with tab1:
        st.subheader("Select a question from the dataset")
        question_list = df['Question'].tolist()
        selected_q = st.selectbox("Choose a question:", question_list)

        if st.button("Get Dataset Answer"):
            with st.spinner("Fetching answer..."):
                time.sleep(1)
                answer = get_answer_from_csv(selected_q)
                st.write(f"**Answer:** {answer}")
                st.session_state.history.append((selected_q, answer))

    # --- Section 2: ChatGPT/Gemini Q&A ---
    with tab2:
        st.subheader("Ask your own question")
        user_q = st.text_input("Type your question here:")

        if st.button("Get ChatGPT Answer"):
            if user_q.strip():
                with st.spinner("Generating answer..."):
                    time.sleep(1)
                    # First check CSV
                    answer = get_answer_from_csv(user_q)
                    if answer:
                        final_answer = answer
                    else:
                        final_answer = answer_from_gemini(user_q)

                st.write(f"**Answer:** {final_answer}")
                st.session_state.history.append((user_q, final_answer))
            else:
                st.warning("Please enter a question.")

    # --- Sidebar Chat History ---
    st.sidebar.title("📝 Chat History")
    if st.sidebar.button("Clear History"):
        st.session_state.history = []

    for i, (q, a) in enumerate(st.session_state.history, 1):
        st.sidebar.markdown(f"**Q{i}:** {q}")
        st.sidebar.markdown(f"*A{i}:* {a}")

if __name__ == "__main__":
    main()
