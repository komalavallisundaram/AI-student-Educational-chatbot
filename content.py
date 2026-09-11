import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ File paths
input_file = r"C:\Users\shenbagam\videos\apanaa\QAfinalcleaned_expanded.csv"
output_file = r"C:/Users/shenbagam/videos/apanaa/QA_cleaned.csv"
log_file = r"C:/Users/shenbagam/videos/apanaa/QA_cleaning_log.txt"
output_embeddings_csv = r"C:/Users/shenbagam/videos/apanaa/question_embeddings.csv"

# Step 1: Load dataset safely
try:
    df = pd.read_csv(input_file, quotechar='"', encoding="utf-8")
except pd.errors.ParserError:
    print("⚠️ ParserError detected. Retrying with python engine...")
    df = pd.read_csv(input_file, quotechar='"', encoding="utf-8", engine="python")

# Step 2: Clean Question/Answer columns
df['Question'] = df['Question'].fillna('').astype(str).str.strip()
df['Answer'] = df['Answer'].fillna('').astype(str).str.strip()

# Step 3: Drop empty rows
dropped_rows = df[(df['Question'] == "") | (df['Answer'] == "")]
df = df[(df['Question'] != "") & (df['Answer'] != "")]

# Step 4: Remove duplicates (ensures non-repeated dataset)
df = df.drop_duplicates(subset=['Question'])

# Step 5: Save cleaned dataset
df.to_csv(output_file, index=False, encoding="utf-8")
print("🎯 Final cleaned dataset saved as QA_cleaned.csv")
print("Total questions:", len(df))
print(df.head())

# Step 6: Log dropped rows (if any)
if not dropped_rows.empty:
    dropped_rows.to_csv(log_file, index=False)
    print(f"⚠️ {len(dropped_rows)} rows dropped. See {log_file} for details.")

# Step 7: Generate embeddings
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
questions = df['Question'].tolist()
embeddings = model.encode(questions, convert_to_numpy=True)

# Step 8: Save embeddings to CSV
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, "Question", questions)
embeddings_df.to_csv(output_embeddings_csv, index=False)
print("✅ Embeddings generated and saved as CSV:", output_embeddings_csv)

# Step 9: Function for answering queries
def answer_question(user_question, top_k=1):
    query_embedding = model.encode([user_question], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_idx = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({
            "Question": df.iloc[idx]['Question'],
            "Answer": df.iloc[idx]['Answer'],
            "Similarity": similarities[idx]
        })
    return results

# Step 10: Example usage
if __name__ == "__main__":
    user_q = "What is Machine Learning?"
    results = answer_question(user_q, top_k=1)
    print("\n🔍 User Question:", user_q)
    print("✅ Best Match:", results[0]['Question'])
    print("💡 Answer:", results[0]['Answer'])
    print("📊 Similarity Score:", results[0]['Similarity'])
