import os
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import re

# --- Config ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- Data: Example Q&A pairs ---
qa_pairs = [
    {"question": "apple", "answer": "A fruit that is usually red, green, or yellow."},
    {"question": "orange", "answer": "A citrus fruit that is orange in color."},
]

# --- Gemini Embedding ---
def get_gemini_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="semantic_similarity"
    )
    return np.array(response["embedding"], dtype=np.float32)

def gemini_retrieval(query, k=2):
    question_embeddings = np.array([get_gemini_embedding(pair["question"]) for pair in qa_pairs], dtype=np.float32)
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(question_embeddings)
    query_emb = get_gemini_embedding(query).reshape(1, -1)
    D, I = index.search(query_emb, k)
    return [qa_pairs[idx] for idx in I[0] if idx < len(qa_pairs)]

# --- Sentence Transformers Embedding ---
model = SentenceTransformer('all-MiniLM-L6-v2')
def get_st_embedding(text):
    emb = model.encode([text], convert_to_numpy=True)
    return emb[0].astype(np.float32)

def st_retrieval(query, k=2):
    question_embeddings = np.array([get_st_embedding(pair["question"]) for pair in qa_pairs], dtype=np.float32)
    faiss.normalize_L2(question_embeddings)
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(question_embeddings)
    query_emb = get_st_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k)
    return [qa_pairs[idx] for idx in I[0] if idx < len(qa_pairs)]

# --- Test Both Approaches ---
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

if __name__ == "__main__":
    # Gemini Embedding Similarity
    emb_apple_gemini = get_gemini_embedding("بطيخ")
    emb_orange_gemini = get_gemini_embedding("انس")
    sim_gemini = cosine_similarity(emb_apple_gemini, emb_orange_gemini)
    print("Gemini Embedding Cosine Similarity (بطيخ vs انس):", sim_gemini)

    # Sentence Transformers Similarity
    emb_apple_st = get_st_embedding("بطيخ")
    emb_orange_st = get_st_embedding("انس")
    sim_st = cosine_similarity(emb_apple_st, emb_orange_st)
    print("Sentence Transformers Cosine Similarity (بطيخ vs انس):", sim_st)
