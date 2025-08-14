from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
import numpy as np
import faiss
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# --- Data Splitting/sorting ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
)

# List of languages
languages = ["en", "ar"]

#load and split general files
def load_general_chunks():
    chunks = []
    for lang in languages:
        path = os.path.join("Data", f"general{lang.upper()}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks.extend(text_splitter.split_text(text))
    return chunks

#load specific files (line by line)
def load_specific_chunks():
    chunks = []
    for lang in languages:
        path = os.path.join("Data", f"specific{lang.upper()}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                chunks.extend([line.strip() for line in f if line.strip()])
    return chunks

general_chunks = load_general_chunks()
specific_chunks = load_specific_chunks()
chunks = general_chunks + specific_chunks

print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunks[{i}] = {chunk}")
# --- Data Splitting/sorting ---

# --- FAISS vector database creation and persistence ---
FAISS_PATH = "faiss_db"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="semantic_similarity"
    )
    return np.array(response["embedding"], dtype=np.float32)

embeddings = np.array([get_gemini_embedding(chunk) for chunk in chunks], dtype=np.float32)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

os.makedirs(FAISS_PATH, exist_ok=True)
faiss.write_index(index, os.path.join(FAISS_PATH, "index.faiss"))
print(f"FAISS vector database created and persisted at: {os.path.join(FAISS_PATH, 'index.faiss')}")
# --- FAISS vector database creation and persistence ---

# --- fast API for endpoint ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    answer = do_query_with_context(request.query)
    return JSONResponse(content={"answer": answer})

def do_query_with_context(query, k=3):
    query_emb = get_gemini_embedding(query).reshape(1, -1)
    D, I = index.search(query_emb, k)
    context_chunks = []
    for idx, dist in zip(I[0], D[0]):
        if dist <= 0.5:
            context_chunks.append(chunks[idx])
    if not context_chunks:
        return "sorry please contact customer support"

    context = "\n".join(context_chunks)
    prompt = f"You are an assistant for a company called bareeq, Context:\n{context}\n\nQuestion: {query}\nAnswer in the same language as the question, and ask if everything is clear be more friendly and don't always use the same sentences"
    model = genai.GenerativeModel("gemini-2.0-flash-lite-001")
    response = model.generate_content(prompt)
    return response.text
# --- fast API for endpoint ---
