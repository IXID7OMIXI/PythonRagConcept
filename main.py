
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import super_secert_api as mysecrets
genai.configure(api_key=mysecrets.GOOGLE_API_KEY)
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import re

class RAGAgent:
    def __init__(self, embedding_dim: int = 384, model_name: str = 'all-MiniLM-L6-v2', k: int = 3, threshold: float = 0.5):
        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer(model_name)
        self.k = k
        self.threshold = threshold
        self.languages = ["en", "ar"]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks = []
        self._load_data()
        self._build_index()

    def _load_data(self):
        facts = self._load_facts()
        context = self._load_context()
        qa_pairs = self._load_qa_pairs()
        self.chunks = context + facts + [f"Q: {pair['question']} A: {pair['answer']}" for pair in qa_pairs]
        print(f"Total chunks: {len(self.chunks)}")
        for i, chunk in enumerate(self.chunks):
            print(f"Chunks[{i}] = {chunk}")

    def _load_facts(self):
        facts = []
        for lang in self.languages:
            path = os.path.join("Data", f"general{lang.upper()}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    facts.extend([line.strip() for line in f if line.strip()])
        return facts

    def _load_context(self):
        context = []
        for lang in self.languages:
            path = os.path.join("Data", f"specific{lang.upper()}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    context.extend([line.strip() for line in f if line.strip()])
        return context

    def _load_qa_pairs(self):
        qa_pairs = []
        qa_pattern = re.compile(r"Q:\s*(.*?)\s*A:\s*(.*)")
        for lang in self.languages:
            path = os.path.join("Data", f"qa{lang.upper()}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = qa_pattern.match(line.strip())
                        if match:
                            question, answer = match.groups()
                            qa_pairs.append({"question": question, "answer": answer})
        return qa_pairs

    def _build_index(self):
        if not self.chunks:
            return
        embeddings = np.array([self.model.encode([chunk], convert_to_numpy=True)[0].astype(np.float32) for chunk in self.chunks], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"FAISS index built with {len(self.chunks)} chunks.")

    def retrieve(self, query: str):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb.reshape(1, -1), self.k)
        top_chunks = []
        top_scores = []
        print(f"\n[FastAPI Query] Top {self.k} results for: '{query}'")
        for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
            print(f"#{rank}: {self.chunks[idx]} (similarity: {dist:.3f})")
            top_chunks.append(self.chunks[idx])
            top_scores.append(dist)
        return top_chunks, top_scores

    def answer(self, query: str):
        confirmations = ["yes", "thank you", "thanks", "clear", "تمام", "شكرا", "واضح"]
        if any(word in query.lower() for word in confirmations):
            prompt = f"The user said: '{query}'. Respond in the same language, confirming that everything is clear and offering further help in a friendly way."
            model = genai.GenerativeModel("gemini-2.0-flash-lite-001")
            response = model.generate_content(prompt)
            return response.text

        top_chunks, top_scores = self.retrieve(query)
        if not any(score >= self.threshold for score in top_scores):
            return "sorry please reformat your question | اعتذر , من فضلك اعد صياغة السؤال"

        context = "\n".join(top_chunks)
        prompt = f"You are an assistant for a company called bareeq. Context:\n{context}\n\nQuestion: {query}\nAnswer in the same language as the question, and ask if everything is clear be more friendly and don't always use the same sentences"
        print("\n[Gemini LLM Prompt]\n" + prompt)
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-lite-001")
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text
            else:
                print("Gemini LLM returned no text or an empty response.")
                return "Sorry, I couldn't generate an answer at this time."
        except Exception as e:
            print(f"Error from Gemini LLM: {e}")
            return "Sorry, there was an error generating the answer."


app = FastAPI()

class QueryRequest(BaseModel):
    query: str

agent = RAGAgent()

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    answer = agent.answer(request.query)
    return JSONResponse(content={"answer": answer})
