"""Rewired version of the notebook's RAG QA bot: the original hardcoded
OpenAI (embeddings + gpt-3.5-turbo) and Pinecone (a separate paid vector
DB). Rewired here to be genuinely runnable at $0 beyond one free-tier
LLM key:

- Embeddings: local sentence-transformers (all-MiniLM-L6-v2) instead of
  OpenAI's text-embedding-3-small -- no cost, no key.
- Vector store: in-memory numpy cosine index instead of Pinecone -- no
  separate paid service, no key.
- Generation: Groq (llama-3.3-70b-versatile) instead of OpenAI gpt-3.5-turbo
  -- Groq's free tier requires no card. GROQ_API_KEY must be set.

The chunking, retrieval, and prompt-construction logic is otherwise the
same shape as the original notebook.

Run: GROQ_API_KEY=... python rag_bot.py
"""

from __future__ import annotations

import os

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

BUSINESS_DOCS = [
    "Company A provides digital marketing services, including SEO, PPC, and analytics.",
    "Our customer support operates 24/7 with an average response time of 2 minutes.",
    "The enterprise plan includes unlimited users, advanced dashboards, and AI integrations.",
    "The starter plan is limited to 5 users and does not include AI integrations.",
    "Refunds are available within 14 days of purchase for annual plans only.",
    "Our data centers are located in Frankfurt and comply with GDPR requirements.",
    "API rate limits are 1000 requests per hour on the enterprise plan.",
    "Onboarding for enterprise customers includes a dedicated account manager.",
]


def chunk_text(text: str, max_words: int = 80) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)] or [text]


class LocalVectorIndex:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.chunks: list[str] = []
        self.vectors: np.ndarray | None = None

    def add_documents(self, docs: list[str]) -> None:
        for doc in docs:
            self.chunks.extend(chunk_text(doc))
        vecs = self.embedder.encode(self.chunks, normalize_embeddings=True)
        self.vectors = np.array(vecs)

    def query(self, text: str, k: int = 3) -> list[str]:
        qvec = self.embedder.encode([text], normalize_embeddings=True)[0]
        sims = self.vectors @ qvec
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.chunks[i] for i in top_idx]


def generate_answer(client: Groq, query: str, context: str) -> str:
    prompt = f"""Answer the question based on the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY to run this.")

    print("Loading local embedder...")
    embedder = SentenceTransformer(EMBED_MODEL)
    index = LocalVectorIndex(embedder)
    index.add_documents(BUSINESS_DOCS)

    client = Groq(api_key=api_key)

    questions = [
        "What services does Company A provide?",
        "How fast is customer support response time?",
        "Does the starter plan include AI integrations?",
        "Where are the data centers located?",
        "What's the refund policy?",
    ]

    for q in questions:
        context = "\n---\n".join(index.query(q, k=3))
        answer = generate_answer(client, q, context)
        print(f"\nQ: {q}")
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
