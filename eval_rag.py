"""Hand-labeled evaluation of the rewired RAG bot (rag_bot.py): for each
question, checks whether the generated answer contains the expected key
fact substring. Real LLM calls via Groq (free tier), real local retrieval.

Run: GROQ_API_KEY=... python eval_rag.py
"""

from __future__ import annotations

import os

from groq import Groq
from sentence_transformers import SentenceTransformer

from rag_bot import BUSINESS_DOCS, GROQ_MODEL, LocalVectorIndex, generate_answer

EVAL_SET = [
    ("What services does Company A provide?", ["seo", "ppc", "analytics"]),
    ("How fast does customer support respond?", ["2 minute"]),
    ("Does the starter plan include AI integrations?", ["starter", "does not include"]),
    ("Where are the data centers located?", ["frankfurt"]),
    ("What's the refund policy?", ["14 days"]),
    ("How many users does the enterprise plan support?", ["unlimited"]),
    ("What's the API rate limit on the enterprise plan?", ["1000"]),
    ("Do enterprise customers get a dedicated account manager?", ["dedicated account manager"]),
]


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY to run this.")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = LocalVectorIndex(embedder)
    index.add_documents(BUSINESS_DOCS)
    client = Groq(api_key=api_key)

    hits = 0
    for question, expected_facts in EVAL_SET:
        context = "\n---\n".join(index.query(question, k=3))
        answer = generate_answer(client, question, context)
        answer_lower = answer.lower()
        hit = any(fact.lower() in answer_lower for fact in expected_facts)
        hits += int(hit)
        print(f"{'OK ' if hit else 'MISS'} | {question}")
        print(f"     -> {answer[:150]}")

    n = len(EVAL_SET)
    print(f"\nFact-inclusion accuracy: {hits}/{n} ({100*hits/n:.1f}%)")


if __name__ == "__main__":
    main()
