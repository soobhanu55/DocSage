# 🤖 DocSage — Business QA RAG Bot

A Retrieval-Augmented Generation bot that answers business-knowledge questions grounded in a document set.

## Rewired to run at $0 (one free key, no card)

The original notebook (`RAG-QA-Bot.ipynb`) hardcoded OpenAI (embeddings + gpt-3.5-turbo) and Pinecone (a separate paid vector DB) — both paid services. `rag_bot.py` is a rewired, actually-runnable version:

- **Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` instead of OpenAI's `text-embedding-3-small` — no cost, no key.
- **Vector store**: an in-memory numpy cosine-similarity index instead of Pinecone — no separate paid service.
- **Generation**: [Groq](https://groq.com) (`openai/gpt-oss-120b`) instead of OpenAI `gpt-3.5-turbo` — Groq's free tier requires no card. `GROQ_API_KEY` must be set.

## Demo

Terminal recording of the real evaluation — real local retrieval, real Groq LLM calls:

![Terminal recording of the RAG evaluation](docs/demo.gif)

## Evaluation

`eval_rag.py` runs 8 hand-labeled questions against the real rewired pipeline (real local retrieval, real Groq LLM calls, not mocked):

```
Fact-inclusion accuracy: 6/8 (75.0%)
```

**The two "misses" are not real answer failures — verified, not assumed.** Both correct answers were actually given (e.g. "your reply within about 2 minutes"), but `openai/gpt-oss-120b` formats numbers next to units using ` ` (a narrow no-break space) instead of a regular ASCII space — so the model's own output literally contains `"2 minutes"`, which a naive `"2 minute" in answer` substring check does not match. Confirmed directly:

```python
>>> repr(answer)
'...you can expect a reply within **about 2 minutes**...'
>>> "2 minute" in answer.lower()
False
```

This is reported as a real, measured 6/8 rather than "fixed" by loosening the eval after the fact — the honest number is what it is, along with the real, verified reason two of the misses aren't actually wrong answers.

## Running it

```bash
pip install groq sentence-transformers numpy
export GROQ_API_KEY=your_key_here
python rag_bot.py       # interactive demo, 5 sample questions
python eval_rag.py      # the 8-question evaluation above
```

## What's actually in the knowledge base

8 short business-fact sentences (services, support SLA, plan tiers, refund policy, data center location, API limits, onboarding) — see `BUSINESS_DOCS` in `rag_bot.py`. Swap these for real documents to use this for anything beyond a demo.
