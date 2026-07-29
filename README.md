# dbt Docs Agent — Day 6 Deployment

A Streamlit chat app over the dbt documentation, powered by hybrid search + a
tool-using Claude agent.

## Structure
- `search.py` — lexical + vector + hybrid search (builds index once, lazily)
- `agent.py` — the tool-use agent loop; imports `search`
- `app.py` — Streamlit chat UI; imports `agent`
- `chunks_headerfirst.jsonl` — the chunked corpus (from the pipeline notebook)
- `embeddings_headerfirst_*.npy` — cached embeddings (auto-created on first run)

## Setup
```powershell
uv add streamlit minsearch sentence-transformers anthropic python-dotenv numpy
copy .env.example .env      # then paste your real ANTHROPIC_API_KEY into .env
```

## Run
```powershell
uv run streamlit run app.py
```
Opens at http://localhost:8501

## Notes
- First launch encodes embeddings if the .npy cache is absent (~20 min once).
  Copy your existing `embeddings_headerfirst_*.npy` into this folder to skip it.
- The corpus file `chunks_headerfirst.jsonl` must be in this folder (or adjust
  the path in `search.py`).
