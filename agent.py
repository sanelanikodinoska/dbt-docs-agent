"""The agent: tool-use loop over the search backend. Imported by app.py."""
import json
import os
from dotenv import load_dotenv
import anthropic

from search import text_search

load_dotenv()
_client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

TOOLS = [{
    "name": "text_search",
    "description": ("Search the dbt documentation. Call whenever you need factual "
                    "info about dbt. You may call it MULTIPLE times with different "
                    "queries. Returns chunks with filenames."),
    "input_schema": {"type": "object",
                     "properties": {"query": {"type": "string"}},
                     "required": ["query"]},
}]
TOOL_FUNCTIONS = {"text_search": text_search}

SYSTEM_PROMPT = (
    "You are a dbt documentation assistant. Use text_search to ground every answer "
    "in the docs. Only state facts present in the search results. End every answer "
    "with the source filenames you used. If the docs don't cover it, say so."
)


def run_agent(question, max_turns=6):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        resp = _client.messages.create(
            model=MODEL, max_tokens=2048,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": resp.content})
        tr = []
        for b in resp.content:
            if b.type == "tool_use":
                tr.append({"type": "tool_result", "tool_use_id": b.id,
                           "content": json.dumps(TOOL_FUNCTIONS[b.name](**b.input))})
        messages.append({"role": "user", "content": tr})
    return "Stopped: reached max turns."


def ask_dbt(question):
    return run_agent(question)
