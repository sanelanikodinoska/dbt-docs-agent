"""Streamlit chat UI for the dbt docs agent. Run: streamlit run app.py"""
import streamlit as st
from agent import ask_dbt

st.set_page_config(page_title="dbt Docs Agent", page_icon="🔍")
st.title("🔍 dbt Docs Agent")
st.caption("Ask questions about dbt — answers grounded in the official docs.")

# chat history lives in session state so it survives reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# example questions in the sidebar
with st.sidebar:
    st.header("Try asking")
    examples = [
        "How do I schedule a dbt job?",
        "What is an incremental model?",
        "How can I automate things in dbt?",
        "Can I use Wizard to create a new model from a SQL query?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.pending = ex

# input: either a clicked example or typed text
prompt = st.chat_input("Ask about dbt...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the docs..."):
            answer = ask_dbt(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
