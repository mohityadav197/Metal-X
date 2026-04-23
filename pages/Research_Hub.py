from __future__ import annotations

import os

import requests
import streamlit as st

from ui.platform import (
    BACKEND_OFFLINE_MESSAGE,
    REQUEST_TIMEOUT_MESSAGE,
    configure_page,
    inject_styles,
    post_form,
    post_json,
    render_sidebar,
)


configure_page("METALLURGIC-X | Research Hub")
inject_styles()
render_sidebar("The Knowledge Base")

if "research_chat" not in st.session_state:
    st.session_state.research_chat = []
if "research_last_answer" not in st.session_state:
    st.session_state.research_last_answer = ""
if "research_last_raw_chunks" not in st.session_state:
    st.session_state.research_last_raw_chunks = []

st.markdown("## The Knowledge Base")
st.caption("Upload PDFs and query indexed metallurgy literature in one streamlined workflow.")

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("Library Intake")
files = st.file_uploader(
    "Upload PDF Research Papers",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Index Uploaded PDFs"):
    if not files:
        st.warning("Add one or more PDF files before indexing.")
    else:
        uploaded_files = files

        # 1. Construct the strict list-of-tuples payload
        multipart_form_data = [
            ("files", (file.name, file.getvalue(), "application/pdf"))
            for file in uploaded_files
        ]

        # 2. Send to backend (ensure the URL matches your actual endpoint)
        with st.spinner(f"Sending {len(uploaded_files)} file(s) to backend..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/research/index",
                    files=multipart_form_data,
                    timeout=120
                )
                # 3. Handle Response
                if response.status_code == 200:
                    st.success("Indexing Complete!")
                    st.json(response.json())
                else:
                    st.error(f"Backend Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend on port 8000.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card' style='margin-top:0.8rem;'>", unsafe_allow_html=True)
st.subheader("Research Intelligence")
st.markdown("<p class='hero-subtext' style='margin-top:-0.35rem;'>Alloy Knowledge Base</p>", unsafe_allow_html=True)

history_html = ["<div class='chat-history full-width-chat'>"]
for msg in st.session_state.research_chat:
    css_class = "msg-user" if msg["role"] == "user" else "msg-agent"
    history_html.append(f"<div class='{css_class}'>{msg['content']}</div>")
history_html.append("</div>")
st.markdown("".join(history_html), unsafe_allow_html=True)

query = st.text_input("Ask the knowledge base", placeholder="What literature supports this Mg-Si ratio?")
if st.button("Query Research Endpoint"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        st.session_state.research_chat.append({"role": "user", "content": query.strip()})
        ok, data, err = post_json(
            "/research",
            {
                "query": query.strip(),
                "top_k": 5,
            },
            timeout=5.0,
        )
        if ok:
            answer = data.get("answer") or data.get("response") or "No response text returned."
            citations = data.get("citations") or data.get("sources") or []
            raw_chunks = data.get("raw_chunks") or []
            if not isinstance(raw_chunks, list):
                raw_chunks = []

            st.session_state.research_last_answer = answer
            st.session_state.research_last_raw_chunks = [str(chunk) for chunk in raw_chunks if str(chunk).strip()]

            response_block = answer
            if citations:
                response_block += "\n\nCitations:\n"
                for citation in citations:
                    response_block += f"- {citation}\n"
            st.session_state.research_chat.append({"role": "assistant", "content": response_block})
            st.rerun()
        else:
            st.session_state.research_last_answer = ""
            st.session_state.research_last_raw_chunks = []
            if err == REQUEST_TIMEOUT_MESSAGE:
                st.warning("Knowledge Base taking longer than expected...")
            elif err == BACKEND_OFFLINE_MESSAGE:
                st.session_state.research_chat.append(
                    {
                        "role": "assistant",
                        "content": BACKEND_OFFLINE_MESSAGE,
                    }
                )
            else:
                st.session_state.research_chat.append(
                    {
                        "role": "assistant",
                        "content": f"Request failed: {err}",
                    }
                )
            st.rerun()

latest_answer = st.session_state.get("research_last_answer", "")
if latest_answer:
    st.markdown("#### Latest Answer")
    st.text_area("Answer", value=latest_answer, height=140, disabled=True)

raw_chunks_for_view = st.session_state.get("research_last_raw_chunks", [])
with st.expander("View Retrieved Context"):
    if raw_chunks_for_view:
        for idx, chunk in enumerate(raw_chunks_for_view, start=1):
            st.markdown(f"**Chunk {idx}**")
            st.write(chunk)
            st.markdown("---")
    else:
        st.caption("No raw context chunks were returned for the latest query.")
st.markdown("</div>", unsafe_allow_html=True)
