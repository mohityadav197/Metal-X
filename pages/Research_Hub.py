from __future__ import annotations

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
        payload_files = []
        for uploaded in files:
            payload_files.append(
                (
                    "files",
                    (uploaded.name, uploaded.getvalue(), "application/pdf"),
                )
            )

        ok, response_data, err = post_form(
            "/research",
            data={"action": "index"},
            files=payload_files,
            timeout=5.0,
        )
        if ok:
            st.success("Research corpus indexed successfully.")
            st.json(response_data)
        else:
            if err == REQUEST_TIMEOUT_MESSAGE:
                st.warning("Knowledge Base taking longer than expected...")
            elif err == BACKEND_OFFLINE_MESSAGE:
                st.error(BACKEND_OFFLINE_MESSAGE)
            else:
                st.error(f"Indexing failed: {err}")
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
            response_block = answer
            if citations:
                response_block += "\n\nCitations:\n"
                for citation in citations:
                    response_block += f"- {citation}\n"
            st.session_state.research_chat.append({"role": "assistant", "content": response_block})
            st.rerun()
        else:
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
st.markdown("</div>", unsafe_allow_html=True)
