from __future__ import annotations

import streamlit as st

from ui.platform import configure_page, inject_styles, post_form, post_json, render_sidebar


configure_page("METALLURGIC-X | Research Hub")
inject_styles()
render_sidebar("The Knowledge Base")

if "research_chat" not in st.session_state:
    st.session_state.research_chat = []

st.markdown("## The Knowledge Base")
st.caption("Upload PDFs and chat with your research library in a modern, colorful message flow")

upload_col, chat_col = st.columns([1.1, 1.7], gap="large")

with upload_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("PDF Intake")
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
                timeout=180.0,
            )
            if ok:
                st.success("Research corpus indexed successfully.")
                st.json(response_data)
            else:
                st.error(f"Indexing failed: {err}")
    st.markdown("</div>", unsafe_allow_html=True)

with chat_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Shiny Research Chat")

    history_html = ["<div class='chat-history'>"]
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
                timeout=120.0,
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
                st.session_state.research_chat.append(
                    {
                        "role": "assistant",
                        "content": (
                            "The /research endpoint is not reachable or returned an error.\n"
                            f"Details: {err}"
                        ),
                    }
                )
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
