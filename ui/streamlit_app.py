import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic RAG Platform",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic RAG Platform")

# -----------------------------------------------------------------------
# Sidebar — document upload + settings
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or HTML files",
        type=["pdf", "docx", "html", "htm"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Documents", disabled=not uploaded_files):
        with st.spinner("Ingesting documents..."):
            files = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]
            try:
                resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=180)
                resp.raise_for_status()
                for r in resp.json()["results"]:
                    name = r["source"].split("/")[-1].split("\\")[-1]
                    if r["status"] == "success":
                        st.success(
                            f"✅ {name}: {r['indexed_chunks']}/{r['total_chunks']} chunks indexed"
                        )
                    else:
                        st.error(f"❌ {name}: ingestion failed")
            except requests.HTTPError as e:
                st.error(f"Ingestion failed: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()

    st.subheader("⚙️ Settings")
    mode = st.selectbox("Retrieval Mode", ["hybrid", "dense", "sparse"], index=0)
    top_k = st.slider("Top-K candidates", min_value=3, max_value=20, value=10)
    session_id = st.text_input("Session ID (for memory)", value="default")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------
# Chat interface
# -----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.write(f"• {src}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/agent/run",
                    json={
                        "query": prompt,
                        "session_id": session_id,
                        "mode": mode,
                        "top_k": top_k,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for src in sources:
                            st.write(f"• {src}")
                st.caption(
                    f"⏱️ {data['duration_s']}s  |  "
                    f"{data['reranked_count']} chunks used  |  "
                    f"model: {data['model']}"
                )

            except requests.HTTPError as e:
                detail = e.response.json().get("detail", str(e))
                answer, sources = f"❌ {detail}", []
                st.error(answer)
            except Exception as e:
                answer, sources = f"❌ {e}", []
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
