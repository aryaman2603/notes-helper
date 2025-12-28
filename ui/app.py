import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/ask_llm"

st.set_page_config(page_title="Notes Helper", layout="wide")
st.title("Notes Helper")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "subject" not in st.session_state:
    st.session_state.subject = "os"


subject = st.selectbox(
    "Select subject",
    ["os", "cn", "cloud"],
    index=["os", "cn", "cloud"].index(st.session_state.subject)
)


if subject != st.session_state.subject:
    st.session_state.subject = subject
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask a question from your notes")

if query:
    
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(
                API_URL,
                json={
                    "query": query,
                    "subject": st.session_state.subject,
                    "session_id": st.session_state.session_id
                }
            )
            data = resp.json()
            answer = data["answer"]

        st.markdown(answer)

    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    
    with st.expander("Sources"):
        for src in data["sources"]:
            page = f"page {src['page']}" if src.get("page") else "page ?"
            st.markdown(f"- **{src['source']}**, {page}")
