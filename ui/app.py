import streamlit as st
import requests
import uuid
API_URL = "http://127.0.0.1:8000/ask_llm"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
st.set_page_config(page_title="Notes Helper", layout="wide")

st.title(" Notes Helper")
st.write("Ask questions from your notes and get cited answers.")
subject = st.selectbox(
    "Select Subject",
    options=["os", "cn", "cloud"]
)
query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        resp = requests.post(API_URL, json={"query": query, "subject": subject, "session_id": st.session_state.session_id})
        data = resp.json()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Sources")
    for src in data["sources"]:
        page = f"page {src['page']}" if src["page"] else "page ?"
        st.markdown(f"- **{src['source']}**, {page}")
