import streamlit as st
from rag_chatbot import ask_astronomy_bot

st.set_page_config(
    page_title="AstronomyBot",
)
st.title("AstronomyBot")
st.caption("RAG-powered astronomy chatbot")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask AstronomyBot a question...")

if question:
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response = st.write_stream(
            ask_astronomy_bot(question)
        )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response
        }
    )