import ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Load embedding model used during indexing
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Load FAISS DB
db = FAISS.load_local(
    "faiss_astronomy_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Store message history
messages = []
MAX_HISTORY = 10

def ask_astronomy_bot(question):
    global messages

    # Retrieve most relevant chunks
    docs = db.similarity_search(
        question, k=4
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
Use ONLY provided astronomy context.
If information is not available in the context, say you do not know based on provided context.

Context:
{context}

Question:
{question}

Answer:
"""
    current_messages = [
        {
            "role": "system",
            "content": "You are chatbot specialized for astronomy. Answer only using retrieved context. Keep answers factual and concise."
        }
    ]
     # Add previous conversation
    current_messages.extend(messages)
    # Add current RAG question
    current_messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    stream = ollama.chat(
        model = "llama3.2:3b",
        messages = current_messages,
        stream = True
    )

    print("\nAstronomyBot:")
    full_response = ""
    for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        full_response += content
    print()

    # Store original user question
    messages.append(
        {
            "role": "user",
            "content": question
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )

    # Prevent unlimited growth
    if len(messages) > MAX_HISTORY * 2:
        messages = messages[-MAX_HISTORY*2:]


while True:
    question = input("\nYour question:\n")
    if question.lower() == "exit":
        break
    ask_astronomy_bot(question)