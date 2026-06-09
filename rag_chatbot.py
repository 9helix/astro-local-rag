from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import ollama, os, time

# 1. Load embedding model used during indexing
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Load FAISS DB
if os.path.exists("faiss_astronomy_index_corrected"):
    db = FAISS.load_local(
        "faiss_astronomy_index_corrected",
        embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    db = FAISS.load_local(
        "faiss_astronomy_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

# Store message history
messages = []
MAX_HISTORY = 5

def save_correction(query, corrected_answer):
    correction_doc = Document(
        page_content=f"When asked '{query}', the correct answer is: {corrected_answer}",
        metadata={
            "type": "correction",
            "original_query": query,
            "source": "user_correction"
        }
    )
    print(f"Correcting query '{query}' with answer '{corrected_answer}'")
    db.add_documents([correction_doc])
    db.save_local("faiss_astronomy_index_corrected")

def ask_astronomy_bot(question :str):
    global messages

    if question.startswith("Correction: "):
        query = next(
            (msg["content"] for msg in reversed(messages)
            if msg.get('role') == 'user'),
            None
        )
        if query:
            corrected_answer = question[len("Correction: "):]
            save_correction(query.strip(), corrected_answer.strip())
            
        response = "Thank you for the feedback! I've saved the correction."
        for word in response.split():
            yield word + " "
            time.sleep(0.02)
        return

    # Retrieve most relevant chunks
    docs = db.similarity_search(
        question, k=5
    )
    corrections = [d for d in docs if d.metadata.get("type") == "correction"]
    regular = [d for d in docs if d.metadata.get("type") != "correction"]

    # Debug print
    print("\n--- Retrieved Documents ---")
    for i, doc in enumerate(corrections + regular):
        print(f"\n[{i+1}] type: {doc.metadata.get('type', 'document')}")
        print(f"     source: {doc.metadata.get('source', 'unknown')}")
        print(f"     content: {doc.page_content[:200]}...")
    print("---------------------------\n")

    correction_context = "\n\n".join(
        doc.page_content
        for doc in corrections
    )
    regular_context = "\n\n".join(
        doc.page_content
        for doc in regular
    )
    context = f"""
    Corrections:
    {correction_context}

    Reference documents:
    {regular_context}
    """

    current_messages = [
        {
            "role": "system",
            "content": """
                You are an astronomy assistant.
                Answer using ONLY the provided context.
                
                If the context does not contain enough information to answer, say:
                "I don't know based on the provided context."

                The section named 'Corrections' has the highest priority.
                If information in Corrections conflicts with Reference documents,
                use the Corrections.

                Do not use outside knowledge. Do not invent facts. Keep answers factual.
            """
        },
    ]

     # Add previous conversation
    current_messages.extend(messages)
    # Add current RAG question
    current_messages.append(
        {
            "role": "user",
            "content": f"""
                Context:
                {context}

                Question:
                {question}
            """
        }
    )

    stream = ollama.chat(
        model = "llama3.2:3b",
        messages = current_messages,
        stream = True
    )

    full_response = ""
    for chunk in stream:
        content = chunk["message"]["content"]
        full_response += content
        yield content

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
    if len(messages) > MAX_HISTORY*2:
        messages = messages[-MAX_HISTORY*2:]
