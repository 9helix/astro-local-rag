from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Učitavanje dokumenata iz direktorija
dokumenti = []
docs_dir = Path("documents")

if docs_dir.exists() and docs_dir.is_dir():
    for file_path in docs_dir.iterdir():
        if file_path.suffix.lower() == ".pdf":
            dokumenti.extend(PyPDFLoader(str(file_path)).load())
        elif file_path.suffix.lower() == ".txt":
            dokumenti.extend(TextLoader(str(file_path), encoding="utf-8").load())
else:
    print(f"Direktorij {docs_dir} ne postoji!")

# 2. Komadanje teksta (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents(dokumenti)

# 3. Inicijalizacija besplatnog lokalnog embedding modela
# (Za HR jezik npr. 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 4. Kreiranje FAISS vektorske baze i pohrana chunka
db = FAISS.from_documents(chunks, embedding_model)

# 5. Spremanje baze lokalno na disk
db.save_local("faiss_astronomy_index")

print("Vektorska baza uspješno kreirana lokalno!")
