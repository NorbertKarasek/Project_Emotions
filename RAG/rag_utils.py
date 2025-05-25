# rag_utils.py

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

KB_PATH = "knowledge.txt"
INDEX_PATH = "faiss_index"

def build_knowledge_base(kb_path: str = KB_PATH, index_path: str = INDEX_PATH):
    """Wczytuje plik tekstowy, dzieli na fragmenty, robi embedding i zapisuje FAISS."""
    # 1. Wczytaj dokument
    loader = TextLoader(kb_path, encoding="utf-8")
    docs = loader.load()

    # 2. Podziel na mniejsze kawałki (np. 500 znaków każdy)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3. Zrób embedding
    embedder = OllamaEmbeddings(model="SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ1_M")

    # 4. Zbuduj FAISS i zapisz na dysk
    vectordb = FAISS.from_documents(chunks, embedder)
    vectordb.save_local(index_path)
    print("📚 Knowledge base zbudowana i zapisana w", index_path)


def load_retriever(index_path: str = INDEX_PATH, k: int = 3):
    """Wczytuje FAISS i zwraca retrievera, który da Ci top-k dokumentów."""
    try:
        # Użyj pełnej ścieżki
        abs_path = os.path.abspath(index_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Katalog indeksu nie istnieje: {abs_path}")

        embedder = OllamaEmbeddings(model="SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ1_M")
        vectordb = FAISS.load_local(abs_path, embedder, allow_dangerous_deserialization=True)
        return vectordb.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        raise Exception(f"Błąd podczas wczytywania retrievera: {str(e)}")


if __name__ == "__main__":
    # Buduj bazę tylko raz
    if not os.path.exists(INDEX_PATH):
        build_knowledge_base()
    else:
        print("Index już istnieje, pomiń budowanie.")
