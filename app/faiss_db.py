"""
faiss_db.py
-----------
Módulo responsável por gerenciar o índice FAISS para persistência e consulta vetorizada de documentos.
Utiliza as bibliotecas: faiss-cpu, langchain e openai.
"""

import os
from dotenv import load_dotenv
from langchain_openai import (
    OpenAIEmbeddings,
)
from langchain_community.vectorstores import FAISS
import streamlit as st  # Adição importante para acessar secrets

load_dotenv()

# Diretório onde o índice FAISS será salvo
INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "text-embedding-3-small")


def get_embeddings():
    """Factory method para criação segura dos embeddings com API Key"""
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=MODEL_EMBEDDING
    )


def init_faiss_index():
    """Inicializa o índice FAISS se não existir"""
    os.makedirs(INDEX_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        empty_index = FAISS.from_texts(
            texts=[""],
            embedding=get_embeddings(),
            metadatas=[{"source": "system_init"}]
        )
        empty_index.save_local(INDEX_DIR)
        st.toast("Índice FAISS inicializado com sucesso")


def add_document_to_index(document_text: str, filename: str):  # Adicione parâmetro filename
    """
    Adiciona documento com metadados ao índice FAISS
    """
    embeddings = get_embeddings()

    # Metadados para armazenar origem do documento
    metadatas = [{"source": filename}]

    if os.path.exists(INDEX_DIR):
        index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        index.add_texts(
            texts=[document_text],
            metadatas=metadatas  # Adicione metadados
        )
    else:
        index = FAISS.from_texts(
            texts=[document_text],
            embedding=embeddings,
            metadatas=metadatas  # Adicione metadados
        )

    index.save_local(INDEX_DIR)
    return index


def search_documents(query: str, k: int = 4):
    """
    Busca retornando documentos com metadados
    """
    try:
        if not os.path.exists(INDEX_DIR):
            return []

        embeddings = get_embeddings()
        index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return index.similarity_search_with_score(query, k=k)

    except Exception as e:
        st.error(f"Erro na busca FAISS: {str(e)}")
        return []


def list_faiss_documents():
    """Lista todos os documentos únicos no índice FAISS"""
    if not os.path.exists(INDEX_DIR):
        return []

    embeddings = get_embeddings()
    index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    return list({
        meta['source']
        for _, (_, meta) in index.docstore._dict.items()
        if 'source' in meta
    })
