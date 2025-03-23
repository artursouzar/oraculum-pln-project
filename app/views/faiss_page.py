import streamlit as st
import os
import shutil
from faiss_db import get_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")


def delete_from_faiss(filename: str):
    """Remove documentos de forma segura com verificação de integridade"""
    try:
        index = FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

        # Coleta documentos válidos para manter
        keep_docs = []
        for doc in index.docstore._dict.values():
            if isinstance(doc.metadata, dict) and doc.metadata.get('source') != filename:
                keep_docs.append({
                    "text": doc.page_content,
                    "meta": doc.metadata
                })

        # Recria o índice apenas se houver documentos restantes
        if keep_docs:
            new_index = FAISS.from_embeddings(
                text_embeddings=[
                    (doc["text"], get_embeddings().embed_query(doc["text"]))
                    for doc in keep_docs
                ],
                embedding=get_embeddings(),
                metadatas=[doc["meta"] for doc in keep_docs]
            )
            new_index.save_local(INDEX_DIR)
        else:
            # Remove completamente o índice se estiver vazio
            shutil.rmtree(INDEX_DIR)

        return True

    except Exception as e:
        st.error(f"Erro na exclusão: {str(e)}")
        return False


def show_faiss_manager():
    st.title("🛠️ Gerenciamento do Banco Vetorial FAISS")

    try:
        if not os.path.exists(INDEX_DIR):
            st.warning("Nenhum índice FAISS encontrado!")
            return

        index = FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

        # Extrai e agrupa documentos de forma segura
        sources = {}
        for doc_id, doc in index.docstore._dict.items():
            if isinstance(doc.metadata, dict):
                source = doc.metadata.get('source', 'Desconhecido')
                sources.setdefault(source, {
                    'count': 0,
                    'content_samples': []
                })
                sources[source]['count'] += 1
                if len(sources[source]['content_samples']) < 3:
                    sources[source]['content_samples'].append(doc.page_content[:100] + '...')

        if not sources:
            st.info("Nenhum documento persistido no FAISS")
            return

        st.subheader("📁 Documentos Persistidos")
        for source, info in sources.items():
            with st.expander(f"🗂️ {source} ({info['count']} registros)"):
                st.write("**Amostras de conteúdo:**")
                for sample in info['content_samples']:
                    st.write(f"- {sample}")

                if st.button(f"❌ Excluir todos de {source}", key=f"del_{source}"):
                    if delete_from_faiss(source):
                        st.success(f"Todos {info['count']} registros de {source} foram removidos!")
                        st.rerun()
                    else:
                        st.error("Erro na exclusão")

    except Exception as e:
        st.error(f"Erro ao carregar índice: {str(e)}")
