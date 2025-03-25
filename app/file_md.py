import os
from typing import List
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Diretório onde os arquivos MD serão salvos
MD_FOLDER = os.getenv("MD_FOLDER", "data/md")

def ensure_md_folder():
    if not os.path.exists(MD_FOLDER):
        os.makedirs(MD_FOLDER)

def persist_document(name, md_content):
    """
    Salva o conteúdo Markdown em um arquivo dentro do diretório MD_FOLDER.
    O nome do arquivo será o nome original sem extensão, acrescido de .md.
    """
    ensure_md_folder()
    base_name, _ = os.path.splitext(name)
    file_name = f"{base_name}.md"
    file_path = os.path.join(MD_FOLDER, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

def update_document(name, md_content):
    """
    Atualiza o conteúdo do arquivo Markdown existente no diretório MD_FOLDER.
    O nome do arquivo deve ser fornecido com a extensão .md.
    """
    ensure_md_folder()
    file_path = os.path.join(MD_FOLDER, name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

# def list_documents():
#     """
#     Retorna a lista de arquivos com extensão .md presentes no diretório MD_FOLDER.
#     """
#     ensure_md_folder()
#     return [f for f in os.listdir(MD_FOLDER) if f.endswith(".md")]
#
# def get_document(name):
#     """
#     Lê e retorna o conteúdo do arquivo Markdown com o nome informado.
#     """
#     file_path = os.path.join(MD_FOLDER, name)
#     if os.path.exists(file_path):
#         with open(file_path, "r", encoding="utf-8") as f:
#             return f.read()
#     return ""


def list_documents() -> List[str]:
    """Lista documentos processados no diretório MD"""
    md_dir = os.getenv("MD_FOLDER", "data/md")
    return [f for f in os.listdir(md_dir) if f.endswith(".md")]


def get_document(filename: str) -> str:
    """Obtém conteúdo de um documento específico"""
    md_dir = os.getenv("MD_FOLDER", "data/md")
    filepath = os.path.join(md_dir, filename)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Erro ao ler documento: {str(e)}")
        return ""