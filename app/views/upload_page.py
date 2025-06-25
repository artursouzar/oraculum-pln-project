import os
from dotenv import load_dotenv

import streamlit as st

from file_md import persist_document, list_documents, get_document, update_document
from utils import convert_file_to_md
from faiss_db import add_document_to_index
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """Você é um especialista em formatação de documentos técnicos. Reformate o texto seguindo estas regras:

1. **Estruturação lógica:**
   - Use headers hierárquicos (#, ##, ###)
   - Organize conteúdo relacionado em seções
   - Mantenha a ordem original das informações

2. **Formatação consistente:**
   - Dados numéricos: padrão local (ex: R$ 1.234,56 ou 12.345,67 unidades)
   - Listas: use marcadores ou numeração quando apropriado
   - Tabelas: para dados tabulares com mais de 3 itens
   - Ênfase: use **negrito** para termos técnicos e _itálico_ para termos estrangeiros

3. **Preservação de conteúdo:**
   - Nunca altere valores ou informações
   - Mantenha termos técnicos originais
   - Preserve referências a arquivos e metadados ([nome_arquivo])

4. **Melhoria de legibilidade:**
   - Adicione espaçamento lógico entre seções
   - Quebras de linha para parágrafos longos
   - Links clicáveis quando detectar URLs

Input: Texto Markdown cru extraído de documentos variados
Output: Versão formatada seguindo padrões técnicos"""

def format_with_ai(text: str) -> str:
    """
    Usa o LLM para melhorar a formatação de texto Markdown mantendo conteúdo original.
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", f"Texto original:\n{text}\n\nTexto reformatado:")
        ])

        chain = prompt | ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.2,
            model="gpt-4o"
        )
        response = chain.invoke({"text": text})
        return response.content

    except Exception as e:
        st.error(f"Erro na formatação via IA: {str(e)}")
        return text


def show():
    """
    Interface principal do Streamlit para upload, edição, persistência e visualização dos documentos.
    """
    st.title("Upload e Processamento de Arquivo")
    st.write("Área destinada ao upload de arquivos, extração e persistência em banco vetorial.")

    # Inicializa estados
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None

    # Sidebar para upload
    st.sidebar.header("Opções de Upload")
    uploaded_file = st.sidebar.file_uploader("Selecione um arquivo", type=["pdf", "docx", "txt", "md"])

    if uploaded_file and st.sidebar.button("Processar arquivo"):
        with st.spinner("Processando arquivo..."):
            try:
                md_content = convert_file_to_md(uploaded_file)
                persist_document(uploaded_file.name, md_content)
                st.session_state.last_uploaded = f"{uploaded_file.name.rsplit('.', 1)[0]}.md"
                st.success("Arquivo processado e salvo com sucesso!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {e}")

    st.subheader("Arquivos Processados")
    docs = list_documents()

    if docs:
        default_index = 0
        if st.session_state.last_uploaded in docs:
            default_index = docs.index(st.session_state.last_uploaded)

        selected_doc = st.selectbox("Selecione um arquivo para visualizar", docs, index=default_index)
        doc_text = get_document(selected_doc)  # Busca única para evitar repetição

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Persistir no FAISS", help="Armazena o documento no banco vetorial"):
                with st.spinner("Persistindo documento no FAISS..."):
                    try:
                        add_document_to_index(doc_text, selected_doc)
                        st.success("Documento persistido no FAISS com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao persistir documento no FAISS: {e}")

        with col2:
            if st.button("Editar" if not st.session_state.edit_mode else "Sair do modo edição"):
                st.session_state.edit_mode = not st.session_state.edit_mode
                st.experimental_rerun()

        if st.session_state.edit_mode:
            edited_content = st.text_area(
                "Editar Conteúdo",
                value=doc_text,
                height=400,
                help="Modifique o conteúdo Markdown ou use a formatação automática"
            )
            st.markdown("**Ações:**")
            if st.button("🪄 Auto-Formatar", help="Melhora a formatação mantendo o conteúdo original"):
                formatted = format_with_ai(edited_content)
                # Atualiza a área de texto com o conteúdo formatado
                st.session_state.last_edited = formatted
                st.experimental_rerun()

            if st.button("📤 Salvar", help="Persiste as alterações no documento"):
                update_document(selected_doc, edited_content)
                st.session_state.edit_mode = False
                st.success("Documento atualizado!")
                st.experimental_rerun()

        else:
            st.markdown(doc_text)

    else:
        st.info("Nenhum arquivo processado ainda.")
