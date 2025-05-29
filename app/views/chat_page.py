import os
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from utils import get_by_session_id
from faiss_db import search_documents  # Função de busca no FAISS

# Carrega variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHAT = os.getenv("MODEL_CHAT")

# ====================
# Funções auxiliares
# ====================

def clear_session_id():
    """Limpa o ID da sessão e reinicia o histórico"""
    session_id = st.session_state.get("session_id_chat")
    if session_id:
        get_by_session_id(session_id).clear()
    st.session_state.session_id_chat = None

def build_prompt_template():
    """Retorna o template de prompt para o modelo com mensagens e contexto"""
    return ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente jurídico especializado em direitos fundamentais. Seu papel é responder perguntas com base **exclusivamente** nos Artigos 1º a 4º da Constituição Federal do Brasil de 1988, com precisão e clareza jurídica.

REGRAS IMPORTANTES:
1. Todas as respostas devem incluir **um trecho literal da Constituição entre aspas** para fundamentar a informação.
2. Ao final da frase em que o trecho for utilizado, **cite a fonte no formato**: Constituição Federal, Art. X e sempre cite a fonte usando o Nome do arquivo citado entre [] ao final da frase relevante.
3. **Não invente ou extrapole** informações.
4. Não faça interpretações subjetivas. Responda apenas com base no texto constitucional.
5. Use linguagem clara, objetiva e fiel ao texto legal.

Contexto disponível:
{context}
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

def load_llm_chain():
    """Configura o modelo LLM com o prompt"""
    prompt_template = build_prompt_template()
    chat_model = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0.5,
        model=MODEL_CHAT,
        streaming=True
    )
    return prompt_template | chat_model

def build_context_from_docs(docs):
    """Formata o contexto a partir dos documentos retornados"""
    if not docs:
        return "Nenhum contexto encontrado."

    context_parts = []
    for i, (doc, score) in enumerate(docs):
        source = doc.metadata.get("source", "Fonte desconhecida")
        context_parts.append(f"**Fonte {i + 1} ({source})**: {doc.page_content}")
    return "\n\n".join(context_parts)

# ====================
# Interface do Streamlit
# ====================

def show():
    st.title("Interface de Chat com RAG")
    st.write("Área para interação via chat utilizando RAG para buscar informações na base FAISS.")

    with st.sidebar:
        st.header("Opções de Chat")
        st.button("Limpar Sessão", on_click=clear_session_id)

    # Cria sessão se não existir
    if not st.session_state.get("session_id_chat"):
        st.session_state.session_id_chat = str(uuid4())

    session_id = st.session_state.session_id_chat
    chat_history_store = get_by_session_id(session_id)
    rag_chain = load_llm_chain()

    # Exibir histórico anterior
    for msg in chat_history_store.messages:
        st.chat_message(msg.type).markdown(msg.content)

    # Input do usuário
    if user_input := st.chat_input("Digite sua mensagem"):
        user_message = HumanMessage(content=user_input)
        chat_history_store.add_messages([user_message])
        st.chat_message("human").markdown(user_input)

        try:
            docs = search_documents(user_input, k=10)
            context = build_context_from_docs(docs)
        except Exception as e:
            st.error(f"Erro na busca de contexto: {str(e)}")
            context = "Nenhum contexto encontrado."

        chat_history = chat_history_store.messages[:-1]

        # Streaming da resposta
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in rag_chain.stream({
                    "question": user_input,
                    "history": chat_history,
                    "context": context
                }):
                    if content := getattr(chunk, "content", ""):
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                chat_history_store.add_messages([AIMessage(content=full_response)])

            except Exception as e:
                st.error(f"Erro na geração da resposta: {str(e)}")
                chat_history_store.add_messages([AIMessage(content="Desculpe, ocorreu um erro interno.")])