import streamlit as st
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from utils import get_by_session_id
from faiss_db import search_documents  # Importação adicionada
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHAT = os.getenv("MODEL_CHAT")


def clear_session_id():
    """Limpa o ID da sessão e reinicia o histórico"""
    st.session_state.session_id_chat = None
    # Limpa o histórico associado à sessão anterior
    if "session_id_chat" in st.session_state:
        get_by_session_id(st.session_state.session_id_chat).clear()


def load_llm():
    """Configura o pipeline de LLM com suporte a RAG"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente jurídico especializado em direitos fundamentais. Seu papel é responder perguntas com base **exclusivamente** nos Artigos 1º a 4º da Constituição Federal do Brasil de 1988, com precisão e clareza jurídica.

            REGRAS IMPORTANTES:
            1. Todas as respostas devem incluir **um trecho literal da Constituição entre aspas** para fundamentar a informação.
            2. Ao final da frase em que o trecho for utilizado, **cite a fonte no formato**: Constituição Federal, Art. X e sempre cite a fonte usando [NOME_DO_ARQUIVO] ao final da frase relevante.
            3. **Não invente ou extrapole** informações. Se o contexto não permitir uma resposta com base **clara e direta** nos Artigos 1º a 4º, diga: "Não há informações suficientes nos Artigos 1º a 4º da Constituição Federal para responder a essa pergunta."
            4. Não faça interpretações subjetivas. Responda apenas com base no texto constitucional.
            5. Use linguagem clara, objetiva e fiel ao texto legal.

            Contexto disponível:
            {context}""",
         ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    return prompt | ChatOpenAI(
        api_key=OPENAI_API_KEY,
        # Usando secrets do Streamlit
        temperature=0.5,
        model=MODEL_CHAT,
        streaming=True
    )


def show():
    st.title("Interface de Chat com RAG")
    st.write("Área para interação via chat utilizando RAG para buscar informações na base FAISS.")

    with st.sidebar:
        st.header("Opções de Chat")
        st.button("Limpar Sessão", on_click=clear_session_id)

    # Gerenciamento de sessão seguro
    if not st.session_state.get("session_id_chat"):
        st.session_state.session_id_chat = str(uuid4())  # Convertendo para string

    chain = load_llm()
    history = get_by_session_id(st.session_state.session_id_chat)

    # Exibir histórico existente
    for msg in history.messages:
        st.chat_message(msg.type).markdown(msg.content)

    # Input do usuário
    if prompt := st.chat_input("Digite sua mensagem"):
        # Adicionar e exibir mensagem do usuário imediatamente
        human_message = HumanMessage(content=prompt)
        history.add_messages([human_message])
        st.chat_message("human").markdown(prompt)

        # Busca RAG no FAISS
        try:
            docs = search_documents(prompt, k=10)
            # context = "\n\n".join([f"Fonte {i + 1}: {d.page_content}" for i, d in enumerate(docs)])
            import pprint
            pprint.pprint(docs)

            context = ""

            # Formatar contexto com fontes
            for i, (doc, score) in enumerate(docs):
                source = doc.metadata.get('source', 'Fonte desconhecida')
                context += f"**Fonte {i + 1} ({source})**: {doc.page_content}\n\n"

        except Exception as e:
            st.error(f"Erro na busca de contexto: {str(e)}")
            context = "Nenhum contexto encontrado."

        # Preparar contexto histórico
        chat_history = history.messages[:-1]

        # Gerar resposta com streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Processar cada chunk do stream
            try:
                for chunk in chain.stream({
                    "question": prompt,
                    "history": chat_history,
                    "context": context
                }):
                    if content := getattr(chunk, 'content', ''):
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                history.add_messages([AIMessage(content=full_response)])

            except Exception as e:
                st.error(f"Erro na geração da resposta: {str(e)}")
                history.add_messages([AIMessage(content="Desculpe, ocorreu um erro interno.")])
