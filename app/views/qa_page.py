import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
import re
from collections import OrderedDict

from file_md import list_documents, get_document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHAT = os.getenv("MODEL_CHAT")
MODEL_QA_GENERATOR = os.getenv("MODEL_QA_GENERATOR")

# Configurações de chunking
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 500
MAX_WORKERS = 4


def chunk_document(text):
    """Divide o documento em partes preservando o contexto"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def process_chunk(args):
    """Processa cada parte do documento em paralelo"""
    chunk, prompt_template, params = args
    try:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=params['temperature'],
            model=MODEL_QA_GENERATOR
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm

        return chain.invoke({
            "num_questions": params['questions_per_chunk'],
            "context_keywords": params['context_keywords'],
            "difficulty": params['difficulty'],
            "document_text": chunk
        }).content

    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        return ""


def generate_qa_full_document(doc_text, prompt_text, params):
    """Processa todo o documento em partes e consolida resultados"""
    chunks = chunk_document(doc_text)
    total_chunks = len(chunks)

    if total_chunks == 0:
        return ""

    # Calcula questões por chunk
    questions_per_chunk = max(1, params['num_questions'] // total_chunks)
    params['questions_per_chunk'] = questions_per_chunk

    st.info(f"Processando documento em {total_chunks} partes...")
    progress_bar = st.progress(0)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            futures.append(executor.submit(
                process_chunk,
                (chunk, prompt_text, params)
            ))
            progress_bar.progress((i + 1) / total_chunks)

        for future in futures:
            results.append(future.result())

    return clean_qa_content("\n\n".join(results))


def clean_qa_content(content):
    """Remove duplicatas e formata consistentemente"""
    qa_pattern = r"\*\*Pergunta \d+:\*\*.*?(?=\n\*\*Pergunta \d+:\*\*|\Z)"
    qa_pairs = re.findall(qa_pattern, content, re.DOTALL)

    unique_qa = OrderedDict()
    for pair in qa_pairs:
        clean_pair = re.sub(r'\s+', ' ', pair).strip()
        unique_qa[clean_pair] = None

    return "\n\n".join(unique_qa.keys())


def show_qa_generator():
    st.title("📝 Gerador de Perguntas e Respostas")

    # Inicialização de estados
    session_defaults = {
        'qa_content': None,
        'show_results': False,
        'selected_doc': None,
        'doc_text': "",
        'prev_selection': None
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    docs = list_documents()
    if not docs:
        st.warning("Nenhum documento disponível. Faça upload primeiro.")
        return

    # Função para atualizar documento selecionado
    def update_document():
        if st.session_state.selected_doc:
            st.session_state.doc_text = get_document(st.session_state.selected_doc)
        else:
            st.session_state.doc_text = ""

    # Seletor de documentos
    new_selection = st.selectbox(
        "Selecione o documento:",
        docs,
        index=0,
        key="selected_doc",
        on_change=update_document,
        help="Documento base para geração das QAs"
    )

    # Atualização imediata ao mudar seleção
    if new_selection != st.session_state.prev_selection:
        update_document()
        st.session_state.prev_selection = new_selection
        st.rerun()

    # Seção de estatísticas
    with st.expander("📊 Estatísticas do Documento", expanded=True):
        if st.session_state.doc_text:
            col1, col2 = st.columns(2)

            with col1:
                char_count = len(st.session_state.doc_text)
                word_count = len(st.session_state.doc_text.split())
                st.metric("Caracteres", f"{char_count:,}".replace(",", "."))
                st.metric("Palavras", f"{word_count:,}".replace(",", "."))

            with col2:
                unique_words = len(set(st.session_state.doc_text.lower().split()))
                avg_word_len = sum(
                    len(word) for word in st.session_state.doc_text.split()) / word_count if word_count > 0 else 0
                st.metric("Palavras Únicas", f"{unique_words:,}".replace(",", "."))
                st.metric("Média Letras/Palavra", f"{avg_word_len:.1f}")
        else:
            st.warning("Nenhum documento selecionado")

    # Pré-visualização do documento
    with st.expander("📄 Pré-visualização do Documento (Primeiros 500 caracteres)"):
        if st.session_state.doc_text:
            preview = st.session_state.doc_text[:500]
            st.markdown(f"```markdown\n{preview}\n```")
        else:
            st.warning("Nenhum conteúdo para pré-visualizar")

    # Formulário de geração
    with st.form("qa_form"):
        default_prompt = """Você é um especialista em criação de conteúdos educacionais. 
        Gere {num_questions} perguntas e respostas baseadas no documento abaixo, seguindo estas regras:

        1. Foco nos contextos: {context_keywords} (priorizar estes termos)
        2. Formato de resposta: **Pergunta X:** [texto] \\n\\n **Resposta X:** [texto]
        3. Nível de detalhe: adequado para profissionais de nível {difficulty}
        4. Inclua exemplos práticos quando relevante

        Documento:
        {document_text}"""

        prompt_text = st.text_area(
            "Instruções para geração:",
            value=default_prompt,
            height=200
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            num_questions = st.number_input("Número de QAs", 1, 200, 20)
        with col2:
            difficulty = st.selectbox(
                "Nível de Dificuldade",
                ["Iniciante", "Intermediário", "Avançado"]
            )
        with col3:
            temperature = st.slider("Criatividade (Temperatura)", 0.0, 1.0, 0.7, 0.1)

        context_keywords = st.text_input(
            "Contextos/Palavras-chave (separadas por vírgula)",
            help="Termos específicos que devem ser priorizados na geração"
        )

        submitted = st.form_submit_button("Gerar QAs")

        if submitted:
            if not st.session_state.doc_text:
                st.error("Selecione um documento válido")
                return

            if len(st.session_state.doc_text) < 100:
                st.warning("Documento muito curto para geração de QAs")
                return

            try:
                params = {
                    'num_questions': num_questions,
                    'context_keywords': context_keywords,
                    'difficulty': difficulty,
                    'temperature': temperature
                }

                full_response = generate_qa_full_document(
                    st.session_state.doc_text,
                    prompt_text,
                    params
                )

                st.session_state.qa_content = full_response
                st.session_state.show_results = True
                st.success("QAs gerados com sucesso!")

            except Exception as e:
                st.error(f"Erro na geração: {str(e)}")
                st.session_state.show_results = False

    # Seção de resultados
    if st.session_state.show_results and st.session_state.qa_content:
        st.markdown("---")
        display_qa_results(st.session_state.qa_content)

        st.download_button(
            label="📥 Baixar QAs",
            data=st.session_state.qa_content,
            file_name="qas_gerados.md",
            mime="text/markdown"
        )


def display_qa_results(content):
    st.markdown("### Resultado Completo")
    st.markdown("---")

    qa_pattern = r"\*\*Pergunta \d+:\*\*.*?(?=\n\*\*Pergunta \d+:\*\*|\Z)"
    qa_pairs = re.findall(qa_pattern, content, re.DOTALL)

    if not qa_pairs:
        st.warning("Nenhum par QA válido encontrado no formato esperado")
        st.text_area("Conteúdo Bruto", content, height=300)
        return

    with st.container():
        tab1, tab2 = st.tabs(["Visualização Formatada", "Conteúdo Bruto"])

        with tab1:
            for i, pair in enumerate(qa_pairs, 1):
                clean_pair = re.sub(r'\n{2,}', '\n\n', pair.strip())

                if "**Resposta" in clean_pair:
                    question, answer = clean_pair.split("**Resposta", 1)
                    question += "**"
                    answer = "**Resposta" + answer
                else:
                    question = clean_pair
                    answer = "Resposta não formatada corretamente"

                with st.expander(f"Pergunta {i}", expanded=True):
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                        <div style='color: #2b5876; font-weight: bold;'>❓ {question}</div>
                        <div style='margin-top: 0.5rem; color: #4a4a4a;'>💡 {answer}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            st.text(content)