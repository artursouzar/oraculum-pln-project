import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os
import re
import time
from threading import Lock
from file_md import list_documents, get_document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_QA_GENERATOR = os.getenv("MODEL_QA_GENERATOR")

# Configurações
INITIAL_CHUNK_SIZE = 15000
MAX_WORKERS = 4
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60  # Aumentado para 45 segundos

def dynamic_chunk_size(text_length):
    if text_length > 200000:
        return 30000
    elif text_length > 100000:
        return 20000
    return INITIAL_CHUNK_SIZE

def chunk_document(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=dynamic_chunk_size(len(text)),
        chunk_overlap=int(dynamic_chunk_size(len(text)) * 0.1),
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )
    return splitter.split_text(text)


def process_chunk(args):
    chunk, prompt_template, params = args
    try:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=params['temperature'],
            model=MODEL_QA_GENERATOR,
            max_retries=2,
            timeout=REQUEST_TIMEOUT  # Usando novo timeout
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm

        questions_needed = params['questions_per_chunk']
        result = ""

        for attempt in range(MAX_RETRIES):
            try:
                response = chain.invoke({
                    "num_questions": questions_needed,
                    "context_keywords": params['context_keywords'],
                    "difficulty": params['difficulty'],
                    "document_text": chunk
                }).content

                generated = len(re.findall(r"\*\*Pergunta \d+:", response))
                result += response + "\n\n"

                if generated >= questions_needed:
                    break

                questions_needed -= generated

            except Exception as e:
                if "timed out" in str(e).lower() and attempt < MAX_RETRIES - 1:
                    time.sleep(2)  # Espera antes de retentar
                    continue
                else:
                    raise e

        return result, None

    except Exception as e:
        return None, str(e)


def generate_qa_streaming(doc_text, prompt_text, params):
    chunks = chunk_document(doc_text)
    total_chunks = len(chunks)

    if total_chunks == 0:
        return ""

    params['questions_per_chunk'] = max(2, params['num_questions'] // max(total_chunks, 1))

    st.session_state.start_time = time.time()
    st.session_state.qa_buffer = []
    st.session_state.lock = Lock()
    st.session_state.completed_chunks = 0
    st.session_state.total_qa_generated = 0
    st.session_state.show_stream = True  # Novo estado para controle de exibição

    # Container persistente para o stream
    stream_container = st.container()

    with stream_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk, (chunk, prompt_text, params)): i
                   for i, chunk in enumerate(chunks)}

        for future in as_completed(futures):
            chunk_index = futures[future]
            result, error = future.result()

            with st.session_state.lock:
                st.session_state.completed_chunks += 1
                progress = st.session_state.completed_chunks / total_chunks

                with stream_container:
                    progress_bar.progress(min(progress, 1.0))

                    if result:
                        qa_count = len(re.findall(r"\*\*Pergunta \d+:", result))
                        st.session_state.total_qa_generated += qa_count
                        st.session_state.qa_buffer.append(result)

                        with results_container:
                            display_qa_chunk(result)

                    elapsed = time.time() - st.session_state.start_time
                    status_text.markdown(f"""
                        **Progresso:** {st.session_state.completed_chunks}/{total_chunks} chunks  
                        **Tempo decorrido:** {elapsed:.1f}s  
                        **QAs coletados:** {st.session_state.total_qa_generated}
                    """)

                    if error:
                        st.error(f"Erro no chunk {chunk_index + 1}: {error}")

    full_content = clean_qa_content("\n\n".join(st.session_state.qa_buffer), params['num_questions'])

    final_count = len(re.findall(r"\*\*Pergunta \d+:", full_content))
    if final_count < params['num_questions']:
        additional_qas = generate_additional_qas(doc_text, params['num_questions'] - final_count, params)
        full_content += "\n\n" + additional_qas

    return clean_qa_content(full_content, params['num_questions'])


def generate_additional_qas(doc_text, num_needed, params):
    """Gera questões adicionais para completar o total solicitado"""
    try:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=params['temperature'],
            model=MODEL_QA_GENERATOR,
            timeout=REQUEST_TIMEOUT
        )
        prompt = ChatPromptTemplate.from_template("""
        Gere no mínimo de {num_questions} perguntas e respostas adicionais seguindo as mesmas regras.
        Documento: {document_text}
        """)

        chain = prompt | llm
        result = chain.invoke({
            "num_questions": num_needed,
            "document_text": doc_text[-10000:]
        })
        return result.content

    except Exception as e:
        st.error(f"Erro ao gerar complemento: {str(e)}")
        return ""


def clean_qa_content(content, num_questions):
    qa_pairs = []
    seen = set()

    # Regex melhorado para capturar pares completos
    pattern = r"(\*\*Pergunta \d+:\*\*.*?)(?=\n\*\*Pergunta \d+:\*\*|\Z)"

    for pair in re.findall(pattern, content, re.DOTALL):
        simplified = re.sub(r'\s+', ' ', pair).strip()
        if simplified not in seen:
            seen.add(simplified)
            qa_pairs.append(pair)
        if len(qa_pairs) >= num_questions:
            break

    return "\n\n".join(qa_pairs[:num_questions])


def display_qa_chunk(content):
    """Exibe um conjunto parcial de QAs"""
    qa_pattern = r"\*\*Pergunta \d+:\*\*.*?(?=\n\*\*Pergunta \d+:\*\*|\Z)"
    qa_pairs = re.findall(qa_pattern, content, re.DOTALL)

    for pair in qa_pairs:
        with st.expander(f"⚡ QA Gerada", expanded=False):
            st.markdown(pair)


def show_qa_generator():
    st.title("📝 Gerador de Perguntas e Respostas")

    session_defaults = {
        'qa_content': None,
        'show_results': False,
        'selected_doc': None,
        'doc_text': "",
        'processing': False
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    docs = list_documents()
    if not docs:
        st.warning("Nenhum documento disponível. Faça upload primeiro.")
        return

    new_selection = st.selectbox(
        "Selecione o documento:",
        docs,
        index=0,
        key="selected_doc",
        on_change=lambda: st.session_state.update({
            'doc_text': get_document(st.session_state.selected_doc),
            'show_results': False
        })
    )

    if st.session_state.selected_doc != new_selection:
        st.session_state.selected_doc = new_selection
        st.session_state.doc_text = get_document(new_selection)
        st.rerun()

    if st.session_state.doc_text:
        with st.expander("📊 Métricas do Documento", expanded=True):
            col1, col2 = st.columns(2)
            text_len = len(st.session_state.doc_text)
            words = st.session_state.doc_text.split()

            with col1:
                st.metric("Caracteres", f"{text_len:,d}".replace(",", "."))
                st.metric("Palavras", f"{len(words):,d}".replace(",", "."))

            with col2:
                st.metric("Palavras Únicas", f"{len(set(words)):,d}".replace(",", "."))
                st.metric("Tamanho Médio", f"{sum(len(w) for w in words) / len(words):.1f}")

    with st.form("qa_form"):
        default_prompt = """Você é um especialista em criação de conteúdos educacionais. 
        Gere no mínimo {num_questions} perguntas e respostas baseadas no documento abaixo:

        REGRAS:
        1. Foco nos contextos: {context_keywords} (priorizar estes termos)
        2. Formato obrigatório: 
        **Pergunta {{número}}:** [texto] \\n\\n **Resposta {{número}}:** [texto]
        3. Nível de detalhe: adequado para profissionais de nível {difficulty}
        4. Inclua exemplos quando relevante

        DOCUMENTO:
        {document_text}"""

        prompt_text = st.text_area(
            "Instruções para geração:",
            value=default_prompt,
            height=200
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            num_questions = st.number_input("Número de QAs", 1, 500, 50)
        with col2:
            difficulty = st.selectbox("Dificuldade", ["Iniciante", "Intermediário", "Avançado"])
        with col3:
            temperature = st.slider("Criatividade", 0.0, 1.0, 0.5)

        context_keywords = st.text_input("Palavras-chave (separadas por vírgula)")

        if st.form_submit_button("🚀 Gerar QAs") and st.session_state.doc_text:
            st.session_state.processing = True
            st.session_state.show_results = False

            params = {
                'num_questions': num_questions,
                'context_keywords': context_keywords,
                'difficulty': difficulty,
                'temperature': temperature
            }

            try:
                st.session_state.qa_content = generate_qa_streaming(
                    st.session_state.doc_text,
                    prompt_text,
                    params
                )
                st.session_state.show_results = True

                elapsed = time.time() - st.session_state.start_time
                st.toast(f"Processo concluído em {elapsed:.1f} segundos", icon="✅")

            except Exception as e:
                st.error(f"Erro crítico: {str(e)}")
            finally:
                st.session_state.processing = False

    if st.session_state.show_results and st.session_state.qa_content:
        st.markdown("---")
        st.subheader("Resultado Final")

        with st.expander("📋 Visualizar Todas as QAs", expanded=True):
            display_qa_results(st.session_state.qa_content)

        st.download_button(
            "💾 Baixar QAs",
            st.session_state.qa_content,
            file_name="qas_gerados.md",
            mime="text/markdown"
        )


def display_qa_results(content):
    # Regex melhorado para separar perguntas e respostas
    qa_pairs = re.findall(r"(\*\*Pergunta \d+:\*\*.*?)(?=\n\*\*Pergunta \d+:\*\*|\Z)", content, re.DOTALL)

    for i, pair in enumerate(qa_pairs, 1):
        with st.container():
            # Separa pergunta e resposta
            parts = re.split(r"\*\*Resposta \d+:\*\*", pair)
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()

                st.markdown(f"**Pergunta {i}**")
                st.markdown(question)
                st.markdown(f"**Resposta {i}**")
                st.markdown(answer)
            else:
                st.markdown(pair.replace("\\n\\n", "\n\n"))

            st.write("---")


if __name__ == "__main__":
    show_qa_generator()