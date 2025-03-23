# utils.py
import tempfile
from docling.document_converter import DocumentConverter
from typing import List
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """
    Implementação customizada de histórico de conversação em memória.
    Herda de BaseChatMessageHistory para integração com LangChain e de BaseModel para validação Pydantic.

    Atributos:
        messages (List[BaseMessage]): Lista de mensagens do chat armazenadas em memória.
            Usa Field com default_factory para garantir uma nova lista por instância.

    Métodos:
        add_messages: Adiciona múltiplas mensagens ao histórico
        clear: Limpa todo o histórico da conversa
    """
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """
        Adiciona uma lista de mensagens ao histórico.

        Parâmetros:
            messages (List[BaseMessage]): Lista de mensagens a serem adicionadas.
                Pode conter tanto HumanMessage quanto AIMessage.
        """
        self.messages.extend(messages)

    def clear(self) -> None:
        """Limpa completamente o histórico de mensagens da conversa."""
        self.messages = []


# Dicionário global para armazenar múltiplos históricos de conversação
# Chave: session_id (string) | Valor: Instância de InMemoryHistory
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    """
    Factory function para gerenciamento de históricos por sessão.
    Mantém diferentes conversas isoladas usando session_ids únicos.

    Parâmetros:
        session_id (str): Identificador único da sessão de conversa.
            Normalmente gerado pelo sistema ou fornecido via aplicação.

    Retorna:
        BaseChatMessageHistory: Instância do histórico específico da sessão.

    Comportamento:
        - Cria novo histórico se o session_id não existir
        - Retorna histórico existente se já estiver registrado
        - Armazenamento em dicionário global (persistência apenas durante execução)
    """
    if session_id not in store:
        # Cria novo histórico para novas sessões
        store[session_id] = InMemoryHistory()
    return store[session_id]


def convert_file_to_md(uploaded_file):
    """
    Utiliza a biblioteca docling para converter o arquivo enviado em conteúdo Markdown.
    O arquivo enviado é salvo temporariamente e seu caminho é utilizado pelo conversor.
    """
    # Cria um arquivo temporário para armazenar o conteúdo enviado
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(uploaded_file.read())
        temp_file_path = tmp.name

    # Inicializa o conversor do docling e converte o arquivo
    converter = DocumentConverter()
    result = converter.convert(temp_file_path)
    md_content = result.document.export_to_markdown()

    return md_content
