"""This module defines functions for loading chat prompt templates
and executing the RAG model using a Qdrant vector store for context retrieval.
"""

import os
from typing import cast

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI

from src.config import MODEL_FOR_RAG, RAG_PROMPT_FILE


def load_chat_prompt_from_file(file_path: str) -> ChatPromptTemplate:
    """Loads a chat prompt template from a specified file.

    Parameters:
    - file_path (str): The path to the template file.

    Returns:
    - ChatPromptTemplate: The loaded chat prompt template, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        raise ValueError(
            f"File not found: {file_path}",
        )

    with open(file_path, encoding="utf-8") as file:
        txt = file.read()

    return ChatPromptTemplate.from_template(txt)


def run_rag(qdrant: Qdrant, question: str, k: int = 2, folder_id: str | None = None) -> str:
    """Executes the RAG model to answer a question using a Qdrant vector store.

    Parameters:
    - qdrant (Qdrant): The Qdrant instance to use for retrieving context.
    - question (str): The question to be answered.
    - k (int, optional): The number of documents to retrieve for context. Defaults to 2.
    - folder_id (str, optional): An optional folder ID to filter the documents.

    Returns:
    - str: The answer generated by the RAG model.

    Raises:
    - ValueError: If `folder_id` is not None or a string, or if `question` is not a string.
    """
    prompt = load_chat_prompt_from_file(RAG_PROMPT_FILE)
    model = ChatOpenAI(model=MODEL_FOR_RAG)
    output_parser = StrOutputParser()

    if folder_id is not None and not isinstance(folder_id, str):
        raise ValueError("folder_id must be a string or None")
    if not isinstance(question, str):
        raise ValueError("question must be a string")

    retriever = qdrant.as_retriever(search_kwargs={"k": k, "filter": {"folder_id": folder_id}})
    setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    chain = setup_and_retrieval | prompt | model | output_parser
    result = chain.invoke(question)
    return cast(str, result)