"""Ingestion Module for Text to Vectors Conversion.

This module is responsible for ingesting text documents from Google Drive,
splitting them into manageable chunks, and storing the resulting vectors
in a Qdrant vector store. It supports various document types including
documents, sheets, and PDFs.
"""

import logging
import sys
from collections.abc import Generator, Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    GoogleDriveLoader,
    UnstructuredFileIOLoader,
)

from src.config import GOOGLE_ACCOUNT_FILE, DEFAULT_CHUNK_SIZE
from unstructured.documents.base import Document

logger = logging.getLogger(__name__)

def load_gdrive_documents(folder_id: str, ingestion_id: str) -> Generator[dict, None, None]:
    """Load documents from a specified Google Drive folder.

    Parameters:
        folder_id (str): The ID of the Google Drive folder to load documents from.
        ingestion_id (str): The ingestion ID to be associated with the loaded documents.

    Yields:
        Document: Yields loaded documents one by one.

    Raises:
        SystemExit: If documents loading fails.
    """
    try:
        loader = GoogleDriveLoader(
            service_account_key=GOOGLE_ACCOUNT_FILE,
            folder_id=folder_id,
            file_loader_cls=UnstructuredFileIOLoader,
            file_types=["document", "sheet", "pdf"],
            recursive=True,
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(
                {
                    "source": doc.metadata["source"],
                    "document_name": doc.metadata["title"],
                    "folder_id": folder_id,
                    "ingestion_id": ingestion_id,
                }
            )
            yield doc
    except ValueError as e:
        logger.error("Failed to load documents: %s", e)
        sys.exit(1)


def split_documents(documents: Iterable[dict], chunk_size: int = DEFAULT_CHUNK_SIZE) -> Generator[Document, None, None]:
    """Split documents into smaller chunks based on the specified chunk size.

    Parameters:
        documents (List[Document]): A list of documents to be split.
        chunk_size (int, optional): The size of each chunk. Defaults to DEFAULT_CHUNK_SIZE.

    Yields:
        Document: Yields document chunks one by one.

    Raises:
        SystemExit: If document splitting fails.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    doc_splits = text_splitter.split_documents(documents)  # type: ignore[arg-type]
    for idx, split in enumerate(doc_splits):
        split.metadata["chunk"] = idx
        yield split