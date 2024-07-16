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
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

from src.config import DEFAULT_CHUNK_SIZE, GOOGLE_ACCOUNT_FILE, UPSERT_BATCH_SIZE

logger = logging.getLogger(__name__)


def load_gdrive_documents(folder_id: str, ingestion_id: str) -> Generator[Document, None, None]:
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


def split_documents(
    documents: Iterable[Document], chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Generator[Document, None, None]:
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
    doc_splits = text_splitter.split_documents(documents)
    for idx, split in enumerate(doc_splits):
        split.metadata["chunk"] = idx
        yield split


def ingest_gdrive_to_vector_store(gdrive_id: str, qdrant: Qdrant, ingestion_id: str) -> None:
    """Ingest documents from a Google Drive folder into a Qdrant vector store.

    Parameters:
        gdrive_id (str): The ID of the Google Drive folder.
        qdrant (Qdrant): The Qdrant vector store instance.

    Raises:
        SystemExit: If ingestion fails.
    """
    batch_size = UPSERT_BATCH_SIZE
    batch = []

    try:
        for document in load_gdrive_documents(folder_id=gdrive_id, ingestion_id=ingestion_id):
            for doc_split in split_documents([document]):
                batch.append(doc_split)
                if len(batch) >= batch_size:
                    qdrant.add_documents(batch)
                    batch = []

        # Process the remaining batch if it's not empty
        if batch:
            qdrant.add_documents(batch)

    except ValueError as e:
        error_message = f"An error occurred during ingestion for ingestion_id= {ingestion_id}: {e}"
        logger.error(error_message)
        sys.exit(1)
    success_message = f"Ingestion completed successfully for ingestion_id: {ingestion_id}"
    logger.info(success_message)
