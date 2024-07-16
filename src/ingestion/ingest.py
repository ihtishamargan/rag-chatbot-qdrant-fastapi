"""Ingestion Module for Text to Vectors Conversion.

This module is responsible for ingesting text documents from Google Drive,
splitting them into manageable chunks, and storing the resulting vectors
in a Qdrant vector store. It supports various document types including
documents, sheets, and PDFs.
"""

import logging
import sys
from collections.abc import Generator
from langchain_community.document_loaders import (
    GoogleDriveLoader,
    UnstructuredFileIOLoader,
)

from src.config import GOOGLE_ACCOUNT_FILE


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