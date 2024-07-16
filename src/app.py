"""A FastAPI application for document processing with Qdrant. Features include:
- Document ingestion from Google Drive.
- Retrieval of documents similar to a query.
- Feature extraction from documents.
- Retrieval-Augmented Generation (RAG) for generating responses based on retrieved documents.
Utilizes OpenAI embeddings and Qdrant for vector storage and similarity search.
"""

import logging

from fastapi import FastAPI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import (
    EMBEDDINGS_MODEL,
    EMBEDDINGS_SIZE,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

app = FastAPI()


def get_qdrant_client() -> Qdrant:
    """Initialize the Qdrant client."""
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
    collections = client.get_collections().collections
    collections_list = [collection.name for collection in collections]

    if QDRANT_COLLECTION_NAME not in collections_list:
        client.create_collection(
            QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDINGS_SIZE, distance=models.Distance.COSINE),
        )

    return Qdrant(client, QDRANT_COLLECTION_NAME, embeddings)