"""A FastAPI application for document processing with Qdrant. Features include:
- Document ingestion from Google Drive.
- Retrieval of documents similar to a query.
- Feature extraction from documents.
- Retrieval-Augmented Generation (RAG) for generating responses based on retrieved documents.
Utilizes OpenAI embeddings and Qdrant for vector storage and similarity search.
"""


import logging
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import (
    EMBEDDINGS_MODEL,
    EMBEDDINGS_SIZE,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
)
from src.extraction.rag import run_rag
from src.ingestion.ingest import ingest_gdrive_to_vector_store
from src.utils.ingestion import get_ingestion_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

app = FastAPI()

class IngestionRequest(BaseModel):
    """Request model for document ingestion from Google Drive.

    Attributes:
        gdrive_id: The Google Drive ID of the document to be ingested.
    """

    gdrive_id: str

class RetrievalRequest(BaseModel):
    """Request model for retrieving documents similar to a query.

    Attributes:
        query: The query string to search for similar documents.
        k: The number of similar documents to retrieve.
        folder_id: Optional; the folder ID to filter documents by.
        injection_id: Optional; the injection ID to filter documents by.
        source: Optional; the source to filter documents by.
    """

    query: str
    k: int
    folder_id: str | None = None
    injection_id: str | None = None
    source: str | None = None

class RAGRequest(BaseModel):
    """Request model for Retrieval-Augmented Generation (RAG).

    Attributes:
        question: The question based on which documents are retrieved and a response is generated.
        k: The number of documents to retrieve for generating the response.
        folder_id: Optional; the folder ID to filter documents by.
        injection_id: Optional; the injection ID to filter documents by.
        source: Optional; the source to filter documents by.
    """

    question: str
    k: int
    folder_id: str | None = None
    injection_id: str | None = None
    source: str | None = None

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

# Exception handler for generic exceptions
@app.exception_handler(Exception)
def exception_handler(_: Any, exc: Exception | int) -> JSONResponse:
    """Handles generic exceptions by returning a JSON response with a 500 status code."""

    logger.error("an exception occurred: %s", str(Exception))
    return JSONResponse(status_code=500, content={"message": "an unhandled exception occurred", "exc": str(exc)})

@app.post("/ingest/gdrive")
async def ingest_documents(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    qdrant: Qdrant = Depends(get_qdrant_client),
) -> dict[str, str]:
    """Endpoint to start the ingestion of documents from Google Drive into the vector store.
    The ingestion process is executed as a background task.
    """
    ingestion_id = get_ingestion_id()
    background_tasks.add_task(ingest_gdrive_to_vector_store, request.gdrive_id, qdrant, ingestion_id)
    return {"message": f"Ingestion started, your ingestion ID is {ingestion_id}"}

@app.post("/retrieve")
async def retrieve_documents(request: RetrievalRequest, qdrant: Qdrant = Depends(get_qdrant_client)) -> dict[str, str]:
    """Endpoint to retrieve documents similar to the query from the vector store.
    Allows filtering by folder_id, injection_id, and source.
    """
    filters = {}
    if request.folder_id:
        filters["folder_id"] = request.folder_id
    if request.injection_id:
        filters["injection_id"] = request.injection_id
    if request.source:
        filters["source"] = request.source

    results = qdrant.similarity_search(query=request.query, k=request.k, filter=filters)
    return {"documents": results}


@app.post("/rag")
async def process_rag_request(
    request: RAGRequest,
    qdrant: Qdrant = Depends(get_qdrant_client),
) -> dict[str, str]:
    """Endpoint to process a Retrieval-Augmented Generation (RAG) request.
    It retrieves documents based on the question and uses them to generate a response.
    """
    result = run_rag(qdrant, request.question, request.k, request.folder_id)
    return {"response": result}

