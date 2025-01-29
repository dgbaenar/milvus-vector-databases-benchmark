from fastapi import FastAPI, Query
from pathlib import Path
from typing import Optional
from src.db.db import (
    connect_to_milvus,
    create_collection,
    process_and_insert_data,
    search_in_milvus,
)

app = FastAPI(
    title="Milvus Bulk Load & Search API",
    description="API for bulk data insertion and search using Milvus.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """
    Initializes the connection to Milvus and ensures the collection exists.
    """
    connect_to_milvus()
    create_collection()


@app.post("/bulk")
def bulk_insert():
    """
    Processes and inserts HTML documents into Milvus.
    """
    docs_dir = Path("data/www.robinsonandhenry.com/blog")
    num_chunks, num_files = process_and_insert_data(docs_dir)
    return {"message": f"Inserted {num_chunks} chunks from {num_files} HTML files into Milvus."}


@app.get("/search")
def search(
    query: str = Query(..., description="Search query text"),
    limit: Optional[int] = Query(None, description="Limit results"),
    similarity_threshold: Optional[float] = Query(None, description="Similarity threshold [0-1]"),
):
    """
    Searches Milvus for similar text based on query.
    """
    results = search_in_milvus(query, limit, similarity_threshold)
    return results
