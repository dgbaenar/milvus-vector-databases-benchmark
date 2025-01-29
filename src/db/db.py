from pathlib import Path
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    model,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.db.config import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    VECTOR_DIM,
    INDEX_PARAMS,
    SEARCH_PARAMS,
)


# Default Milvus embedding function
embedding_fn = model.DefaultEmbeddingFunction()


def connect_to_milvus():
    """
    Establish a connection with Milvus.
    """
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")


def create_collection():
    """
    Creates a collection in Milvus if it doesn't exist.
    Returns the collection object.
    """
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    # Define collection fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, description="Embedded documents collection")

    # Create collection
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create an index on the vector field
    collection.create_index(field_name="vector", index_params=INDEX_PARAMS)
    print(f"Created collection '{COLLECTION_NAME}' with index: {INDEX_PARAMS}")

    return collection


def extract_text_from_html(file_path: Path, docs_dir: Path) -> Dict[str, str]:
    """
    Extracts clean text from an HTML file and generates its original URL.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""

    # Generate source URL
    relative_path = file_path.relative_to(docs_dir)
    if relative_path.name == "index.html":
        source_url = f"https://www.robinsonandhenry.com/blog/{relative_path.parent}/"
    else:
        source_url = f"https://www.robinsonandhenry.com/blog/{relative_path.stem}/"

    source_url = source_url.replace("//", "/")  # Normalize

    return {"source": source_url, "text": text}


def process_and_insert_data(docs_dir: Path):
    """
    Processes HTML files, splits them into chunks, generates embeddings, and inserts them into Milvus.
    """
    html_files = list(docs_dir.rglob("*.html"))
    documents = [extract_text_from_html(path, docs_dir) for path in html_files]

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = []
    for doc in documents:
        chunked_texts = text_splitter.create_documents(
            texts=[doc["text"]],
            metadatas=[{"source": doc["source"]}]
        )
        for chunk in chunked_texts:
            chunks.append({
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
            })

    # Generate embeddings
    texts_list = [c["text"] for c in chunks]
    vectors = embedding_fn.encode_documents(texts_list)

    # Prepare data for insertion
    data_to_insert = [
        {"text": chunk["text"], "source": chunk["source"], "vector": vectors[idx]}
        for idx, chunk in enumerate(chunks)
    ]

    # Insert into Milvus
    collection = Collection(COLLECTION_NAME)
    insert_data(collection, data_to_insert)

    return len(data_to_insert), len(html_files)


def insert_data(collection: Collection, data: List[Dict[str, Any]]):
    """
    Inserts a list of embedded documents into Milvus.
    """
    vectors = [d["vector"] for d in data]
    texts = [d["text"] for d in data]
    sources = [d["source"] for d in data]

    collection.insert([vectors, texts, sources])
    print(f"Inserted {len(vectors)} records into Milvus collection {collection.name}")

    collection.flush()
    collection.load()


def search_in_milvus(
    query_text: str, limit: Optional[int] = None, similarity_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Searches Milvus using a text query with optional result limit and similarity threshold.
    """
    if limit is None:
        limit = 10

    query_vector = embedding_fn.encode_queries([query_text])

    collection = Collection(COLLECTION_NAME)
    results = collection.search(
        data=query_vector,
        anns_field="vector",
        param=SEARCH_PARAMS,
        limit=limit,
        output_fields=["text", "source"],
    )

    if not results:
        return []

    hits = []
    for hit in results[0]:  # First (and only) query response
        distance = hit.distance
        similarity = 1 - distance  # Convert distance to similarity

        if similarity_threshold is not None and similarity < similarity_threshold:
            continue

        hits.append({
            "score": similarity,
            "text": hit.entity.get("text"),
            "source": hit.entity.get("source"),
        })

    return hits
