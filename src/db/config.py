import os

# Milvus connection settings
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Milvus collection settings
COLLECTION_NAME = "demo_collection"
VECTOR_DIM = 768  # Default embedding function output

# Index parameters
INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

# Search parameters
SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},
}
