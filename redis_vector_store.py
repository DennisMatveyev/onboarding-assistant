import hashlib
import os
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
)
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.index import AsyncSearchIndex

from configs import DOCS_PATH, EMBEDDING_MODEL
from log import logger


llm_cache = SemanticCache(
    name="llm_cache_onboarding",
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    distance_threshold=0.01,
    vectorizer=HFTextVectorizer("redis/langcache-embed-v2"),
)

schema = {
    "index": {
        "name": "onboarding_idx",
    },
    "fields": [
        {"name": "text", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 3072, "distance_metric": "cosine", "algorithm": "hnsw"
            }
        },
        {"name": "source", "type": "tag"},
        {"name": "chunk_id", "type": "numeric"}
    ]
}

index = AsyncSearchIndex.from_dict(
    schema, redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
)

FINGERPRINT_KEY = "onboarding_idx:fingerprint"

def _docs_fingerprint() -> str:
    entries = [
        f"{path}:{path.stat().st_mtime}"
        for path in sorted(Path(DOCS_PATH).rglob("*"))
        if path.is_file()
    ]
    return hashlib.md5("\n".join(entries).encode()).hexdigest()

def _load_documents():
    loaders = [
        DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DOCS_PATH, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    return docs

async def _embed_docs_to_index():
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunks = (
        RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        .split_documents(_load_documents())
    )
    texts = [doc.page_content for doc in chunks]
    vectors = await embeddings_model.aembed_documents(texts)
    records = []

    for i, (doc, vec) in enumerate(zip(chunks, vectors)):
        records.append({
            "text": doc.page_content,
            "embedding": np.array(vec, dtype=np.float32).tobytes(),
            "source": Path(doc.metadata.get("source", "")).name,
            "chunk_id": i,
        })

    await index.load(records)

async def sync_documents():
    if not await index.exists():
        logger.info("Creating index...")
        await index.create()
        logger.info("Index created successfully.")
    
    current = _docs_fingerprint()
    stored = await index.client.get(FINGERPRINT_KEY)

    if stored is not None and stored.decode() == current:
        logger.info("Documents unchanged. Skipping ingestion.")
        return

    logger.info("Documents changed. Re-ingesting...")
    await index.clear()
    await _embed_docs_to_index()
    await index.client.set(FINGERPRINT_KEY, current)
    llm_cache.clear()
    logger.info("Ingestion complete. LLM cache cleared.")
