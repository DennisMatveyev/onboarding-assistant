from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from redisvl.query import VectorQuery

from configs import EMBEDDING_MODEL
from redis_vector_store import index


embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


class RedisRetriever(BaseRetriever):
    def _get_relevant_documents(self, *_):
        raise NotImplementedError("Use async retrieval via ainvoke")

    async def _aget_relevant_documents(self, query: str, *_) -> list[Document]:
        vec = await embeddings.aembed_query(query)
        q = VectorQuery(
            vector=vec,
            vector_field_name="embedding",
            return_fields=["text", "vector_distance"],
            num_results=3,
        )
        results = await index.query(q)
        
        return [Document(page_content=r["text"]) for r in results]
