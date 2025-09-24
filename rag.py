from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
)
from configs import DOCS_PATH, EMBEDDING_MODEL, SYSTEM_PROMPT
from llm import model


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


def _get_vectorstore():
    splitted_docs = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(_load_documents())
    
    return InMemoryVectorStore.from_documents(splitted_docs, OpenAIEmbeddings(model=EMBEDDING_MODEL))


_vectorstore = _get_vectorstore()
_retriever = _vectorstore.as_retriever()

_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT + "\nRelevant information:\n{context}"),
    HumanMessagePromptTemplate.from_template("{input}")
])

_question_answer_chain = create_stuff_documents_chain(model, _prompt_template)

retrieval_chain = create_retrieval_chain(_retriever, _question_answer_chain)
