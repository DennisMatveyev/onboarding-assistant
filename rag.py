from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from configs import SYSTEM_PROMPT, MODEL_NAME, TEMPERATURE
from redis_retriever import RedisRetriever


_model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

_retriever = RedisRetriever()

_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{input}")
])

_question_answer_chain = create_stuff_documents_chain(_model, _prompt_template)

retrieval_chain = create_retrieval_chain(_retriever, _question_answer_chain)
