import os
from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import START, END, MessagesState, StateGraph

from redis_vector_store import llm_cache
from rag import retrieval_chain


def _create_graph() -> StateGraph:
    graph = StateGraph(state_schema=MessagesState)

    graph.add_node("retrieval_chain_node", _call_retrieval_chain)
    graph.add_node("nearest_birthday_node", _get_nearest_birthday)
    graph.add_node("router_node", lambda state: state)
    
    graph.add_edge(START, "router_node")
    graph.add_conditional_edges(
        "router_node",
        _router,
        {
            "call_retrieval_chain": "retrieval_chain_node",
            "call_get_nearest_birthday": "nearest_birthday_node",
        }
    )
    graph.add_edge("retrieval_chain_node", END)
    graph.add_edge("nearest_birthday_node", END)

    checkpointer = None
    with RedisSaver.from_conn_string(os.getenv("REDIS_URL", "redis://localhost:6379")) as _checkpointer:
        _checkpointer.setup()
        checkpointer = _checkpointer

    return graph.compile(checkpointer=checkpointer)


def _router(state: MessagesState) -> str:
    if "nearest birthday" in state["messages"][-1].content.lower():
        return "call_get_nearest_birthday"
    
    return "call_retrieval_chain"


async def _get_nearest_birthday(_: MessagesState) -> dict:
    current_date = datetime.now().strftime("%Y-%m-%d")
    input_ = f"When is the nearest birthday after {current_date}?"

    hits = await llm_cache.acheck(prompt=input_)
    if hits:
        return {"messages": [AIMessage(content=hits[0]["response"])]}

    result = await retrieval_chain.ainvoke({"input": input_})
    answer = result["answer"]
    await llm_cache.astore(prompt=input_, response=answer)

    return {"messages": [AIMessage(content=answer)]}


async def _call_retrieval_chain(state: MessagesState) -> dict:
    user_message = state["messages"][-1].content

    hits = await llm_cache.acheck(prompt=user_message)
    if hits:
        return {"messages": [AIMessage(content=hits[0]["response"])]}

    result = await retrieval_chain.ainvoke({"input": user_message})
    answer = result["answer"]
    await llm_cache.astore(prompt=user_message, response=answer)

    return {"messages": [AIMessage(content=answer)]}


state_graph = _create_graph()
