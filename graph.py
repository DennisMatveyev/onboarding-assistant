from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph

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

    return graph.compile(checkpointer=MemorySaver())


def _router(state: MessagesState) -> str:
    if "nearest birthday" in state["messages"][-1].content.lower():
        return "call_get_nearest_birthday"
    
    return "call_retrieval_chain"


async def _get_nearest_birthday(_: MessagesState) -> dict:
    current_date = datetime.now().strftime("%Y-%m-%d")
    input_ = f"When is the nearest birthday after {current_date}?"
    result = await retrieval_chain.ainvoke({"input": input_})
    answer = result["answer"]

    return {"messages": [AIMessage(content=answer)]}


async def _call_retrieval_chain(state: MessagesState) -> dict:
    user_message = state["messages"][-1].content
    result = await retrieval_chain.ainvoke({"input": user_message})
    answer = result["answer"]
    
    return {"messages": [AIMessage(content=answer)]}


state_graph = _create_graph()
