from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph

from rag import retrieval_chain


def _create_graph() -> StateGraph:
    workflow = StateGraph(state_schema=MessagesState)

    workflow.add_node("qa_node", _call_retrieval_chain)
    workflow.add_edge(START, "qa_node")
    workflow.add_edge("qa_node", END)

    return workflow.compile(checkpointer=MemorySaver())


async def _call_retrieval_chain(state: MessagesState) -> dict:
    user_message = state["messages"][-1].content
    result = await retrieval_chain.ainvoke({"input": user_message})
    answer = result["answer"]
    
    return {"messages": [AIMessage(content=answer)]}


state_graph = _create_graph()
