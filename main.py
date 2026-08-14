import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from graph import create_graph
from log import logger
from redis_vector_store import sync_documents


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the application...")
    await sync_documents()
    app.state.graph = await create_graph()
    yield
    logger.info("Shutting down the application...")


app = FastAPI(lifespan=lifespan)


@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    state_graph = websocket.app.state.graph
    thread_id = websocket.query_params.get("thread_id") or uuid.uuid4().hex
    workflow_config = {"configurable": {"thread_id": thread_id}}
    await websocket.send_text(f"__session__:{thread_id}")

    try:
        while True:
            query = await websocket.receive_text()
            response = await state_graph.ainvoke(
                {"messages": [HumanMessage(query)]}, workflow_config
            )
            await websocket.send_text(response["messages"][-1].content)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for thread {thread_id}")

    except Exception as e:
        logger.error(f"Unexpected Error for thread {thread_id}: {e}")
        await websocket.close(reason="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
