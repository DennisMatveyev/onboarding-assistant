import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from graph import state_graph
from log import logger
from redis_vector_store import sync_documents


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the application...")
    await sync_documents()
    yield
    logger.info("Shutting down the application...")


app = FastAPI(lifespan=lifespan)


@app.websocket("/chat/{user_id}")
async def chat(websocket: WebSocket, user_id: str):
    await websocket.accept()
    workflow_config = {"configurable": {"thread_id": f"{user_id}"}}

    try:
        while True:
            query = await websocket.receive_text()
            input_messages = [HumanMessage(query)]
            output = await state_graph.ainvoke({"messages": input_messages}, workflow_config)
            await websocket.send_text(output["messages"][-1].content)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
        await state_graph.checkpointer.adelete_thread(f"{user_id}")

    except Exception as e:
        logger.error(f"Unexpected Error for user {user_id}: {e}")
        await asyncio.gather(
            state_graph.checkpointer.adelete_thread(f"{user_id}"),
            websocket.close(reason="Internal server error")
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
