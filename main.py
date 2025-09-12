import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from graph import state_graph
from log import logger


load_dotenv()

app = FastAPI()


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
