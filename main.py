import asyncio
import time
import uuid
from typing import List

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from frame_extraction import generate_context
from server import get_conversation_response, process_file
from shared_state import TaskStatus, tasks_status
from transcription import encode_filename


class Video(BaseModel):
    video_url: str
    objective: str


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: List[Message]
    text: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/process")
async def process_video_endpoint(video: Video, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks_status[task_id] = TaskStatus()
    tasks_status[task_id].empty()

    background_tasks.add_task(
        process_file_task, task_id, video.video_url, video.objective
    )
    return {"task_id": task_id, "status": "Task started"}


def process_file_task(task_id: str, video_url: str, objective: str):
    tasks_status[task_id].append("Downloading video")
    tasks_status[task_id].complete("Downloading video")
    asyncio.run(process_file(video_url, objective, task_id))


@app.get("/status/{task_id}")
async def poll_status(task_id: str):
    return {"status": tasks_status.get(task_id, None).get_tasks()}


@app.get("/display/{token}")
async def display_content(token: str):
    return {"text": open(f"frames/{encode_filename(token)}.md", "r").read()}


@app.post("/conversation/{token}")
def get_response(conversation: Conversation, token: str):
    return asyncio.run(get_conversation_response(conversation, token))


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0", reload=True)
