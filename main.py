import asyncio
import time
import uuid

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from frame_extraction import generate_context
from server import process_file
from shared_state import TaskStatus, tasks_status
from transcription import process_video


class Video(BaseModel):
    video_url: str
    objective: str


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


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0", reload=True)
