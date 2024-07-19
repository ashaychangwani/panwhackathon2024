import asyncio

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from server import process_file


class Video(BaseModel):
    video_url: str
    objective: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/process")
async def process_video(video: Video):
    video_url = video.video_url
    objective = video.objective
    return await process_file(video_url, objective)


@app.post("/upload")
async def upload_output(video: Video):
    return await (video.video_url, video.objective)


if __name__ == "__main__":
    # uvicorn.run(app, port=8080, host="0.0.0.0")
    asyncio.run(
        process_video(Video(video_url="recording.mp4", objective="Generate a runbook"))
    )
