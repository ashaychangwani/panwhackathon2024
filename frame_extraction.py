import asyncio
import base64
import io
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from skimage.metrics import structural_similarity as ssim

from transcription import TranscriptSentence, process_video


@dataclass
class Frame:
    image: bytes
    timestamp: float

    def to_openai_message(self):
        return ChatCompletionUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": f"Frame at {self.timestamp} seconds"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.image}"},
                },
            ],
        )


async def calculate_ssim(frame1, frame2):
    # Resize frames to speed up SSIM calculation
    frame1_resized = cv2.resize(frame1, (320, 240))
    frame2_resized = cv2.resize(frame2, (320, 240))
    return ssim(
        frame1_resized, frame2_resized, channel_axis=len(frame1_resized.shape) - 1
    )


def process_context(context: List[TranscriptSentence | Frame]):
    # this function will combine transcript sentences into a single object
    new_context = []
    running_object = None
    for obj in context:
        if isinstance(obj, Frame):
            if running_object is not None:
                new_context.append(
                    ChatCompletionUserMessageParam(role="user", content=running_object)
                )
                running_object = None
            new_context.append(obj.to_openai_message())
        elif isinstance(obj, TranscriptSentence):
            if running_object is None:
                running_object = [
                    {
                        "type": "text",
                        "text": str(obj),
                    }
                ]
            else:
                running_object.append(
                    {
                        "type": "text",
                        "text": str(obj),
                    }
                )
    if running_object is not None:
        new_context.append(
            ChatCompletionUserMessageParam(role="user", content=running_object)
        )
    return new_context


async def generate_context(video_path) -> List[Frame | TranscriptSentence]:
    try:
        transcript_sentences = await process_video(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 3
        frames = []
        prev_frame = None
        frame_count = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is None:
                is_different = True
            else:
                is_different = await calculate_ssim(frame, prev_frame) < 0.97

            if is_different:
                print(
                    f"[{frame_count / fps} / {cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps}] Frame added"
                )
                _, encoded_frame = cv2.imencode(".jpg", frame)
                byte64_image_data = base64.b64encode(encoded_frame).decode("utf-8")
                second = frame_count / fps
                frames.append(
                    Frame(
                        image=byte64_image_data,
                        timestamp=second,
                    )
                )
                prev_frame = frame.copy()

            frame_count += int(fps * frame_interval)

        cap.release()
        combined_context = frames + transcript_sentences
        combined_context.sort(key=lambda x: x.timestamp)
        combined_context = process_context(combined_context)
        return combined_context
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    asyncio.run(generate_context("recording.mp4"))
