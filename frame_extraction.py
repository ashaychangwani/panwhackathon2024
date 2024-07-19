import asyncio
import base64
import io
import os
import time
from dataclasses import dataclass
from typing import List

import cv2
import dill
import numpy as np
from openai import AzureOpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from skimage.metrics import structural_similarity as ssim

from shared_state import TaskStatus, tasks_status
from transcription import TranscriptSentence, process_video

client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2024-02-15-preview",
)


@dataclass
class Frame:
    image: bytes
    timestamp: float

    def to_openai_image(self):
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

    def to_openai_message(self):
        return ChatCompletionUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": f"Frame at {self.timestamp} seconds"},
                {
                    "type": "text",
                    "text": self.caption,
                },
            ],
        )

    def set_caption(self, caption):
        self.caption = caption


async def calculate_ssim(frame1, frame2):
    # Resize frames to speed up SSIM calculation
    frame1_resized = cv2.resize(frame1, (320, 240))
    frame2_resized = cv2.resize(frame2, (320, 240))
    return ssim(
        frame1_resized, frame2_resized, channel_axis=len(frame1_resized.shape) - 1
    )


def caption_image(
    image: Frame,
    new_context: List[ChatCompletionUserMessageParam],
    running_object: List[dict],
):
    messages = []
    for obj in new_context:
        if isinstance(obj, Frame):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=[
                        {
                            "type": "text",
                            "text": f"Frame at {obj.timestamp} seconds. Caption: {obj.caption}",
                        }
                    ],
                )
            )
        else:
            messages.append(obj)
    if running_object is not None:
        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=running_object,
            )
        )
    messages.append(
        ChatCompletionSystemMessageParam(
            role="system",
            content=f"Be as detailed as possible. The frame is from a screenshot in a virtual presentation. Understand what is the highlight and caption to gather the most relevant and useful information. The caption should contain all of the relevant information in the slides/frame while also including the textual transcript. Do not repeat any information already included in the captions of the above frames. Caption only the primary forcus of the frame. Do not include any irrelevant information that is not the focus of the presentation. If anything has changed in the frame compared to the last frame, include every relevant change in the caption.",
        )
    )
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": f"Given the above context, caption this frame that was captured at timestamp: {image.timestamp} seconds.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image.image}"},
                },
            ],
        )
    )

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=512
    )
    return response.choices[0].message.content


def process_context(context: List[TranscriptSentence | Frame], task_id: str):
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
            tasks_status[task_id].append(
                f"Contextualizing frame {len([1 for obj in new_context if isinstance(obj, Frame)])} of total {len([1 for obj in context if isinstance(obj, Frame)])}"
            )
            obj.set_caption(caption_image(obj, new_context, running_object))
            tasks_status[task_id].complete(
                f"Contextualizing frame {len([1 for obj in new_context if isinstance(obj, Frame)])} of total {len([1 for obj in context if isinstance(obj, Frame)])}"
            )
            new_context.append(obj)

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


def generate_frames(video_path: str, timestamps: List[int]) -> None:
    cap = cv2.VideoCapture(video_path)
    for i, timestamp in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * cap.get(cv2.CAP_PROP_FPS)))
        ret, frame = cap.read()
        if not ret:
            print(f"Frame not found at timestamp {timestamp}.")
            continue
        _, encoded_frame = cv2.imencode(".jpg", frame)
        with open(f"frames/{timestamp}.png", "wb") as f:
            f.write(encoded_frame)
    cap.release()


async def generate_context(
    video_path, task_id: str
) -> List[Frame | TranscriptSentence]:
    try:
        tasks_status[task_id].append("Transcribing video")
        transcript_sentences = await process_video(video_path, task_id)
        tasks_status[task_id].complete("Transcribing video")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 5
        frames = []
        prev_frame = None
        frame_count = 0

        tasks_status[task_id].append("Extracting important frames from video")
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
        tasks_status[task_id].complete("Extracting important frames from video")

        combined_context = frames + transcript_sentences
        combined_context.sort(key=lambda x: x.timestamp)
        tasks_status[task_id].append("Generating context")
        combined_context = process_context(combined_context, task_id)
        tasks_status[task_id].complete("Generating context")
        return combined_context
    except Exception as e:
        tasks_status[task_id].append(f"Error: {e}")
        return []


if __name__ == "__main__":
    asyncio.run(generate_context("recording.mp4", "example_task_id"))
