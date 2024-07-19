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

from transcription import TranscriptSentence, process_video

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
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
        model="gpt-4o-mini", messages=messages, max_tokens=512
    )
    return response.choices[0].message.content


def process_context(context: List[TranscriptSentence | Frame]):
    # this function will combine transcript sentences into a single object
    new_context = []
    running_object = None
    for idx, obj in enumerate(context):
        if isinstance(obj, Frame):
            if running_object is not None:
                new_context.append(
                    ChatCompletionUserMessageParam(role="user", content=running_object)
                )
                running_object = None
            obj.set_caption(caption_image(obj, new_context, running_object))
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
        print(f"Processed {idx + 1} of {len(context)} at timestamp {obj.timestamp}")
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
