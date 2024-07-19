import asyncio
import base64
import io
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from transcription import TranscriptSentence, process_video


@dataclass
class FrameContext:
    image: bytes
    transcript: List[TranscriptSentence]
    timestamp: float


async def calculate_ssim(frame1, frame2):
    # Resize frames to speed up SSIM calculation
    frame1_resized = cv2.resize(frame1, (320, 240))
    frame2_resized = cv2.resize(frame2, (320, 240))
    return ssim(
        frame1_resized, frame2_resized, channel_axis=len(frame1_resized.shape) - 1
    )


async def generate_context(video_path) -> List[FrameContext]:
    try:
        transcript_sentences = await process_video(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 3
        contexts = []
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
                print(f"[{frame_count / fps}")
                _, encoded_frame = cv2.imencode(".jpg", frame)
                byte64_image_data = base64.b64encode(encoded_frame).decode("utf-8")
                second = frame_count / fps
                matching_sentences = [
                    sentence
                    for sentence in transcript_sentences
                    if abs(sentence.timestamp - second) <= 30
                ]
                contexts.append(
                    FrameContext(
                        image=byte64_image_data,
                        transcript=matching_sentences,
                        timestamp=second,
                    )
                )
                prev_frame = frame.copy()

            frame_count += int(fps * frame_interval)

        cap.release()
        return contexts
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    asyncio.run(generate_context("recording.mp4"))
