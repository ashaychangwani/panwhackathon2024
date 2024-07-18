import asyncio
import base64
import io
from dataclasses import dataclass

import cv2

from transcription import TranscriptSentence, process_video


@dataclass
class FrameContext:
    image: bytes
    transcript: list[TranscriptSentence]
    timestamp: float


async def generate_context(video_path):
    try:
        transcript_sentences = await process_video(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(5 * fps)
        contexts = []
        frame_count = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            _, encoded_frame = cv2.imencode(".jpg", frame)
            image_stream = io.BytesIO(encoded_frame.tobytes())
            image_data = image_stream.getvalue()
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
            frame_count += frame_interval

        cap.release()
        return contexts
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    asyncio.run(generate_context("recording.mp4"))
