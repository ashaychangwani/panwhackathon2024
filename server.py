import asyncio
import base64
from dataclasses import dataclass
from typing import List

import dill
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from frame_extraction import FrameContext, generate_context
from transcription import TranscriptSentence


@dataclass
class Frame:
    image: bytes
    timestamp: float
    caption: str
    context: List[TranscriptSentence]


@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(12))
async def call_openai(image_data: str, body: str, timestamp: float) -> str:
    response = await AsyncOpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Textual Narration: {body}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": f"The above image is captured at timestamp {timestamp}. An image is captured every 5 seconds, so don't explain things that are already known. Explain only things that are newly relevant to the conversation based on the context provided. If text or code is being referenced, then write out that code or text.",
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


async def generate_captions(video_path: str) -> List[str]:
    frame_contexts = await generate_context(video_path)
    frames = []

    async def process_frame(frame_context):
        body = "\n".join([str(sentence) for sentence in frame_context.transcript])
        timestamp = frame_context.timestamp
        # image_data = base64.b64decode(frame_context.image)
        # with open("image.jpg", "wb") as f:
        #     f.write(image_data)
        caption = await call_openai(frame_context.image, body, timestamp)
        print(f"[{timestamp}] {caption})")
        return Frame(
            image=frame_context.image,
            timestamp=timestamp,
            caption=caption,
            context=frame_context.transcript,
        )

    frames = []
    for fc in frame_contexts:
        frames.append(await process_frame(fc))
    # frames = await asyncio.gather(*[process_frame(fc) for fc in frame_contexts])
    return frames


if __name__ == "__main__":
    import asyncio

    video_path = "recording.mp4"
    captions = asyncio.run(generate_captions(video_path))
    with open("captions.dill", "wb") as f:
        dill.dump(captions, f)
