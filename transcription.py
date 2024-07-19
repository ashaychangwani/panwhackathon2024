import asyncio
import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import List

import dill
import dotenv
import fastapi_poe as fp
import moviepy.editor as mp
from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydub import AudioSegment

dotenv.load_dotenv()


@dataclass
class TranscriptSentence:
    text: str
    timestamp: float

    def __str__(self) -> str:
        return f"[{self.timestamp}] {self.text}"

    def to_openai_message(self):
        return ChatCompletionUserMessageParam(
            role="user",
            content=str(self),
        )


def extract_audio_from_video(video_file_path: str, output_audio_file_path: str) -> None:
    video = mp.VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(output_audio_file_path)


def encode_filename(filename):
    """
    Encode the given filename using SHA-256 hashing.
    """
    # Create a SHA-256 hash object
    sha256 = hashlib.sha256()

    # Encode the filename to bytes and update the hash object
    sha256.update(filename.encode())

    # Get the hexadecimal representation of the hash
    encoded_filename = sha256.hexdigest()

    return encoded_filename


async def get_responses(api_key: str, messages: List[str]) -> None:
    async for partial in fp.get_bot_response(
        messages=messages, bot_name="GPT-3.5-Turbo", api_key=api_key
    ):
        print(partial)


def get_audio_segment_size(audio_segment: AudioSegment) -> int:
    return len(audio_segment.raw_data)


def split_wav_file(file_path: str, segment_size_mb: int = 20) -> List[str]:
    audio = AudioSegment.from_wav(file_path)
    segment_size_bytes = segment_size_mb * 1024 * 1024
    bytes_per_ms = len(audio.raw_data) / len(audio)
    segment_duration_ms = segment_size_bytes / bytes_per_ms

    start = 0
    end = len(audio)
    segment_index = 0
    output_files = []
    while start < end:
        segment_end = min(start + segment_duration_ms, end)
        segment = audio[start:segment_end]

        unique_id = uuid.uuid4()
        output_file = f"audio_segments/{unique_id}_segment_{segment_index}.wav"
        segment.export(output_file, format="wav")
        output_files.append(output_file)

        start = segment_end
        segment_index += 1
    return output_files


async def get_transcription_async(
    audio_file_path: str, start_timestamp: float
) -> List[TranscriptSentence]:
    audio_file = open(audio_file_path, "rb")
    transcription = []
    transcript = await AsyncOpenAI().audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    for segment in transcript.model_extra["segments"]:
        transcription.append(
            TranscriptSentence(
                segment["text"], round(segment["end"] + start_timestamp, 1)
            )
        )
    return transcription


async def transcribe_segments(
    file_path: str, segment_size_mb: int = 20
) -> List[TranscriptSentence]:
    output_files = split_wav_file(file_path, segment_size_mb)
    tasks = []
    cumulative_duration = 0.0
    for output_file in output_files:
        audio_segment = AudioSegment.from_wav(output_file)
        tasks.append(get_transcription_async(output_file, cumulative_duration))
        cumulative_duration += len(audio_segment) / 1000.0
    transcriptions = await asyncio.gather(*tasks)
    combined_transcription = [
        sentence for transcription in transcriptions for sentence in transcription
    ]
    return combined_transcription


async def process_video(video_file_path: str) -> List[TranscriptSentence]:
    transcription_file = encode_filename(video_file_path)
    if os.path.exists(f"transcriptions/{transcription_file}.dill"):
        loaded_transcription = dill.load(
            open(f"transcriptions/{transcription_file}.dill", "rb")
        )
        transcription = [
            TranscriptSentence(**sentence.__dict__) for sentence in loaded_transcription
        ]
        return transcription
    audio_file_path = f"audio_segments/{uuid.uuid4().hex}.wav"
    extract_audio_from_video(video_file_path, audio_file_path)
    transcription = await transcribe_segments(audio_file_path)
    dill.dump(transcription, open(f"transcriptions/{transcription_file}.dill", "wb"))
    return transcription


if __name__ == "__main__":
    video_file_path = "recording.mp4"
    transcription = asyncio.run(process_video(video_file_path))
    print(transcription)
