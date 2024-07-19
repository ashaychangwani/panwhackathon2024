import asyncio
import hashlib
import os
import re
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

from shared_state import tasks_status

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


async def extract_audio_from_video(
    video_file_path: str, output_audio_file_path: str
) -> None:
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
    base_path, _ = os.path.splitext(os.path.basename(file_path))
    bytes_per_ms = len(audio.raw_data) / len(audio)
    segment_duration_ms = segment_size_bytes / bytes_per_ms

    start = 0
    end = len(audio)
    segment_index = 0
    output_files = []
    while start < end:
        segment_end = min(start + segment_duration_ms, end)
        segment = audio[start:segment_end]
        output_file = f"audio_segments/{base_path}_segment_{segment_index}.wav"
        segment.export(output_file, format="wav")
        output_files.append(output_file)

        start = segment_end
        segment_index += 1
    return output_files


async def get_transcription_async(
    audio_file_path: str, start_timestamp: float, task_id: str, segment_number: int
) -> List[TranscriptSentence]:
    def time_to_seconds(time_str):
        h, m, s_ms = time_str.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    transcription = []
    tasks_status[task_id].append(f"Transcribing segment {segment_number}")
    transcript = await AsyncOpenAI().audio.transcriptions.create(
        file=open(audio_file_path, "rb"),
        model="whisper-1",
        response_format="srt",
        timestamp_granularities=["segment"],
    )
    tasks_status[task_id].complete(f"Transcribing segment {segment_number}")
    pattern = re.compile(
        r"\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n",
        re.DOTALL,
    )
    matches = pattern.findall(transcript)
    transcription.extend(
        [
            TranscriptSentence(
                match[2].replace("\n", " "),
                round(time_to_seconds(match[1]) + start_timestamp, 1),
            )
            for match in matches
        ]
    )
    return transcription


async def transcribe_segments(
    file_path: str, task_id: str, segment_size_mb: int = 24
) -> List[TranscriptSentence]:
    tasks_status[task_id].append(f"Splitting audio into segments")
    tasks = []
    output_files = split_wav_file(file_path, segment_size_mb)
    tasks_status[task_id].complete(f"Splitting audio into segments")
    tasks = []
    cumulative_duration = 0.0
    tasks_status[task_id].append(
        f"Concurrently transcribing all {len(output_files)} segments"
    )
    for idx, output_file in enumerate(output_files):
        audio_segment = AudioSegment.from_wav(output_file)
        tasks.append(
            asyncio.create_task(
                get_transcription_async(output_file, cumulative_duration, task_id, idx)
            )
        )
        cumulative_duration += len(audio_segment) / 1000.0
    transcriptions = await asyncio.gather(*tasks)
    tasks_status[task_id].complete(
        f"Concurrently transcribing all {len(output_files)} segments"
    )
    tasks_status[task_id].delete_prefix("Transcribing segment ")
    combined_transcription = [
        sentence for transcription in transcriptions for sentence in transcription
    ]
    return combined_transcription


async def process_video(video_file_path: str, task_id: str) -> List[TranscriptSentence]:
    transcription_file = encode_filename(video_file_path)

    if os.path.exists(f"transcriptions/{transcription_file}.dill"):
        tasks_status[task_id].append("Loading transcription from cache")
        transcription = dill.load(
            open(f"transcriptions/{transcription_file}.dill", "rb")
        )
        tasks_status[task_id].complete("Loading transcription from cache")
    else:
        tasks_status[task_id].append("Extracting audio from video")
        audio_file_path = f"audio_segments/{transcription_file}.wav"
        tasks_status[task_id].complete("Extracting audio from video")
        await extract_audio_from_video(video_file_path, audio_file_path)
        transcription = await transcribe_segments(audio_file_path, task_id)
        dill.dump(
            transcription, open(f"transcriptions/{transcription_file}.dill", "wb")
        )

    return transcription


if __name__ == "__main__":
    video_file_path = "recording.mp4"
    transcription = asyncio.run(process_video(video_file_path), "tmp")
    print(transcription)
