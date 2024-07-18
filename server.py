import moviepy.editor as mp
import dotenv
import os
import asyncio
import fastapi_poe as fp
from openai import OpenAI
from pydub import AudioSegment
import dill


dotenv.load_dotenv()

def extract_audio_from_video(video_file_path, output_audio_file_path):

    video = mp.VideoFileClip(video_file_path)
    
    # Extract the audio
    audio = video.audio
    
    # Write the audio to a .wav file
    audio.write_audiofile(output_audio_file_path)
    # extract_audio_from_video('trimmed-recording.mp4', 'audio.wav')



async def get_responses(api_key, messages):
    async for partial in fp.get_bot_response(messages=messages, bot_name="GPT-3.5-Turbo", api_key=api_key):
        print(partial)

def get_audio_segment_size(audio_segment: AudioSegment) -> int:
    return len(audio_segment.raw_data)

def split_wav_file(file_path, segment_size_mb=20):
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Convert segment size from MB to bytes
    segment_size_bytes = segment_size_mb * 1024 * 1024
    
    # Calculate the duration of the segment in milliseconds
    bytes_per_ms = len(audio.raw_data) / len(audio)
    segment_duration_ms = segment_size_bytes / bytes_per_ms
    
    # Initialize variables to track segments
    start = 0
    end = len(audio)
    segment_index = 0

    while start < end:
        # Determine the length of the segment in milliseconds
        segment_end = min(start + segment_duration_ms, end)
        segment = audio[start:segment_end]
        
        # Save the segment
        output_file = f"{os.path.splitext(file_path)[0]}_segment_{segment_index}.wav"
        segment.export(output_file, format="wav")
        print(f"Exported {output_file}")
        
        # Update the start position
        start = segment_end
        segment_index += 1


def get_transcription(audio_file_path):
    audio_file = open(audio_file_path, "rb")
    
    transcript = OpenAI().audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["segment"]
        )
    print(transcript)
    with open('transcript.dill', 'wb') as f:
        dill.dump(transcript, f)
    return transcript

def transcribe_segments(file_path, segment_size_mb=20):
    split_wav_file(file_path, segment_size_mb)
    
    base_file_name = os.path.splitext(file_path)[0]
    
    segment_index = 0
    
    while True:
        segment_file = f"{base_file_name}_segment_{segment_index}.wav"
        
        if not os.path.exists(segment_file):
            break
        
        transcript = get_transcription(segment_file)
        output_file = f"transcriptions/{base_file_name}_segment_{segment_index}_transcript.dill"
        with open(output_file, 'wb') as f:
            dill.dump(transcript, f)
        
        print(f"Saved transcript for {segment_file} to {output_file}")
        
        # Increment the segment index
        segment_index += 1

# transcribe_segments('audio.wav')
