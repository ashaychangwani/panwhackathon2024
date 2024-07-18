import moviepy.editor as mp
import dotenv
import os
import asyncio
import fastapi_poe as fp


dotenv.load_dotenv()

def extract_audio_from_video(video_file_path, output_audio_file_path):

    video = mp.VideoFileClip(video_file_path)
    
    # Extract the audio
    audio = video.audio
    
    # Write the audio to a .wav file
    audio.write_audiofile(output_audio_file_path)
    # extract_audio_from_video('recording.mp4', 'audio.wav')



async def get_responses(api_key, messages):
    async for partial in fp.get_bot_response(messages=messages, bot_name="GPT-3.5-Turbo", api_key=api_key):
        print(partial)
 
api_key = os.getenv('POE_API_KEY')
message = fp.ProtocolMessage(role="user", content="Hello world")

# Run the event loop
# For Python 3.7 and newer
asyncio.run(get_responses(api_key, [message]))

# For Python 3.6 and older, you would typically do the following:
# loop = asyncio.get_event_loop()
# loop.run_until_complete(get_responses(api_key))
# loop.close()