import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List

import dill
from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from frame_extraction import (
    Frame,
    TranscriptSentence,
    generate_context,
    generate_frames,
)
from shared_state import TaskStatus, tasks_status
from transcription import encode_filename

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2024-02-15-preview",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@retry(
    wait=wait_random_exponential(min=1, max=120),
    stop=stop_after_attempt(12),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
async def call_openai(
    messages: List[ChatCompletionMessageParam | Frame],
    tools: List[ChatCompletionToolParam] = None,
    client=client,
) -> str:
    messages = [
        message.to_openai_message() if isinstance(message, Frame) else message
        for message in messages
    ]
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        max_tokens=4096,
    )
    if not tools:
        return response.choices[0].message.content
    response_message = response.choices[0].message if response.choices else None
    tool_calls = response_message.tool_calls
    return response_message, tool_calls


async def process_subtask(
    context: List[TranscriptSentence | Frame],
    description: str,
    subtask: str,
    task_id: str,
    idx: int,
) -> List[str]:
    messages = []
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Given the following context:",
        )
    )
    tasks_status[task_id].append(
        f"Processing subtask: {idx+1} with objective: {subtask}"
    )
    messages.extend(
        [obj.to_openai_message() if isinstance(obj, Frame) else obj for obj in context]
    )
    messages.extend(
        [
            ChatCompletionUserMessageParam(
                role="user",
                content=f'I need to accomplish this: "{description}". To serve that end, the subtask you need to solve is: "{subtask}".',
            ),
            ChatCompletionSystemMessageParam(
                role="system",
                content='Place screenshots in the to make the result clearer. Use the format ![<image title>](<timestamp>.jpg) to represent Frames present in the context. Ensure that timestamp of the screenshot is present in the context, only then you can use it in the output. eg "![Terminal Window with YAML](1234.jpg)" for the frame at timestamp 1234 seconds',
            ),
        ]
    )
    response = await call_openai(messages)
    tasks_status[task_id].complete(
        f"Processing subtask: {idx} with objective: {subtask}"
    )
    return response


def prepare_frames(response: str, video_path: str) -> None:
    timestamps = [
        match for match in re.findall(r"!\[.*?\]\((\d+(\.\d+)?)\.jpg\)", response)
    ]
    generate_frames(video_path, timestamps)


async def process_file(
    video_file_path: str, description: str, task_id: str
) -> List[str]:
    encoded_video_file = encode_filename(video_file_path)

    if os.path.exists(f"context/{encoded_video_file}.dill"):
        context = dill.load(open(f"context/{encoded_video_file}.dill", "rb"))
    else:
        context = await generate_context(video_file_path)
        dill.dump(context, open(f"context/{encoded_video_file}.dill", "wb"))
    tasks_status[task_id].append("Breaking down objective into subtasks")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "process_subtask",
                "description": "Sends a subtask to my assistant who will generate the response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subtask": {
                            "type": "string",
                            "description": "The subtask to solve.",
                        }
                    },
                    "required": ["subtask"],
                },
            },
        }
    ]

    messages = [
        ChatCompletionUserMessageParam(
            role="user",
            content=f"This is my goal - {description}.\n The context for performing this task is as follows",
        )
    ]
    messages.extend(context)

    messages.append(
        ChatCompletionSystemMessageParam(
            role="system",
            content="To perform this task, break it down into subtasks and provide detailed instructions for each subtask. Make the subtasks as detailed and nuanced as possible so that the topic is covered in thorough detail. More subtasks is better than larger subtasks.",
        )
    )
    print("Calling openai")
    response, tool_calls = await call_openai(messages, tools)
    messages.append(response)
    tasks_status[task_id].complete("Breaking down objective into subtasks")
    if tool_calls:
        subtasks = [
            asyncio.create_task(
                process_subtask(
                    context=context,
                    description=description,
                    subtask=json.loads(tool_call.function.arguments)["subtask"],
                    task_id=task_id,
                    idx=idx,
                )
            )
            for idx, tool_call in enumerate(tool_calls)
        ]
        tasks_status[task_id].append(
            f"Processing {len(subtasks)} subtasks concurrently"
        )
        tool_responses = await asyncio.gather(*subtasks)
        tasks_status[task_id].complete(
            f"Processing {len(subtasks)} subtasks concurrently"
        )
        for tool_call, tool_response in zip(tool_calls, tool_responses):
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_response,
                }
            )
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content='Given the above tool responses, stitch it all together in as much detail as humanly possible. Make sure to be as thorough as possible without leaving behind any details. Make it as long as possible. Use markdown formatting. Use the format ![<image title>](<timestamp>.jpg) to represent Frames present in the context. Ensure that timestamp of the screenshot is present in the context, only then you can use it in the output. eg "![Terminal Window with YAML](1234.jpg)" for the frame at timestamp 1234 seconds.',
        )
    )
    tasks_status[task_id].append("Stitching all subtasks together")
    response = await call_openai(messages)
    messages.append(response)
    tasks_status[task_id].complete("Stitching all subtasks together")
    tasks_status[task_id].append("Generating frames for the final output")
    prepare_frames(response, video_file_path)
    tasks_status[task_id].complete("Generating frames for the final output")
    base_path, _ = os.path.splitext(os.path.basename(encoded_video_file))
    with open(f"frames/{base_path}.md", "w") as f:
        f.write(response)
    with open(f"conversation/{base_path}.dill", "wb") as f:
        dill.dump(messages, f)
    tasks_status[task_id].finished(True)
    return response


if __name__ == "__main__":
    video_file_path = "recording.mp4"  # Replace with the actual path to your video
    # description = "Generate a detailed runbook from the following video"
    # response = asyncio.run(
    #     process_file(video_file_path, description, "example_task_id")
    # )
    # print(response)

    encoded_video_file = encode_filename(video_file_path)
    base_path, _ = os.path.splitext(os.path.basename(encoded_video_file))
    with open(f"frames/{base_path}.md", "r") as f:
        response = f.read()
    prepare_frames(response, video_file_path)
