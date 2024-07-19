import asyncio
import base64
import json
import os
import re
from dataclasses import dataclass
from typing import List

import dill
from openai import AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from tenacity import retry, stop_after_attempt, wait_random_exponential

from frame_extraction import (
    Frame,
    TranscriptSentence,
    generate_context,
    generate_frames,
)

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2024-02-15-preview",
)


# @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(12))
async def call_openai(
    messages: List[ChatCompletionMessageParam | Frame],
    tools: List[ChatCompletionToolParam] = None,
    client=client,
) -> str:
    messages = [
        message.to_openai_message() if isinstance(message, Frame) else message
        for message in messages
    ]
    response = client.chat.completions.create(
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
    context: List[TranscriptSentence | Frame], description: str, subtask: str
) -> List[str]:
    messages = []
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Given the following context:",
        )
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
                content="Place screenshots in the to make the result clearer. Use the format ![image](<timestamp: int>.png) to represent Frames present in the context. Ensure that timestamp of the screenshot is present in the context, only then you can use it in the output.",
            ),
        ]
    )
    response = await call_openai(messages)
    print("done processing subtask")
    return response


def prepare_frames(response: str, video_path: str) -> None:
    timestamps = [
        int(match) for match in re.findall(r"!\[image\]\((\d+)\.png\)", response)
    ]
    generate_frames(video_path, timestamps)


async def process_file(video_file_path: str, description: str) -> List[str]:
    # context = await generate_context(video_file_path)
    with open("combined_context.dill", "rb") as f:
        context = dill.load(f)

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
    response, tool_calls = await call_openai(messages, tools)
    messages.append(response)
    if tool_calls:
        available_functions = {
            "process_subtask": process_subtask,
        }
        subtasks = [
            available_functions[tool_call.function.name](
                context=context,
                description=description,
                subtask=json.loads(tool_call.function.arguments)["subtask"],
            )
            for tool_call in tool_calls
        ]
        print(f"starting to process {len(subtasks)} subtasks")
        tool_responses = await asyncio.gather(*subtasks)
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
            content="Given the above tool responses, stitch it all together in as much detail as humanly possible. Make sure to be as thorough as possible without leaving behind any details. Make it as long as possible. Use markdown formatting. Use the format ![image](<timestamp: int>.png) to represent Frames present in the context. Ensure that timestamp of the screenshot is present in the context, only then you can use it in the output.",
        )
    )

    response = await call_openai(messages)
    prepare_frames(response, video_file_path)
    return response


# if __name__ == "__main__":

# video_path = "recording.mp4"
# description = "Generate a detailed runbook with the contents of this entire meeting"
# captions = asyncio.run(process_file(video_path, description=description))
# with open("captions.dill", "wb") as f:
#     dill.dump(captions, f)

# if os.path.exists("captions.dill"):
#     with open("captions.dill", "rb") as f:
#         captions = dill.load(f)
#         with open("frames/output.md", "w") as f:
#             f.write(captions)
#         prepare_frames(captions, "recording.mp4")
