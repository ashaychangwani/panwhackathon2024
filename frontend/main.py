import difflib
import re
import time
from io import BytesIO  # Add this import

import requests
import streamlit as st
from fpdf import FPDF  # Add this import

st.set_page_config(layout="centered")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []


def main():
    st.title("SirSleepsALot")
    uploaded_file = st.file_uploader("Choose a file")
    objective = st.text_input("Enter your objective")

    col1, col2, col3, col4, _ = st.columns(5)  # Create two columns

    with col1:
        submit_button = st.button("Submit")
    with col2:
        display_button = st.button("Display")
    with col3:
        export_button = st.button("Export")
    with col4:
        download_button = st.button("Download PDF")

    with st.sidebar:
        messages = st.container(height=600)
        if prompt := st.chat_input("Say something"):
            messages.chat_message("user").write(prompt)
            response = get_chatbot_response(prompt, uploaded_file.name)
            st.session_state.conversation.append({"user": prompt, "system": response})
    for message in st.session_state.conversation:
        messages.chat_message("user").write(f"{message['user']}")
        messages.chat_message("system").write(f"{message['system']}")

    if submit_button:
        if uploaded_file is not None and objective:
            token = uploaded_file.name  # Use the filename as a token
            data = {"video_url": token, "objective": objective}
            response = requests.post(
                "http://localhost:8080/process", json=data, timeout=10
            )
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                st.success("Data successfully sent to the server!")
                status_placeholder = st.empty()
                while True:
                    status_response = requests.get(
                        f"http://localhost:8080/status/{task_id}"
                    )
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status_list = status_data.get("status", [])
                        status_text = "\n".join(
                            f"- {'✅' if item['status'] == 'completed' else '🔄'} {item['description']}"
                            for item in status_list
                        )
                        status_placeholder.markdown(status_text)
                        if any(
                            item["description"] == "completed" for item in status_list
                        ):
                            st.success("Processing completed!")
                            break
                    else:
                        st.error("Failed to get status from the server.")
                        break
                    time.sleep(1)
            else:
                st.error("Failed to send data to the server.")
        else:
            st.warning("Please upload a file and enter an objective.")

    if display_button:
        if uploaded_file is not None:
            token = uploaded_file.name
            display_response = requests.get(f"http://localhost:8080/display/{token}")
            if display_response.status_code == 200:
                markdown_content = display_response.json()["text"]
                st_markdown(markdown_content)
            else:
                st.error(
                    "File has not been processed yet. Please wait for the processing to complete."
                )

    if export_button:
        if uploaded_file is not None:
            token = uploaded_file.name
            display_response = requests.get(f"http://localhost:8080/display/{token}")
            if display_response.status_code == 200:
                markdown_content = display_response.json()["text"]
                pdf_data = convert_md_to_html_with_embedded_images(markdown_content)
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name="output.pdf",
                    mime="application/pdf",
                )
            else:
                st.error(
                    "File has not been processed yet. Please wait for the processing to complete."
                )


def get_chatbot_response(user_input, token):
    conversation = {
        "messages": st.session_state.conversation,
        "text": user_input,
    }
    response = requests.post(
        f"http://localhost:8080/conversation/{token}", json=conversation, timeout=120
    )
    print(response.json())
    return response.json()["text"]


def st_markdown(markdown_string):
    parts = re.split(r"!\[(.*?)\]\((.*?)\)", markdown_string)
    for i, part in enumerate(parts):
        if i % 3 == 0:
            st.markdown(part)
        elif i % 3 == 1:
            title = part
        else:
            st.image(f"../frames/{part}")  # Add caption if you want -> , caption=title


def show_diffs(text1, text2):
    """
    Compare two texts word by word and return the differences with highlighting in Markdown format.

    Arguments:
    text1 -- The first text to compare
    text2 -- The second text to compare

    Returns:
    diff -- The differences between the two texts with highlighting in Markdown format
    """
    diff = difflib.Differ().compare(text1.split(), text2.split())
    diff_output = []

    for item in diff:
        prefix = item[0]
        word = item[2:]

        if prefix == "-":
            diff_output.append(f":red[{word.strip()}]")

        elif prefix == "+":
            diff_output.append(f":green[{word.strip()}]")

        elif prefix == "?":
            continue
        else:
            diff_output.append(word.strip())

    return " ".join(diff_output)


import base64
import os

import markdown2


def embed_images_in_html(html):
    img_tags = re.findall(r'<img [^>]*src="([^"]+)"', html)

    for img_tag in img_tags:
        img_path = os.path.join("../frames", img_tag)
        if os.path.exists(img_path):
            with open(img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")
                img_ext = os.path.splitext(img_path)[1][
                    1:
                ]  # Get the image file extension
                img_data_uri = f"data:image/{img_ext};base64,{img_data}"
                html = html.replace(img_tag, img_data_uri)
        else:
            print(f"Image not found: {img_path}")

    return html


def convert_md_to_html_with_embedded_images(md_content):
    html_content = markdown2.markdown(md_content)
    html_with_images = embed_images_in_html(html_content)
    with open("test.html", "w") as f:
        f.write(html_with_images)
    return html_with_images


if __name__ == "__main__":
    main()
