import asyncio
import base64
import json
import os
import re

import aiohttp
import markdown2
from dotenv import load_dotenv
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

load_dotenv()


class GDrive:
    def __init__(self):
        self.secret_file = os.getenv("GOOGLE_CREDENTIALS_FILENAME")
        self.drive = self.authenticate_drive()

    def authenticate_drive(self):
        settings = {
            "client_config_backend": "service",
            "service_config": {
                "client_json_file_path": self.secret_file,
            },
        }
        gauth = GoogleAuth(settings=settings)
        gauth.ServiceAuth()
        return GoogleDrive(gauth)

    def embed_images_in_html(self, html):
        img_tags = re.findall(r'<img [^>]*src="([^"]+)"', html)

        for img_tag in img_tags:
            img_path = os.path.join("frames", img_tag)
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

    def convert_md_to_html_with_embedded_images(self, md_content):
        html_content = markdown2.markdown(md_content)
        html_with_images = self.embed_images_in_html(html_content)
        return html_with_images

    def create_and_upload_document(self, content):
        htmldoc = self.convert_md_to_html_with_embedded_images(content)
        gdoc = self.drive.CreateFile(
            {
                "title": "My Shiny New Google Doc from Python!",
                "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            }
        )
        with open("tmp.html", "w") as f:
            f.write(htmldoc)
        gdoc.SetContentString(htmldoc)
        gdoc.Upload()
        gdoc.InsertPermission({"type": "anyone", "role": "writer", "value": "anyone"})
        return gdoc["alternateLink"]


if __name__ == "__main__":
    drive = GDrive()
    content = open(
        "frames/c07dd6ed396b2b5a0e62ebbf040d2547d555f72635fffb5d18db4d99706d33e1.md",
        "r",
    ).read()
    asyncio.run(drive.create_and_upload_document(content))
