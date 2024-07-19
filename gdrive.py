import json
import os

from dotenv import load_dotenv
from markdown import markdown
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

load_dotenv()


def document_template(content: dict):
    text = f"""# My Google Doc!

    ## Abstract

    This is some introductory text.
    We could have lorem ipsum'd it,
    but that would be too generic for my tastes.
    So instead, I have chosen to write freely from my mind.

    ## Section 1

    - First bullet point
    - Second bullet point

    <hr class="pb">

    ## Section 2

    {content["lorem ipsum"]}

    | hello | world |
    |:-----:|:-----:|
    |  1  |  a  |
    |  2  |  b  |

    <hr class="pb">

    """
    return text


secret_file = os.getenv("GOOGLE_CREDENTIALS_FILENAME")


settings = {
    "client_config_backend": "service",
    "service_config": {
        "client_json_file_path": secret_file,
    },
}


gauth = GoogleAuth(settings=settings)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)


content = {"lorem ipsum": "Lorem ipsum dolor sit amet."}
text = document_template(content)
htmldoc = markdown(text)

gdoc = drive.CreateFile(
    {
        "title": "My Shiny New Google Doc from Python!",
        "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)
gdoc.SetContentString(htmldoc)

gdoc.Upload()
gdoc.InsertPermission({"type": "anyone", "role": "writer", "value": "anyone"})
print(gdoc["alternateLink"])
