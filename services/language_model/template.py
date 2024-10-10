import os


def get_chat_template():
    file_path = os.path.join(
        os.path.dirname(__file__), "resources", "simplified_chat_template.j2"
    )
    with open(file_path, "r") as file:
        chat_template = file.read()
    return chat_template
