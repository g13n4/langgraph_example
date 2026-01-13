from langchain_core.tools import tool

@tool
def save_to_file(content:str, filename: str = "content.txt"):
    """
    Save the given content to a text file.

    Args:
        content (str): The content to save.
        filename (str): The name of the file to save the content to.
    """
    with open(filename, 'w') as file:
        file.write(content)
    return f"Content saved to {filename}"
