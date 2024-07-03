import  tiktoken
import os 
relative_path ="data/Raptor Contract.docx"
absolute_path = os.path.join(os.getcwd(), relative_path)
print(absolute_path)
file_path = absolute_path

def count_tokens_in_file(file_path):
    """
    Calculates the number of tokens in a file using the tiktoken library.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        int: The number of tokens in the file.
    """
    with open(file_path, 'r', encoding="latin-1") as file:
        text = file.read()
    
    # Useing the 'cl100k_base' encoding name to count the tokens for embedding model
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    return len(tokens)

num_tokens = count_tokens_in_file(file_path)
print(f"The file contains {num_tokens} tokens.")
