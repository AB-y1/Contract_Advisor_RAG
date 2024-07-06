# from langchain.document_loaders import DirectoryLoader
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document 
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os 
import shutil
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

relative_path ="data"
absolute_path = os.path.join(os.getcwd(), relative_path)
print (f"absolute path = {absolute_path}")
CHROMA_PATH = "chroma2"
DATA_PATH = absolute_path


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.docx")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=600,
        length_function=len,
        add_start_index=True,
    )
    chunks = []
    for document in tqdm(documents, desc="Splitting documents"):
        chunks.extend(text_splitter.split_documents([document]))    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Check if the vector store already exists
    if os.path.exists(CHROMA_PATH):
        print(f"Vector store already exists at {CHROMA_PATH}. Skipping creation.")
        return
    else:
        os.makedirs(CHROMA_PATH, exist_ok=True)

        
        # Create a new DB from the documents.
        db = Chroma.from_documents(
            tqdm(chunks, desc="Embedding chunks"),
            OpenAIEmbeddings( 
                 model= "text-embedding-3-large",
                 openai_api_key= os.getenv("OPENAI_API_KEY")
            ),
            persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")



if __name__ == "__main__":
    main()