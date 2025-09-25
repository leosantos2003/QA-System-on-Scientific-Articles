import os
from langchain.document_loaders import PyPDFLoader
from typing import List
from langchain.docstore.document import Document

# Defines de path to the directory that contains the PDF articles
# The script expects until 'corpus_pdfs' to be at the same level as itself
CORPUS_PATH = "corpus_pdfs"

def load_corpus(folder_path: str) -> List[Document]:
    """
    Loads all PDF documents from a folder and returns them as Langchain Document list of objects
    Each PDF page becomes a Document
    """
    # Checks if the specified path is a valid folder
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' was not found.\n")
        return []
    
    # Lists all files in the folder that end with '.pdf'
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    # If no PDFs are found, informs the user and returns an empty list
    if not pdf_files:
        print(f"No PDF files found in the folder 'folder_path'.\n")
        return []
    
    print(f"Found {len(pdf_files)} PDF files. Starting loading...\n")

    # List that will store all documents (pages) of all PDFs
    all_documents = []

    # Iterates over each PDF name found
    for file_name in pdf_files:
        # Assembles the full path for the file
        full_path = os.path.join(folder_path, file_name)
        print(f"\t- Loading 'file_name'...\n")

        # Uses PyPDFLoader to load the PDF content
        loader = PyPDFLoader(full_path)

        # .load() returns a list of Documents, one for each page
        pages = loader.load()

        # Adds the loaded pages to the main list
        all_documents.extend(pages)
        print(f"\t -> {len(pages)} pages loaded.\n")

    return all_documents

if __name__ == "__main__":
    print("--- Starting corpus data ingestion process ---\n")

    # Calls the main function to load the PDFs
    documents = load_corpus(CORPUS_PATH)

    # If loadiing succeded, shows a summary
    if documents:
        print("--- Data Ingestion Completed Successfully! ---\n")
        print(f"Total ('Document' objects) pages loaded: {len(documents)}")

        # Shows a sample from the first loaded document for verification
        print("Sample from first document loaded (first 200 characters):")
        print(documents[0].page_content[:200])

        # Shows the document origin (metadata)
        print(f"Source from the document sample: {documents[0].metadata['source']}\n")
        
