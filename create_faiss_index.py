import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_and_save_faiss_index():
    """
    Loads the chunks and embeddings, creates a FAISS index, and saves it locally
    """
    print("--- Starting FAISS Index Creation ---\n")

    # 1. Loads the embeddings ands chunks saved
    print("Phase 1: Loading 'corpus_vectors.pkl'...\n")
    try:
        with open("vetores_corpus.pkl", "rb") as f:
            data = pickle.load(f)
            text_from_chunks = data["chunks"]
            embeddings = data["embeddings"]
    except FileNotFoundError:
        print("Error: 'vetores_corpus.pkl' file was not found.")
        return
    
    print(f"{len(textos_dos_chunks)} chunks and {len(embeddings)} embeddings loaded.\n")

    # 2. Initialize the embedding model
    # Langchain needs the model in order to know how to process incoming questions
    print("\nPhase 2: Initializing the embedding model for LangChain...")
    embedding_model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("Embedding model initialized.\n")

    # 3. Creates the FAISS index from the embeddings and texts
    # LangChain eases this process with the 'from_embeddings' method
    # It deals with the inicialization and addition of the vectors to the index
    print("\nPhase 3: Creating the FAISS index into memory...")
    faiss_index = FAISS.from_embeddings(
        text_embeddings=list(zip(text_from_chunks, embeddings)),
        embedding=embedding_function
    )
    print("FAISS index successfully created.\n")

    # 4. Saves the FAISS index locally
    # The index will be saved to the folder to be able to be fastly loaded later
    INDEX_FOLDER_NAME = "faiss_index"
    print(f"\nPhase 4: Saving the index to the '{INDEX_FOLDER_NAME}' folder...\n")
    faiss_index.save_local(INDEX_FOLDER_NAME)
    print("INdex successfully saved!\n")

    return INDEX_FOLDER_NAME

def test_faiss_index(index_folder_name: str):
    """
    Loads the saved index and performs a similarity search as test
    """
    print("--- Testing saved FAISS Index ---\n")

    # Inicializes the same embedding model used in the index creation
    embedding_model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Loads the index to disk
    print(f"Loading index to the '{index_folder_name}' folder...")
    loaded_faiss_index = FAISS.load_local(
        index_folder_name,
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )
    print("Index successfully saved.\n")

    # performs a similarity search
    query = "What are the main conclusions about semantic chunking?"
    print(f"\nPerforming similarity search for the query: '{query}'\n")

    # 'similarity_search' returns the most relevant documents
    results = loaded_faiss_index.similarity_search(query, k=2) # Gets the 2 most relevant chunks

    print("Query results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content: {doc.page_content[:500]}...")

if __name__ == "__main__":
    folder_name = create_and_save_faiss_index()
    if folder_name:
        test_faiss_index(folder_name)