import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

def create_and_save__faiss_index():
    """
    Loads documents and embeddings, creates a robust FAISS index
    that preserves metadata, and saves it locally
    """
    print("--- Initializing FAISS Index creation ---")

    # 1. Loads the data
    print("\nPhase 1: Loading 'corpus_vectors.pkl'...")
    try:
        with open("corpus_vectors.pkl", "rb") as f:
            data = pickle.load(f)
            chunks_documents = data["chunks"]
            embeddings = data["embeddings"]
    except FileNotFoundError:
        print("Error: 'corpus_vectors.pkl' file not found.")
        return
    
    print(f"{len(chunks_documents)} chunks loaded (as Document objects).")

    # 2. Prepares the data for the FAISS
    # Extracts the texts and metadata from Documentos list
    texts = [doc.page_content for doc in chunks_documents]
    metadata = [doc.metadata for doc in chunks_documents]

    # 3. Creates FAISS index the right way
    print("\nPhase 3: Creating FAISS index...\n")
    
    # Takes the mbeddings dimensions (ex: 768)
    embedding_dim = embeddings.shape[1] 
    
    # Initializes the raw FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Adds the embedding vectors to the index
    index.add(embeddings)
    
    # Creates the docstore and the ID mapping
    docstore = InMemoryDocstore({i: doc for i, doc in enumerate(chunks_documents)})
    index_to_docstore_id = {i: i for i in range(len(chunks_documents))}
    
    # Initializes the embedding model for the LangChain
    embedding_model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Envolves everything into FAISS object from LangChain
    faiss_index = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    print("FAISS index successfully creates, with preserved metadata.\n")

    # 4. Saves the FAISS index locally
    INDEX_FOLDER_NAME = "faiss_index"
    print(f"\nPhase 4: Saving index to the '{INDEX_FOLDER_NAME}' folder...\n")
    faiss_index.save_local(INDEX_FOLDER_NAME)
    print("Index successfully saved!\n!")

    return INDEX_FOLDER_NAME

# A função de teste continua a mesma do script anterior...
def test_faiss_index(index_folder_name: str):
    """
    Loads the saved index and performs a similarity search as test.
    """
    print("\n--- Testing saved FAISS Index ---")
    
    embedding_model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Loading index to the '{index_folder_name}' folder...")
    loaded_faiss_index = FAISS.load_local(
        index_folder_name,
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )
    print("Index successfully loaded.\n")

    query = "What are the main conclusions about semantic chunking?"
    print(f"\nPerforming similarity search for the query: '{query}'\n")
    
    results = loaded_faiss_index.similarity_search(query, k=2)
    
    print("\nQuery results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content: {doc.page_content[:500]}...")

if __name__ == "__main__":
    folder_name = create_and_save__faiss_index()
    if folder_name:
        test_faiss_index(folder_name)