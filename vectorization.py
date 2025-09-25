import pickle
from sentence_transformers import SentenceTransformer
import nltk

from data_ingestion import load_corpus, CORPUS_PATH
from chunking import semantic_chunking

def generate_and_save_embeddings():
    """
    Main function that loads the chunks, generates the embeddings and save them to disk
    """
    print("--- Starting the Vectorization process ---\n")

    # 1. Loads the document chunks
    print("Phase 1: Loading document chunks...\n")
    # Ensures that the NLTK needed packages are available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK needed packages ('punkt', 'punkt_tab')")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    loaded_documents = load_corpus(CORPUS_PATH)

    if not loaded_documents:
        print("No document was loaded. Shutting down the script.\n")
        return
    
    # 2. Selects and loads the multilingual embedding model
    print("Phase 2: Loading the multilingual embedding model...\n")
    print("Model: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'\n")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    print("Model succesfully loaded.\n")

    # The semantic chunking strategy will be used
    similarity_threshold = 0.8
    chunks = semantic_chunking(loaded_documents, model, similarity_threshold)

    # Extracts only the textual content from chunks to generate embeddings
    text_from_chunks = [chunk.page_content for chunk in chunks]

    # 3. Converts chunks into text in vectorial embeddings
    print(f"Phase 3: Generating embeddings for {len(text_from_chunks)} chunks...\n")
    # The enconding process may take a while...
    embeddings = model.encode(
        text_from_chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print("Embeddings succesfully generated!\n")
    print("Embeddings matrix format: {embeddings.shape}")

    # 4. Saves the embeddings and texts from the chunks
    print("Phase 4: Saving embeddings and texts from chunks to disk...\n")

    # Saving the data into a binary file using pickle to preserve the structure
    with open("corpus_vectors.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

    print("'corpus_vectors.pkl' file successfully saved!\n")
    print("This file will be used as input to create the FAISS index.\n")

if __name__ == "__main__":
    generate_and_save_embeddings()