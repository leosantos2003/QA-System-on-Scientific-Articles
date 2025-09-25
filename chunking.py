import nltk
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

from data_ingestion import load_corpus, CORPUS_PATH

# --- Strategy 1: Baseline ---

def chunking_baseline(documents: List[Document]) -> List[Document]:
    """
    Applies the standard chunking technique using 'RecursiveCharacterTextSplitter'
    """
    print("\n--- Running Strategy 1: Chunking Baseline ---\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Number of generated chunks: {len(chunks)}\n")
    print("Example of chunk (300 first characters):\n")
    print(chunks[10].page_content[:300])

# --- Strategy 2: Semantic Chunking ---

def semantic_chunking(documents: List[Document], model_embedding: SentenceTransformer, threshold: float) -> List[Document]:
    """
    Applies the semantic chunking strategy
    1. Splits the text of each document into sentences
    2. Generates embedding for each sentence
    3. Calculates the similarity between adjacent sentences
    4. Groups sequential sentences with similarity above a threshold
    """
    print("--- Running Strategy 2: Semantic Chunking ---\n")

    semantic_chunks = []

    for doc in documents:
        # 1. Splits the texts into sentences
        sentences = nltk.sent_tokenize(doc.page_content)

        if len(sentences) > 0:
            # 2. Generates the sentences embeddings
            embeddings = model_embedding.encode(sentences, convert_to_tensor=True, show_progress_bar=True)

            # 3. Calculates the similarity between adjacent sentences
            similarities = util.cos_sim(embeddings[:-1], embeddings[1:])

            current_chunk = sentences[0]

            for i in range(len(sentences) - 1):
                # 4. Groups based on the similarity threshold
                if similarities[i][i] >= threshold:
                    current_chunk += " " + sentences[i+1]
                else:
                    # If the similarity is low, ends the current chunk e start a new one
                    semantic_chunks.append(Document(page_content=current_chunk, metadata=doc.metadata))
                    current_chunk = sentences[i+1]

            # Adds the last chunk
            semantic_chunks.append(Document(page_content=current_chunk, metadata=doc.metadata))

    print(f"Number of generated chunks: {len(semantic_chunks)}\n")
    print("Example of chunk (300 first characters):")
    print(semantic_chunks[10].page_content[:300])
    return semantic_chunks

if __name__ == "__main__":
    # Loads the corpus documents using the function from data_ingestion.py
    loaded_documents = load_corpus(CORPUS_PATH)

    if loaded_documents:
        # --- Running Strategy 1 ---
        chunks_baseline = chunking_baseline(loaded_documents)

        # --- Running Strategy 2 ---
        # Loads an embedding model for the semantic chunking
        print("\nLoading embedding model for the semantic chunking...\n")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Threshold is an hyperparameter
        # Higher values create samller and more cohesive chunks
        similarity_threshold = 0.8

        semantic_chunks = semantic_chunking(loaded_documents, model, similarity_threshold)
