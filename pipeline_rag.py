from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from setup_llm import load_sabia_model, create_rag_prompt_template

# 1. Configuration and loading functions

def load_retriever():
    """
    Loads the FAISS index and initializes a retriever
    The retriever is responsable for the similarity search
    """
    print("Loading FAISS index and the embedding model...\n")

    # Defines the same embedding model used to create the index
    embedding_model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Loads locally saved FAISS index
    faiss_index = FAISS.load_local(
        "faiss_index",
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )

    print("Index successfully loaded.\n")

    # Conerts the index into a retriever, which can be used in the pipeline
    # k=8 means that it will retrieve the 4 most relevant chunks
    return faiss_index.as_retriever(search_kwargs={"k": 8})

def format_documents(docs):
    """
    Formats the list of retrieved documents into a single string to be inserted into the prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# 2. Main pipeline run

if __name__ == "__main__":
    print("--- Initializing RAG Pipeline ---\n")

    # Loads the needed components
    retriever = load_retriever()
    llm = load_sabia_model("models/sabia-7b.Q4_K_M.gguf")
    prompt = create_rag_prompt_template()

    # 3. Building chain using LCEL

    print("Building RAG chain with LCEL...\n")

    # CADEIA 1: Apenas para recuperar o contexto
    recuperation_chain = retriever | format_documents

    # CADEIA 2: A cadeia RAG completa
    rag_chain = (
        {"context": recuperation_chain, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chain explanation:
    # 1. {"context": ..., "question": ...}: Prepares the inputs for the prompt
    #    - "contenxt" is obtained passing the question to the the 'retriever' and then formating the output
    #    - "question" is the original querry, passing directly through 'RunnablePassthrough'
    # 2. | prompt: The data ("context" and "question") are inserted in the prompt template
    # 3. | llm: The formatted prompt is sent to the Sabiá model
    # 4. | StrOutputParser: The model result is converted into a simple string

    print("RAG chain successfully built.\n")

    # 4. Testing the Pipeline

    print("--- Testing the Pipeline with a Question ---\n")
    question = "De acordo com o artigo, qual foi o nível de correlação entre os julgamentos dos LLMs e os julgamentos humanos?"

    print(f"Question: {question}")

    # ETAPA DE DEPURAÇÃO: Invocamos a primeira cadeia para ver o contexto
    print("\n--- CONTEXTO RECUPERADO PELO FAISS ---")
    recuperated_context = recuperation_chain.invoke(question)
    print(recuperated_context)
    print("--------------------------------------")

    print("\nWait, Sabiá is generating the answer...\n")

    # Invokes the question chain
    # This is the moment when the RAG process happens
    answer = rag_chain.invoke(question)

    print("--- Answer generated ---\n")
    print(answer)