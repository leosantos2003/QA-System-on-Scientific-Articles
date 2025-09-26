from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

def load_sabia_model(model_path: str):
    """
    Loads a Sabiá quantized language model (GGUF) using LlamaCpp
    """
    print(f"Loading Sabiá model from: {model_path}\n")

    # Parameters for LlamaCpp
    # n_gpu_layers: Number of layers to be downloaded fot the GPU
    #               If one doesn't have a powerful GPU, set to 0
    # n_batch: Batch size for prompt processing (adjust according to your RAM)
    # n_ctx: Maximum context size (prompt + answer)
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=0,
        n_batch=512,
        n_ctx=4096,
        f16_kv=True, # Must be True for higher quality
        max_tokens=512,
        stop=["\n", "Pergunta:"],
        verbose=True, # Displays processing details
    )
    print("Sabiá model succesfully loaded.\n")
    return llm

def create_rag_prompt_template():
    """
    Creates and returns an optimized prompt template for the RAG task
    """
    print("Creating the RAG prompt template...\n")

    template_string = """### CONTEXTO
{context}

### PERGUNTA
{question}

### INSTRUÇÕES
1. Use apenas o contexto acima para responder à pergunta.
2. Responda com uma única frase concisa em português.

### RESPOSTA
"""

    prompt_template = PromptTemplate(
        template=template_string,
        input_variables=["context", "question"]
    )
    print("Prompt template successfully created.\n")
    return prompt_template

if __name__ == "__main__":
    # Path to the downloaded GGUF model
    gguf_model_path = "models/sabia-7b.Q4_K_M.gguf"

    # Tests the model loading
    llm_sabia = load_sabia_model(gguf_model_path)

    # Tests prompt creation
    rag_prompt = create_rag_prompt_template()

    print("LLM model and prompt template are ready to be used in the pipeline.\n")
    # Example of the prompt would be filled
    print("Formatted prompt example:\n")
    prompt_example = rag_prompt.format(
        context = "A UFRGS foi fundada em 1934.",
        question="Quando a UFRGS foi fundada?"
    )
    print(prompt_example)