# Question-Answer System on Scientific Articles with Sabiá

## About

The system is a Question-Answer (QA) pipeline that uses the RAG (Retrieval-Augmented Generation) architecture, as well as the Brazilian language model "Sabiá", to answer the user's questions about a corpus of scientific articles published by researchers at UFRGS.

Links of the articles used:

* [Can we trust LLMs as relevance judges?](https://sol.sbc.org.br/index.php/sbbd/article/view/30724)
* [Evaluation of Question Answer Generation for Portuguese: Insights and Datasets](https://aclanthology.org/2024.findings-emnlp.306.pdf)
* [Portuguese word embeddings for the oil and gas industry: Development and evaluation](https://www.sciencedirect.com/science/article/abs/pii/S0166361520305819)
* [Assessing the Impact of OCR Errors in Information Retrieval](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_13)
* [Evaluating and mitigating the impact of OCR errors on information retrieval](https://link.springer.com/article/10.1007/s00799-023-00345-6)
* [Embeddings for Named Entity Recognition in Geoscience Portuguese Literature](https://aclanthology.org/2020.lrec-1.568.pdf)
* [An Efficient Approach for Semantic Relatedness Evaluation Based on Semantic Neighborhood](https://ieeexplore.ieee.org/document/8995302)

## System Arquiteture

The project implements a complete RAG pipeline, which is divided into two distinct phases.

### Phase 1: Indexing

At this phase, the knowledge base is built and prepared for consultation.

1. **Data Ingestion**: A corpus of scientific articles in PDF format is loaded. The text of each page is extracted and stored as an individual document, with source and page metadata preserved.
2. **Semantic Chunking**: Each document is divided into chunks of text. The approach used is inspired by research on semantic chunking, where text is segmented into sentences and grouped based on the cosine similarity of their embeddings, respecting the text's natural semantic boundaries.
3. **Embedding**: Each chunk of text is converted into a high-dimensional numeric vector using the `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` multilingual embedding model. This model was chosen to enable the system to process questions in Portuguese on an English corpus.
4. **Indexing**: Embedding vectors are stored in a FAISS (Facebook AI Similarity Search) index, a high-performance library that enables high-speed similarity searches. The final index is saved to disk for persistence.

### Phase 2: Answer Generation

This phase is executed in real time for each user question.

1. **User Query**: The user submits a question in natural language (Portuguese or English).
2. **Question Embedding**: The question is vectorized using the same multilingual model.
3. **Retrieval**: The FAISS index is queried to find the text chunks whose vectors are most semantically similar to the question vector.
4. **Prompt Construction**: The original question and retrieved chunks (the "context") are inserted into an prompt template, which instructs the language model to respond based solely on the information provided.
5. **Response Generation**: The final prompt is sent to the Sabiá-7B language model, which generates a factual and concise response in Portuguese.

## Project Modules

1. `data_ingestion.py`: Its only function is to load the raw data (PDF articles) and transform it into a format that the LangChain library can understand.
      * The `load_corpus` function iterates over all `.pdf` files within a `corpus_pdfs` folder. Using the `pypdf` library, it reads each document page by page, extracts the textual content, and creates a LangChain Document object for each page. It appends the correct metadata to each Document, such as the source filename and page number.
      * **Output**: A list of Document objects, where each object represents a single page from one of the articles in the corpus.

2. `chunking.py`: Process the uploaded documents, breaking the text of entire pages into smaller, more semantically meaningful chunks.
      * The script offers two strategies. The main one is `semantic_chunking`, which implements an advanced approach. For each Document, the function first splits the text into individual sentences using `nltk`. It then generates embeddings for each sentence and calculates the cosine similarity between adjacent sentences. Sequential sentences with high similarity are grouped into the same chunk. When the similarity drops below a threshold, a semantic boundary is identified, and a new chunk is started. This method ensures that the original metadata is preserved in each new chunk created.
      * **Output**: A refined list of Document objects, where each object now represents a semantically cohesive chunk.

3. `vectorization.py`: Transforms text chunks into numeric vectors (embeddings).
      * The script imports the chunking function, loads the multilingual embedding model (`paraphrase-multilingual-mpnet-base-v2`), and uses it to convert the text content of each chunk into a high-dimensional vector. The result of this process is saved for future use. **This part took a long time.**
      * **Output**: A `vectors_corpus.pkl` file, which contains a dictionary with two keys: "chunks" (the complete list of Document objects) and "embeddings" (the array of corresponding vectors).

4. `create_faiss_index.py`: Builds the vector search index from the embeddings generated in the previous step.
      * The script loads the `vetors_corpus.pkl` file. It then uses the faiss library to build a data structure optimized for similarity searching. The script then integrates this structure with LangChain's Docstore, which stores the textual content and metadata, ensuring that each vector in the index is mapped to its original Document.
      * **Output**: A folder named `faiss_index`, containing the `index.faiss` and `index.pkl` files.

5. `setup_llm.py`: Configures the components of the response generation phase: the Large Language Model (LLM) and the prompt template.
      * It ontains two main functions. `load_sabia_model` loads the **Sabiá-7B** model in GGUF format using the LlamaCpp library. It sets crucial parameters such as temperature=0 (for factual responses) and stop words (to control the end of generation). The `create_rag_prompt_template` function defines the structure of the prompt that will be sent to LLM, containing placeholders for {context} and {question} and clear instructions on how the model should behave.
      * **Output**: Functions ready for loading an LLM object and a PromptTemplate object.

6. `pipeline_rag.py`: Controls all the components to create and run the complete RAG pipeline.
      * This is the main script for the query phase. It imports and utilizes functions from the previous modules. First, it loads the FAISS index and transforms it into a retriever. Then, it loads the Sabiá model and the prompt. Using the LangChain Expression Language (LCEL), it constructs the RAG chain: the user's question is passed to the retriever, the retrieved documents are formatted and inserted into the prompt along with the question, and the final prompt is sent to the **Sabiá** LLM to generate the answer.
      * **Output**: The final answer to the user's question, printed to the console.
  
## Results

1. **Question 1**: De acordo com o artigo, qual foi o nível de correlação entre os julgamentos dos LLMs e os julgamentos humanos?
     * **Answer**: A correlação entre os julgamentos dos LLMs e os julgamentos humanos é 0,26.

2. **Question 2**: Quais foram os F1-scores para Reconhecimento de Entidade Nomeada no artigo 'Embeddings for Named Entity Recognition in Geoscience'?
     * **Answer**: O F1-score do modelo baseado em Embeddings para Reconhecimento de Entidade Nomeada na Literatura de Geologia no artigo 'Embeddings for Named Entity Recognition in Geoscience' foi 77.

3. **Question 3**: De acordo com o artigo 'An Efficient Approach for Semantic Relatedness', qual é a ideia principal por trás do conceito de vizinhança semântica?
     * **Answer**: <empty>

4. **Question 4**: Como o impacto dos erros de OCR na recuperação de informação foi medido no artigo 'Assessing the Impact'?
     * **Answer**: Como o impacto dos erros de OCR na recuperação de informação foi medido no artigo 'Assessing the Impact'?

5. **Question 5**: Qual o melhor software comercial para OCR?
     * **Answer**: <empty>

## Analysis

1. **Question 1 and 2**: The system demonstrated satisfatory performance in factual questions, which require the location and extraction of specific data.

2. **Question 3 and 4**: For more open-ended questions, which require the synthesis of concepts or methodologies, the system encountered difficulties. For Question 3, the model returned an empty answer. For Question 4, however, the model returned the question itself, an erratic LLM behavior known as **"parroting"**, where the model repeats the question instead of refusing to answer.
     * Analyzing the retrieved context revealed that the vector similarity retriever, while effective at finding topically related passages (summaries, introductions), failed to identify the paragraphs with the highest information density, which contained definitions and detailed explanations. This is **a well-known challenge in Information Retrieval** and highlights the need for more advanced retrieval techniques.

4. **Question 5**: The system proved itself robust by refusing to answer a question whose answers were not contained in the text, strictly sticking to the prompt's instructions. Therefore, the model returned an empty answer.

## How to run the project

Prerequisites:

* C++ build tools ([Microsoft C++ Build Tools on Windows](https://visualstudio.microsoft.com/downloads/?q=build+tools) or build-essential on Linux).

1. Clone the repository:

```console
git clone https://github.com/leosantos2003/Sabia-QA-System-on-Scientific-Articles
```

2. Create and activate a virtual environment:

```console
# Linux
python3 -m venv .venv
source ven/bin/activate

# Windows
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the dependencies:

```console
pip install -r requirements.txt
```

4. Download the LLM:
     * Create a `models` folder at the root of the project.
     * Dowload the GGUF model `sabia-7b.Q4_K_M.gguf` from the [repository](https://huggingface.co/TheBloke/sabia-7B-GGUF) and save it in `models`.

5. Build the vectorial index:
     * Create a `corpus_pdfs` folder at the root of the project.
     * Add the scientific articles in PDF format in `corpus_pdfs`
     * **Note that you can add instead any texts of your choosing in PDF format, to see and test different results.**
     * Run the indexing scripts in order:
          * ```console
            # 1.
            # This one may take a while...
            python vectorization.py
            # 2.
            python create_faiss_index.py
            ```

6. Run the QA pipeline:
     * **Note that you can change the `question` variable in `pipeline_rag.py` to any question you'd like.**
     * Run the script:
          * ```console
            python pipeline_rag.py
            ```

7. Excerpt of an expected terminal response:

```console
[...]
--- Testing the Pipeline with a Question ---

Question: Quais foram os F1-scores para Reconhecimento de Entidade Nomeada no artigo 'Embeddings for Named Entity Recognition in Geoscience'?

[...]

Wait, Sabiá is generating the answer...

--- Answer generated! ---

Answer: O F1-score do modelo baseado em Embeddings para Reconhecimento de Entidade Nomeada na Literatura de Geologia no artigo 'Embeddings for Named Entity Recognition in Geoscience' foi 77.
------------------------
[...]
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
