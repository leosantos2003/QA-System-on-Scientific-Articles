# Question-Answer System on Scientific Articles

## About

The system is a Question-Answer (QA) pipeline that uses the RAG (Retrieval-Augmented Generation) architecture to answer questions based on a corpus of scientific articles published by researchers at UFRGS.

Links of the articles:

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
    * The `load_corpus` function iterates over all `.pdf` files within the `corpus_pdfs` folder. Using the `pypdf` library, it reads each document page by page, extracts the textual content, and creates a LangChain Document object for each page. It appends the correct metadata to each Document, such as the source filename and page number.
3. ``:
4. ``:
5. ``:
6. ``:
7. ``:

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
