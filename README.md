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

At this stage, the knowledge base is built and prepared for consultation.

1. **Data Ingestion**: A corpus of scientific articles in PDF format is loaded. The text of each page is extracted and stored as an individual document, with source and page metadata preserved.
2. **Semantic Chunking**: Each document is divided into chunks of text. The approach used is inspired by research on semantic chunking, where text is segmented into sentences and grouped based on the cosine similarity of their embeddings, respecting the text's natural semantic boundaries.
3. **Embedding**: Each chunk of text is converted into a high-dimensional numeric vector using the `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` multilingual embedding model. This model was chosen to enable the system to process questions in Portuguese on an English corpus.
4. **Indexing**: Embedding vectors are stored in a FAISS (Facebook AI Similarity Search) index, a high-performance library that enables high-speed similarity searches. The final index is saved to disk for persistence.

### Phase 2: Answer Generation

1. fwgdvsc

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
