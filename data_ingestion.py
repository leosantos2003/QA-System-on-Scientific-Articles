import os
from typing import List
from langchain.docstore.document import Document
import pypdf # Usaremos a biblioteca pypdf diretamente

CORPUS_PATH = "corpus_pdfs"

def carregar_corpus(caminho_da_pasta: str) -> List[Document]:
    """
    Carrega todos os documentos PDF de uma pasta, garantindo que cada página
    visual se torne um objeto Document separado com a numeração correta.
    """
    if not os.path.isdir(caminho_da_pasta):
        print(f"Erro: O diretório '{caminho_da_pasta}' não foi encontrado.")
        return []

    arquivos_pdf = [f for f in os.listdir(caminho_da_pasta) if f.endswith('.pdf')]
    
    if not arquivos_pdf:
        print(f"Nenhum arquivo PDF encontrado no diretório '{caminho_da_pasta}'.")
        return []

    print(f"Encontrados {len(arquivos_pdf)} arquivos PDF. Iniciando o carregamento...")

    lista_final_de_documentos = []

    for nome_arquivo in arquivos_pdf:
        caminho_completo = os.path.join(caminho_da_pasta, nome_arquivo)
        print(f"\t- Processando '{nome_arquivo}'...")
        
        try:
            # Abre o arquivo PDF
            with open(caminho_completo, 'rb') as f:
                leitor_pdf = pypdf.PdfReader(f)
                
                # Itera sobre cada página real do PDF
                for num_pagina, pagina in enumerate(leitor_pdf.pages):
                    # Extrai o texto da página
                    texto_pagina = pagina.extract_text()
                    
                    if texto_pagina: # Apenas adiciona se a página tiver texto
                        # Cria o metadado correto
                        metadata = {"source": caminho_completo, "page": num_pagina + 1}
                        
                        # Cria um objeto Document para a página e adiciona à lista final
                        novo_documento = Document(page_content=texto_pagina, metadata=metadata)
                        lista_final_de_documentos.append(novo_documento)
            
            print(f"\t  -> {len(leitor_pdf.pages)} páginas carregadas e numeradas corretamente.")
        
        except Exception as e:
            print(f"\t  -> Erro ao ler o arquivo {nome_arquivo}: {e}")

    return lista_final_de_documentos

if __name__ == "__main__":
    print("--- Testando o processo de ingestão de dados do corpus ---")
    documentos = carregar_corpus(CORPUS_PATH)
    if documentos:
        print("\n--- Ingestão de Dados Concluída com Sucesso! ---")
        print(f"Total de páginas (objetos 'Document') carregadas: {len(documentos)}")
        print("\nAmostra do primeiro documento carregado:")
        print(f"Fonte: {documentos[0].metadata['source']}, Página: {documentos[0].metadata['page']}")
        print(f"Conteúdo: {documentos[0].page_content[:200]}...")
