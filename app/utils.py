import os
import pandas as pd
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
from langchain_core.documents import Document
import json
import re

def process_func(x):
    """Extract Wikipedia URLs from table cells"""
    try:
        return 'https://en.wikipedia.org' + x['urls'][0]['url']
    except (TypeError, KeyError, IndexError):
        return None

def prepare_documents(urls):
    """Create documents from Wikipedia URLs"""
    documents = []
    for url in tqdm(urls, desc="Loading documents"):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            # parallel
            for doc in docs:
                doc.page_content = re.sub(r'\n+', '\n', doc.page_content)
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=200, chunk_overlap=40
            )
            documents.extend(text_splitter.split_documents(docs))
        except Exception as e:
            print(f"Error loading {url}: {str(e)}")
    return documents


def get_retriever_tool(table_id: str, df: pd.DataFrame):
    """Get persistent retriever tool with caching"""
    chroma_db_path = "./app/chroma_data"
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    
    collection_name = f"wiki_rag_{table_id.replace('-', '_')[:50]}"
    docs_key = f"{collection_name}_docs"
    docs_path = os.path.join(chroma_db_path, f"{docs_key}.json")
    
    existing_collections = [col.name for col in chroma_client.list_collections()]
    
    if collection_name in existing_collections:
        collection = chroma_client.get_or_create_collection(
            name= collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200
                }
            }
        )

        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(),
        )
        
        if os.path.exists(docs_path):
            with open(docs_path, "r") as f:
                try:
                    doc_dicts = json.load(f)
                    documents = [
                        Document(page_content=doc["page_content"], metadata=doc["metadata"])
                        for doc in doc_dicts
                    ]
                except Exception as e:
                    print(f"Error loading documents from {docs_path}: {str(e)}")
    else:
        urls = df.map(process_func).stack().dropna().unique().tolist()
        if not urls:
            return None
            
        documents = prepare_documents(urls)
        if not documents:
            return None

        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(),
        )
        
        #docs_content = [doc.page_content for doc in documents]
        #metadata = [doc.metadata for doc in documents]
        ids = [str(i) for i in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=ids)
        
        with open(docs_path, "w") as f:
            json.dump([doc.dict() for doc in documents], f)
    
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    if documents:
        bm25_retriever = BM25Retriever.from_documents(
            documents=documents, 
            k=3
        )
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.4, 0.6]
        )
    else:
        retriever = chroma_retriever

    return create_retriever_tool(
        retriever,
        "retrieve_wikipedia_passages",
        "Searches Wikipedia passages for additional context"
    )