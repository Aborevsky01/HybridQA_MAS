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
import asyncio
from datasets import load_dataset
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
import pickle
from typing_extensions import List, Dict, Optional, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun



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

_retriever = None
_retriever_lock = asyncio.Lock()

class TableRetriever(BaseRetriever):
    """Retriever that returns tables with their IDs"""
    table_map: Dict
    retriever: BaseRetriever
    
    def _get_relevant_documents(self, query: str) -> List[Dict]:
        docs = self.retriever.get_relevant_documents(query)
        results = []
        for d in docs:
            table = self.table_map[d.page_content]
            results.append({
                "table_id": d.page_content,
                "table": table
            })
        return results
    
    async def aget_relevant_documents(self, query: str) -> List[Dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_relevant_documents, query)

async def _process_chunk(chunk):
    """Process a chunk of dataset rows"""
    local_tables = {}
    for example in chunk:
        table_id = example['table_id']
        if table_id not in local_tables:
            table = example['table']
            table['table_id'] = table_id
            local_tables[table_id] = table
    return local_tables

async def extract_unique_tables(dataset, chunk_size=1000):
    """Async processing to extract unique tables with IDs"""
    n = len(dataset)
    chunks = [dataset.select(range(i, min(i+chunk_size, n))) for i in range(0, n, chunk_size)]
    results = await asyncio.gather(*[_process_chunk(chunk) for chunk in chunks])
    
    unique_tables = {}
    for res in results:
        unique_tables.update(res)
    return unique_tables

async def build_retriever():
    """Build and cache the retriever"""
    global _retriever
    
    dataset = load_dataset("wenhu/hybrid_qa", split='train')
    unique_tables = await extract_unique_tables(dataset)
    
    documents = []
    for table_id, _ in unique_tables.items():
        documents.append(Document(
            page_content=table_id,
            #metadata={"table": table}
        ))
    
    embeddings = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
    vector_store.dump("./table_id_vectors")
    
    with open("table_map.pkl", "wb") as f:
        pickle.dump(unique_tables, f)
    
    _retriever = TableRetriever(table_map=unique_tables, retriever=vector_store.as_retriever(search_kwargs={"k": 1}))
    return _retriever

async def get_retriever_local():
    """Get or create the retriever instance"""
    global _retriever
    if _retriever is not None:
        return _retriever
    
    async with _retriever_lock:
        if _retriever is not None:
            return _retriever
        
        import os
        if os.path.exists("./table_id_vectors") and os.path.exists("./table_map.pkl"):
            embeddings = OpenAIEmbeddings()
            vector_store = InMemoryVectorStore.load("table_id_vectors", embeddings)
            with open("table_map.pkl", "rb") as f:
                table_map = pickle.load(f)
            _retriever = TableRetriever(table_map=table_map, retriever=vector_store.as_retriever(search_kwargs={"k": 1}))
        else:
            _retriever = await build_retriever()
    return _retriever