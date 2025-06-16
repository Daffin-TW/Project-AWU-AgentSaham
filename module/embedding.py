#!/usr/bin/python3

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import os

class StockNewsEmbeddingSystem:
    def __init__(
            self,
            openai_api_key: str = None,
            persist_directory: str = './chroma_db',
            collection_name: str = 'stock_news'
        ):
        # Set up OpenAI API key
        load_dotenv()

        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        elif not os.getenv('OPENAI_API_KEY'):
            raise ValueError('OpenAI API key is not provided')
        
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        
        # Initialize text splitter for large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # ChromaDB settings
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
        self.documents = []
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
    
    # Load stock news data from CSV file
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['Date', 'Title', 'Link', 'Text']
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')
        
        # Clean data
        df = df.dropna(subset=['Date', 'Title', 'Text'])
        df['Date'] = df['Date'].astype(str)
        df['Title'] = df['Title'].astype(str)
        df['Text'] = df['Text'].astype(str)
        df['Link'] = df['Link'].astype(str)
        
        print(f'Loaded {len(df)} news articles')
        return df
    
    # Convert DataFrame to LangChain Document objects
    def prepare_documents(self, df: pd.DataFrame) -> List[Document]:
        documents = []
        
        for idx, row in df.iterrows():
            # Combine title and text for better context
            content = f'Title: {row['Title']}\n\nContent: {row['Text']}'
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    'date': row['Date'],
                    'title': row['Title'],
                    'link': row['Link'],
                    'doc_id': idx,
                    'content_length': len(row['Text'])
                }
            )
            documents.append(doc)
        
        return documents
    
    # Split large documents into smaller chunks
    def split_documents(self, documents: List[Document]) -> List[Document]:
        split_docs = self.text_splitter.split_documents(documents)
        print(f'Split {len(documents)} documents into {len(split_docs)} chunks')
        return split_docs
    
    # Create embeddings and build ChromaDB vector store
    def create_embeddings(self, documents: List[Document]) -> Chroma:
        print('Creating embeddings with ChromaDB...')
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(
            f'Created embeddings for {len(documents)} '
            'document chunks in ChromaDB'
        )
        return vectorstore
    
    # Complete pipeline to build the embedding system
    def build_embedding_system(
            self,
            csv_path: str,
            split_documents: bool = True,
            force_rebuild: bool = False
        ):
        # Try to load existing collection first
        if not force_rebuild and self.load_existing_vectorstore():
            print('Using existing ChromaDB collection')
            return
        
        df = self.load_csv_data(csv_path)
        documents = self.prepare_documents(df)
        self.documents = documents
        
        # Split documents if requested
        if split_documents:
            documents = self.split_documents(documents)
        
        # Create embeddings and vector store
        self.vectorstore = self.create_embeddings(documents)
        
        print('Embedding system built successfully with ChromaDB!')
    
    # Search for similar documents with metadata filtering
    def similarity_search_metadata_filter(
            self,
            query: str,
            k=5, 
            metadata_filter: Dict[str, Any] = None
        ) -> List[Dict[str, Any]]:

        if not self.vectorstore:
            raise ValueError('Embedding system not built yet.')
        
        # Perform similarity search with metadata filter
        if metadata_filter:
            results = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=metadata_filter
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': score,
                'date': doc.metadata.get('date', 'N/A'),
                'title': doc.metadata.get('title', 'N/A'),
                'link': doc.metadata.get('link', 'N/A')
            })
        
        return formatted_results
    
    # Basic similarity search
    def similarity_search(self, query: str, k=5) -> List[Dict[str, Any]]:
        return self.similarity_search_metadata_filter(query, k)
    
    # Semantic search with optional score filtering and metadata filtering
    def semantic_search(
            self,
            query: str,
            k=5,
            score_threshold: float = None,
            metadata_filter: Dict[str, Any] = None
        ) -> List[Dict[str, Any]]:
        
        results = self.similarity_search_metadata_filter(
            query, k, metadata_filter
        )
        
        if score_threshold:
            results = [
                r for r in results if r['similarity_score'] <= score_threshold
            ]
        
        return results

   # Get metadata for all documents in the collection 
    def get_all_documents_metadata(self) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            raise ValueError('Vector store not initialized')
        
        # Get all documents from ChromaDB
        all_docs = self.vectorstore.get()
        
        metadata_list = []
        for i, metadata in enumerate(all_docs['metadatas']):
            metadata_list.append({
                'id': all_docs['ids'][i],
                'metadata': metadata
            })
        
        return metadata_list
    
    # Add new documents to existing ChromaDB collection
    def add_documents(self, documents: List[Document]):
        if not self.vectorstore:
            raise ValueError('Vector store not initialized.')
        
        # Split documents if needed
        split_docs = self.split_documents(documents)
        
        # Add to existing collection
        self.vectorstore.add_documents(split_docs)
        self.vectorstore.persist()
        
        print(f'Added {len(split_docs)} new document chunks to ChromaDB')

    # Load existing ChromaDB vector store if it exists
    def load_existing_vectorstore(self) -> bool:
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Check if collection has data
            collection_count = len(self.vectorstore.get()['ids'])
            if collection_count > 0:
                print(
                    'Loaded existing ChromaDB collection with '
                    f'{collection_count} documents'
                )
                return True
            else:
                print("ChromaDB collection exists but is empty")
                return False
                
        except Exception as e:
            print(f"Could not load existing ChromaDB collection: {e}")
            return False


def main():
    # Initialize workspace pathing
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    ws_path = os.path.dirname(folder_path)
    db_path = os.path.join(ws_path, 'chroma_db')
    data_dir = os.path.join(ws_path, 'dataset')

    # Initialize the embedding system with ChromaDB
    EmbeddingSystem = StockNewsEmbeddingSystem(
        persist_directory=db_path,
        collection_name='stock_news'
    )
    
    # Build the embedding system from CSV
    csv_path = os.path.join(data_dir, '2025-06-16_30p.csv')
    EmbeddingSystem.build_embedding_system(
        csv_path, split_documents=True, force_rebuild=False
    )
    
    # Example searches
    print('\n=== Example Searches ===')
    query = 'Volalitas pasar saham'
    search_result = EmbeddingSystem.semantic_search(query, k=3)
    print('\nrelated news:')

    for i, result in enumerate(search_result, 1):
        print(f'{i}. {result['title']}')
        print(f'   Similarity Score: {result['similarity_score']:.4f}')
        print(f'   Date: {result['date']}')
        print(f'   Link: {result['link']}')
        print()
    
    # Example: Add new documents to existing collection
    # new_df = pd.read_csv('new_stock_news.csv')
    # new_documents = embedding_system.prepare_documents(new_df)
    # embedding_system.add_documents(new_documents)

if __name__ == '__main__':
    main()