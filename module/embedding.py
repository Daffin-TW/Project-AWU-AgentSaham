#!/usr/bin/python3

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_chroma import Chroma
from transformers import pipeline
import pandas as pd
import os


class ChromaEmbeddings():
    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        model='Qwen/Qwen3-Embedding-0.6B',
        persist_directory='./chroma_db',
        collection_name='stock_news'
    ):
        # Construct attributes
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(model=model, device=1)
        
        # Initialize text splitter for large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load stock news data from CSV file"""

        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['date', 'title', 'url', 'text']
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')
        
        # Clean data
        df = df.dropna(subset=['date', 'title', 'text'])
        df['date'] = df['date'].astype(str)
        df['title'] = df['title'].astype(str)
        df['text'] = df['text'].astype(str)
        df['url'] = df['url'].astype(str)
        
        print(f'Loaded {len(df)} news articles')
        return df
    
    def prepare_documents(self, df: pd.DataFrame) -> list[Document]:
        """Convert DataFrame to LangChain Document objects"""
        
        documents = []
        
        for idx, row in df.iterrows():
            # Combine title and text for better context
            content = f'Title: {row['title']}\n\nContent: {row['text']}'
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    'date': row['date'],
                    'title': row['title'],
                    'url': row['url'],
                    'doc_id': idx,
                    'content_length': len(row['text'])
                }
            )
            documents.append(doc)
        
        return documents
    
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split large documents into smaller chunks"""

        split_docs = self.text_splitter.split_documents(documents)
        print(f'Split {len(documents)} documents into {len(split_docs)} chunks')
        return split_docs

    def create_embeddings(self, documents: list[Document]) -> Chroma:
        """Create embeddings and build ChromaDB vector store"""

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
    
    def build_embedding_system(
        self,
        csv_path: str,
        split_documents: bool = True,
        force_rebuild: bool = False
    ):
        """Complete pipeline to build the embedding system"""

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

    def load_existing_vectorstore(self) -> bool:
        """Load existing ChromaDB vector store if it exists"""

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
    # Initialize Pathing
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    ws_path = os.path.dirname(folder_path)
    data_dir = os.path.join(ws_path, 'dataset')
    db_path = os.path.join(ws_path, 'chroma_db')
    csv_path = os.path.join(data_dir, '2025-06-17_90p.csv')

    # Model building
    Embeddings = ChromaEmbeddings(
        chunk_size=1000,
        chunk_overlap=200,
        persist_directory=db_path,
        collection_name='stock_news'
    )
    Embeddings.build_embedding_system(
        csv_path, split_documents=True, force_rebuild=False
    )
    



    # print('Load Model...')
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    # sentences_1 = [
    #     "The dog plays in the garden",
    #     "The new movie is so great",
    #     "A woman watches TV",
    # ]

    # sentences_2 = [
    #     "The new movie is awesome",
    #     "The cat sits outside",
    #     "A man is playing guitar",
    # ]

    # print('Encode 1...')
    # embeddings1 = model.encode(sentences_1)

    # print('Encode 2...')
    # embeddings2 = model.encode(sentences_2)

    # # Compute cosine similarities
    # similarities = model.similarity(embeddings1, embeddings2)

    # # Output the pairs with their score
    # for idx_i, sentence1 in enumerate(sentences_1):
    #     print(sentence1)
    #     for idx_j, sentence2 in enumerate(sentences_2):
    #         print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
    
    # summarizer = pipeline("summarization", model="thivh/t5-base-indonesian-summarization-cased-finetuned-indosum")

    # ARTICLE = """Bank Nasional Komersial (NCB), bank terbesar di Arab Saudi berdasarkan aset, telah setuju untuk membeli pesaingnya, Samba Financial Group, seharga $15 miliar dalam merger perbankan terbesar tahun ini. NCB akan membayar 28,45 riyal ($7,58) untuk setiap saham Samba, menurut pernyataan pada Minggu, dengan nilai total sekitar 55,7 miliar riyal. NCB akan menawarkan 0,739 saham baru untuk setiap saham Samba, di batas bawah rasio 0,736-0, 787 yang ditetapkan bank-bank saat menandatangani perjanjian kerangka kerja awal pada Juni. Penawaran ini merupakan premi 3,5% dari harga penutupan Samba pada 8 Oktober sebesar 27,50 riyal dan sekitar 24% lebih tinggi dari level perdagangan saham sebelum pembicaraan diumumkan. Bloomberg News pertama kali melaporkan pembicaraan merger tersebut. Bank baru ini akan memiliki total aset lebih dari $220 miliar, menjadikannya pemberi pinjaman terbesar ketiga di kawasan Teluk. Kapitalisasi pasar entitas ini sebesar $46 miliar hampir menyamai Qatar National Bank QPSC, yang masih menjadi pemberi pinjaman terbesar di Timur Tengah dengan aset sekitar $268 miliar."""
    # result = summarizer(ARTICLE, do_sample=False)

    # print(result)


if __name__ == '__main__':
    main()


# ==================================================== #
# OLD

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.schema import Document
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# import pandas as pd
# import os

# class StockNewsEmbeddingSystem:
#     def __init__(
#             self,
#             openai_api_key: str = None,
#             persist_directory: str = './chroma_db',
#             collection_name: str = 'stock_news'
#         ):
#         # Set up OpenAI API key
#         load_dotenv()

#         if openai_api_key:
#             os.environ['OPENAI_API_KEY'] = openai_api_key
#         elif not os.getenv('OPENAI_API_KEY'):
#             raise ValueError('OpenAI API key is not provided')
        
#         self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        
#         # Initialize text splitter for large documents
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#         )
        
#         # ChromaDB settings
#         self.persist_directory = persist_directory
#         self.collection_name = collection_name
#         self.vectorstore = None
#         self.documents = []
        
#         # Create persist directory if it doesn't exist
#         os.makedirs(persist_directory, exist_ok=True)
    
#     # Load stock news data from CSV file
#     def load_csv_data(self, csv_path: str) -> pd.DataFrame:
#         df = pd.read_csv(csv_path)
        
#         # Validate required columns
#         required_columns = ['date', 'title', 'url', 'text']
#         missing_columns = [
#             col for col in required_columns if col not in df.columns
#         ]
        
#         if missing_columns:
#             raise ValueError(f'Missing required columns: {missing_columns}')
        
#         # Clean data
#         df = df.dropna(subset=['date', 'title', 'text'])
#         df['date'] = df['date'].astype(str)
#         df['title'] = df['title'].astype(str)
#         df['text'] = df['text'].astype(str)
#         df['url'] = df['url'].astype(str)
        
#         print(f'Loaded {len(df)} news articles')
#         return df
    
#     # Convert DataFrame to LangChain Document objects
#     def prepare_documents(self, df: pd.DataFrame) -> List[Document]:
#         documents = []
        
#         for idx, row in df.iterrows():
#             # Combine title and text for better context
#             content = f'title: {row['title']}\n\nContent: {row['text']}'
            
#             # Create document with metadata
#             doc = Document(
#                 page_content=content,
#                 metadata={
#                     'date': row['date'],
#                     'title': row['title'],
#                     'link': row['url'],
#                     'doc_id': idx,
#                     'content_length': len(row['text'])
#                 }
#             )
#             documents.append(doc)
        
#         return documents
    
#     # Split large documents into smaller chunks
#     def split_documents(self, documents: List[Document]) -> List[Document]:
#         split_docs = self.text_splitter.split_documents(documents)
#         print(f'Split {len(documents)} documents into {len(split_docs)} chunks')
#         return split_docs
    
#     # Create embeddings and build ChromaDB vector store
#     def create_embeddings(self, documents: List[Document]) -> Chroma:
#         print('Creating embeddings with ChromaDB...')
        
#         vectorstore = Chroma.from_documents(
#             documents=documents,
#             embedding=self.embeddings,
#             persist_directory=self.persist_directory,
#             collection_name=self.collection_name
#         )
        
#         print(
#             f'Created embeddings for {len(documents)} '
#             'document chunks in ChromaDB'
#         )
#         return vectorstore
    
#     # Complete pipeline to build the embedding system
#     def build_embedding_system(
#             self,
#             csv_path: str,
#             split_documents: bool = True,
#             force_rebuild: bool = False
#         ):
#         # Try to load existing collection first
#         if not force_rebuild and self.load_existing_vectorstore():
#             print('Using existing ChromaDB collection')
#             return
        
#         df = self.load_csv_data(csv_path)
#         documents = self.prepare_documents(df)
#         self.documents = documents
        
#         # Split documents if requested
#         if split_documents:
#             documents = self.split_documents(documents)
        
#         # Create embeddings and vector store
#         self.vectorstore = self.create_embeddings(documents)
        
#         print('Embedding system built successfully with ChromaDB!')
    
#     # Search for similar documents with metadata filtering
#     def similarity_search_metadata_filter(
#             self,
#             query: str,
#             k=5, 
#             metadata_filter: Dict[str, Any] = None
#         ) -> List[Dict[str, Any]]:

#         if not self.vectorstore:
#             raise ValueError('Embedding system not built yet.')
        
#         # Perform similarity search with metadata filter
#         if metadata_filter:
#             results = self.vectorstore.similarity_search_with_score(
#                 query, k=k, filter=metadata_filter
#             )
#         else:
#             results = self.vectorstore.similarity_search_with_score(query, k=k)
        
#         # Format results
#         formatted_results = []
#         for doc, score in results:
#             formatted_results.append({
#                 'content': doc.page_content,
#                 'metadata': doc.metadata,
#                 'similarity_score': score,
#                 'date': doc.metadata.get('date', 'N/A'),
#                 'title': doc.metadata.get('title', 'N/A'),
#                 'link': doc.metadata.get('link', 'N/A')
#             })
        
#         return formatted_results
    
#     # Basic similarity search
#     def similarity_search(self, query: str, k=5) -> List[Dict[str, Any]]:
#         return self.similarity_search_metadata_filter(query, k)
    
#     # Semantic search with optional score filtering and metadata filtering
#     def semantic_search(
#             self,
#             query: str,
#             k=5,
#             score_threshold: float = None,
#             metadata_filter: Dict[str, Any] = None
#         ) -> List[Dict[str, Any]]:
        
#         results = self.similarity_search_metadata_filter(
#             query, k, metadata_filter
#         )
        
#         if score_threshold:
#             results = [
#                 r for r in results if r['similarity_score'] <= score_threshold
#             ]
        
#         return results

#    # Get metadata for all documents in the collection 
#     def get_all_documents_metadata(self) -> List[Dict[str, Any]]:
#         if not self.vectorstore:
#             raise ValueError('Vector store not initialized')
        
#         # Get all documents from ChromaDB
#         all_docs = self.vectorstore.get()
        
#         metadata_list = []
#         for i, metadata in enumerate(all_docs['metadatas']):
#             metadata_list.append({
#                 'id': all_docs['ids'][i],
#                 'metadata': metadata
#             })
        
#         return metadata_list
    
#     # Add new documents to existing ChromaDB collection
#     def add_documents(self, documents: List[Document]):
#         if not self.vectorstore:
#             raise ValueError('Vector store not initialized.')
        
#         # Split documents if needed
#         split_docs = self.split_documents(documents)
        
#         # Add to existing collection
#         self.vectorstore.add_documents(split_docs)
#         self.vectorstore.persist()
        
#         print(f'Added {len(split_docs)} new document chunks to ChromaDB')

#     # Load existing ChromaDB vector store if it exists
#     def load_existing_vectorstore(self) -> bool:
#         try:
#             self.vectorstore = Chroma(
#                 persist_directory=self.persist_directory,
#                 embedding_function=self.embeddings,
#                 collection_name=self.collection_name
#             )
            
#             # Check if collection has data
#             collection_count = len(self.vectorstore.get()['ids'])
#             if collection_count > 0:
#                 print(
#                     'Loaded existing ChromaDB collection with '
#                     f'{collection_count} documents'
#                 )
#                 return True
#             else:
#                 print("ChromaDB collection exists but is empty")
#                 return False
                
#         except Exception as e:
#             print(f"Could not load existing ChromaDB collection: {e}")
#             return False


# def main():
#     # Initialize workspace pathing
#     file_path = os.path.abspath(__file__)
#     folder_path = os.path.dirname(file_path)
#     ws_path = os.path.dirname(folder_path)
#     db_path = os.path.join(ws_path, 'chroma_db')
#     data_dir = os.path.join(ws_path, 'dataset')

#     # Initialize the embedding system with ChromaDB
#     EmbeddingSystem = StockNewsEmbeddingSystem(
#         persist_directory=db_path,
#         collection_name='stock_news'
#     )
    
#     # Build the embedding system from CSV
#     csv_path = os.path.join(data_dir, '2025-06-16_30p.csv')
#     EmbeddingSystem.build_embedding_system(
#         csv_path, split_documents=True, force_rebuild=False
#     )
    
#     # Example searches
#     print('\n=== Example Searches ===')
#     query = 'Volalitas pasar saham'
#     search_result = EmbeddingSystem.semantic_search(query, k=3)
#     print('\nrelated news:')

#     for i, result in enumerate(search_result, 1):
#         print(f'{i}. {result['title']}')
#         print(f'   Similarity Score: {result['similarity_score']:.4f}')
#         print(f'   date: {result['date']}')
#         print(f'   url: {result['link']}')
#         print()
    
#     # Example: Add new documents to existing collection
#     # new_df = pd.read_csv('new_stock_news.csv')
#     # new_documents = embedding_system.prepare_documents(new_df)
#     # embedding_system.add_documents(new_documents)

# if __name__ == '__main__':
#     main()