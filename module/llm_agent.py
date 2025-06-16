#!/usr/bin/python3

from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime
import json
import os

# LangChain imports
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI

# Import the embedding system from the previous file
from module import StockNewsEmbeddingSystem

class StockSharesAIAgent:
    def __init__(
            self, 
            openai_api_key: str = None,
            model_name: str = 'o4-mini-2025-04-16',
            # temperature: float = 0.7,
            max_tokens: int = 1000,
            embedding_system: StockNewsEmbeddingSystem = None,
            persist_directory: str = './chroma_db',
            collection_name: str = 'stock_news'
        ):
                
        # Set up OpenAI API key
        load_dotenv()

        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        elif not os.getenv('OPENAI_API_KEY'):
            raise ValueError('OpenAI API key must be provided')
        
        # Initialize ChatGPT model
        self.llm = ChatOpenAI(
            model_name=model_name,
            # temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Set up embedding system for RAG
        self.embedding_system = embedding_system
        if not self.embedding_system and StockNewsEmbeddingSystem:
            self.embedding_system = StockNewsEmbeddingSystem(
                openai_api_key=openai_api_key,
                persist_directory=persist_directory,
                collection_name=collection_name
            )

            self.embedding_system.load_existing_vectorstore()
        
        # Agent personality and instructions
        self.system_prompt = self.create_system_prompt()

        # Set up RAG chain
        self.rag_chain = None
        self.setup_rag_chain()
    
    # Create the system prompt for the AI agent
    def create_system_prompt(self) -> str:
        return """
            Anda adalah Agen AI Saham yang ahli dengan akses ke berita keuangan 
            real-time.

            Kemampuan Anda meliputi:
            - Memberikan wawasan tentang saham dan perusahaan individu
            - Menjelaskan konsep keuangan dan pergerakan pasar
            - Memberikan perspektif investasi (bukan saran keuangan)
            - Menerjemahkan dampak berita terhadap harga saham
            - Analisis teknis dan fundamental

            Pedoman:
            1. Selalu dasarkan tanggapan Anda pada konteks yang disediakan dari 
            artikel berita dan data
            2. Jelaskan dengan jelas perbedaan antara fakta dari sumber dan 
            analisis Anda
            3. Sertakan peringatan bahwa ini untuk tujuan informasional, 
            bukan nasihat keuangan
            4. Bersikap objektif dan sajikan berbagai sudut pandang saat relevan
            5. Sebutkan artikel berita atau poin data spesifik saat membuat 
            klaim
            6. Jika Anda tidak memiliki cukup informasi, mintalah klarifikasi 
            atau sarankan data apa yang berguna

            Ingat: Anda memiliki akses ke basis data komprehensif artikel berita 
            saham melalui RAG. Gunakan informasi ini untuk memberikan tanggapan 
            yang terinformasi dan kontekstual.
        """

    # Set up the RAG chain for retrieval-augmented generation
    def setup_rag_chain(self):
        if not self.embedding_system or not self.embedding_system.vectorstore:
            print(
                'Warning: No embedding system available. '
                + 'RAG functionality will be limited.'
            )
            return
        
        try:
            # Create retriever from the vector store
            retriever = self.embedding_system.vectorstore.as_retriever(
                search_kwargs={'k': 5}  # Retrieve top 5 most relevant documents
            )
            
            # Create context-aware retriever
            contextualize_q_system_prompt = """
                Berdasarkan riwayat obrolan dan pertanyaan terbaru pengguna yang 
                mungkin merujuk pada konteks dalam riwayat obrolan, buatlah 
                pertanyaan mandiri yang dapat dipahami tanpa perlu merujuk pada 
                riwayat obrolan. Jangan menjawab pertanyaan tersebut, cukup 
                reformulasikan jika diperlukan, dan jika tidak, kembalikan 
                pertanyaan tersebut apa adanya.
            """
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ('system', contextualize_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                self.llm, retriever, contextualize_q_prompt
            )
            
            # Create QA prompt
            qa_system_prompt = f"""{self.system_prompt}
                Gunakan potongan konteks yang telah diperoleh berikut ini untuk 
                menjawab pertanyaan. Jika Anda tidak mengetahui jawabannya 
                berdasarkan konteks, katakan bahwa Anda tidak memiliki 
                cukup informasi. Konteks: {{context}}
            """

            qa_prompt = ChatPromptTemplate.from_messages([
                ('system', qa_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ])
            
            # Create document chain
            question_answer_chain = create_stuff_documents_chain(self.llm,
                                                                 qa_prompt)
            
            # Create RAG chain
            self.rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )
            
            print('RAG chain initialized successfully!')
            
        except Exception as e:
            print(f'Error setting up RAG chain: {e}')
            self.rag_chain = None
    
    # Main chat interface for the AI agent
    def chat(
            self, user_input: str,
            chat_history, 
            use_rag: bool = True
        ) -> Dict[str, Any]:
        try:
            if use_rag and self.rag_chain:
                # Use RAG chain for enhanced responses
                response = self.rag_chain.invoke({
                    'input': user_input,
                    'chat_history': chat_history
                })
                
                answer = response['answer']
                source_documents = response.get('context', [])
                
                return {
                    'response': answer,
                    'sources': [doc.metadata for doc in source_documents],
                    'source_count': len(source_documents),
                    'used_rag': True,
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                # Direct LLM response without RAG
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=user_input)
                ]
                
                response = self.llm.invoke(messages)
                answer = response.content
                
                return {
                    'response': answer,
                    'sources': [],
                    'source_count': 0,
                    'used_rag': False,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'response': f'Sorry, I encountered an error: {str(e)}',
                'sources': [],
                'source_count': 0,
                'used_rag': False,
                'error': True,
                'timestamp': datetime.now().isoformat()
            }
    
    # Search for relevant news articles
    def search_relevant_news(
            self, query: str, k: int = 5
        ) -> List[Dict[str, Any]]:
        
        if not self.embedding_system:
            return []
        
        try:
            return self.embedding_system.similarity_search(query, k=k)
        except Exception as e:
            print(f'Error searching news: {e}')
            return []
    
    # Analyze a specific stock based on available news and data
    def analyze_stock(
            self, stock_symbol: str, analysis_type: str = 'umum'
        ) -> Dict[str, Any]:
        """analysis_type: umum/teknis/fundamental/sentimen_berita"""

        # Search for relevant news about the stock
        relevant_news = self.search_relevant_news(
            f'{stock_symbol} Berita saham dan laporan keuangan', k=10
        )
        
        # Create analysis prompt based on type
        analysis_prompts = {
            'umum': (
                f'Berikan analisis umum tentang saham {stock_symbol} '
                + 'berdasarkan berita terbaru dan data pasar.'
            ), 
            'teknis': (
                f'Lakukan analisis teknis tentang saham {stock_symbol} '
                + 'berdasarkan informasi yang tersedia.'
            ),
            'fundamental': (
                'Analisis aspek fundamental dari perusahaan dan '
                + f'saham {stock_symbol} perusahaan dan sahamnya.'
            ),
            'sentimen_berita': (
                'Analisis sentimen dan dampak berita terbaru terhadap saham '
                + f'{stock_symbol}.'
            )
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts['umum'])
        
        # Get AI analysis
        analysis_result = self.chat(prompt, use_rag=True)
        
        return {
            'stock_symbol': stock_symbol,
            'analysis_type': analysis_type,
            'analysis': analysis_result['response'],
            'relevant_news_count': len(relevant_news),
            'sources': analysis_result['sources'],
            'timestamp': datetime.now().isoformat()
        }
    
    # Get general market insights and trends
    def get_market_insights(
            self, topic: str = 'tren pasar secara umum'
        ) -> Dict[str, Any]:
        prompt = (
            'Tren pasar umum Berikan wawasan dan analisis tentang '
            + f'{topic} berdasarkan berita keuangan terbaru dan data pasar.'
        )
        
        result = self.chat(prompt, use_rag=True)
        
        return {
            'topic': topic,
            'insights': result['response'],
            'sources': result['sources'],
            'timestamp': datetime.now().isoformat()
        }
    
    # Explain financial concepts with real-world examples
    def explain_financial_concept(self, concept: str) -> Dict[str, Any]:
        prompt = (
            f"Jelaskan konsep keuangan '{concept}' dengan contoh "
            + "nyata dari peristiwa pasar dan berita terkini.."
        )
        
        result = self.chat(prompt, use_rag=True)
        
        return {
            'concept': concept,
            'explanation': result['response'],
            'sources': result['sources'],
            'timestamp': datetime.now().isoformat()
        }
    

def main():
    # Initialize workspace pathing
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    ws_path = os.path.dirname(folder_path)
    db_path = os.path.join(ws_path, 'chroma_db')

    # Initialize embedding system
    collection_name='stock_news'

    # Initialize agent
    Agent = StockSharesAIAgent(
        model_name='o4-mini-2025-04-16',
        persist_directory=db_path,
        collection_name=collection_name,
        max_tokens=5000
    )
    
    # Example interactions
    # result = Agent.chat('Saham terbaru Semen Indonesia')
    # print(result['response'])
    
    # analysis = Agent.analyze_stock('ISAT', 'umum')
    # print(analysis['analysis'])
    
    insights = Agent.get_market_insights('Mobil')
    print(insights['insights'])


# Main execution
if __name__ == '__main__':
    main()