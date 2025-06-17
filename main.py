#!/usr/bin/python3

from streamlit import session_state as ss
from streamlit import secrets as sc
import streamlit as st
import os


# Initialize pathing
ws_path = os.getcwd()
data_dir = os.path.join(ws_path, 'dataset')
db_path = os.path.join(ws_path, 'chroma_db')

# Page configuration
st.set_page_config(
    page_title='AWU Agen AI Saham',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Initialize session state
if 'messages' not in ss:
    ss.messages = []

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .stButton > button {
        background-color: #2196f3;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .dataset-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration and dataset scraping
with st.sidebar:
    st.title('🛠️ Konfigurasi')
    
    # AI Model Settings
    st.subheader('Pengaturan AI')
    model_type = st.selectbox(
        'Pilih model AI',
        ['o4-mini-2025-04-16'],
        index=0
    )
    
    max_tokens = st.slider('Max Tokens', 100, 4000, 1000, 100)

# Main chat interface
st.title('🤖 AWU Agen AI Saham')
st.markdown("""
    Selamat datang di asisten AI Anda yang dilengkapi dengan
    kemampuan analisis berita saham dan kripto terbaru!
""")

# Chat element
chat_container = st.container()

with chat_container:
    with st.chat_message('user'):
        st.markdown('Hello')

# Chat input
msg = (
    'Tanyakan apa saja kepada saya atau analisis data'
    + 'berita yang telah Anda kumpulkan...'
)


if prompt := st.chat_input(msg):
    pass


# Quick actions
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('🧹 Hapus Chat'):
        st.rerun()

with col2:
    if st.button('📊 Rangkuman Data'):
        st.rerun()

with col3:
    if st.button('💡 Ide Pertanyaan'):
        st.rerun()

with col4:
    if st.button('🔄 Muat Ulang'):
        st.rerun()

# Footer
st.divider()
st.markdown("""
    **AWU Agen AI Saham** - Dibuat menggunakan Streamlit
    | Bangun percakapan cerdas berbasis data
""")

# ==================================================== #
# OLD

# from streamlit import session_state as ss
# from streamlit import secrets as sc
# from datetime import datetime
# import streamlit as st
# import pandas as pd
# import random
# import time
# import os

# # Import all of the function from the module
# from module import WebScraper, StockSharesAIAgent, StockNewsEmbeddingSystem


# # Initialize pathing
# ws_path = os.getcwd()
# data_dir = os.path.join(ws_path, 'dataset')
# db_path = os.path.join(ws_path, 'chroma_db')

# # Initialize embedding system
# collection_name='stock_news'

# # Page configuration
# st.set_page_config(
#     page_title='AWU Stock Shares AI Agent',
#     page_icon='🤖',
#     layout='wide',
#     initial_sidebar_state='expanded'
# )

# # Initialize session state
# if 'messages' not in ss:
#     ss.messages = []

# # Sidebar for configuration and dataset scraping
# with st.sidebar:
#     st.title('🛠️ Configuration')
    
#     # AI Model Settings
#     st.subheader('AI Settings')
#     model_type = st.selectbox(
#         'Select AI Model',
#         ['o4-mini-2025-04-16'],
#         index=0
#     )
    
#     max_tokens = st.slider('Max Tokens', 100, 4000, 1000, 100)
    
#     st.divider()
    
#     # Dataset Scraping Section
#     st.subheader('📊 Dataset Scraping')
    
#     start_date = st.date_input('Start Date')
#     days_periods = st.slider(
#         'Days Periods', min_value=1, max_value=90, value=7, format='%d Days'
#     )

#     if st.button('🔍 Start Scraping') and start_date:
#         Scraper = WebScraper(data_dir=data_dir)
#         Scraper.set_periods(start_date, days_periods)

#         with st.spinner('Scraping in progress, please wait...'):
#             Scraper.start_scraping()

#             st.toast('Scraping success! Dataset has been saved', icon='✔')
            
#         Scraper.get_csv()

# # Main chat interface
# st.title('🤖 AWU Stock Shares AI Agent')
# st.markdown(
#     'Welcome to your AI-powered assistant with dataset analysis capabilities!'
# )

# # Chat history
# chat_container = st.container()

# with chat_container:
#     for message in ss.messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])

# # Chat input
# if prompt := st.chat_input('Ask me anything or analyze your scraped data...'):
#     # Add user message
#     ss.messages.append({'role': 'user', 'content': prompt})
    
#     with st.chat_message('user'):
#         st.markdown(prompt)
    
#     # Generate AI response
#     with st.chat_message('assistant'):
#         with st.spinner('Thinking...'):
#             Agent = StockSharesAIAgent(
#                 model_name=model_type,
#                 persist_directory=db_path,
#                 collection_name=collection_name,
#                 max_tokens=max_tokens
#             )

#             result = Agent.chat(prompt, ss.messages)
#             response = result['response']

#         st.markdown(response)
        
#         # for i, elem in enumerate(result['sources']): 
#         #     msg = f'{i}: {elem['title']} - {elem['link']}'
#         #     st.markdown(msg)

#     # Add assistant response to chat history
#     ss.messages.append({'role': 'assistant', 'content': response})

#     # Delete old response
#     if len(ss.messages) > 10:
#         ss.messages = ss.messages[:9]

# # Quick actions
# st.divider()
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     if st.button('🧹 Clear Chat'):
#         ss.messages = []
#         st.rerun()

# with col2:
#     if st.button('📊 Show Dataset Summary'):
#         if ss.scraped_datasets:
#             st.rerun()
#         else:
#             st.info('No datasets available. Please scrape or upload data first.')

# with col3:
#     if st.button('💡 Get Suggestions'):
#         suggestions = [
#             'Try asking me to analyze patterns in your data',
#             'I can help create visualizations from your datasets',
#             "Ask me to compare different datasets you've scraped",
#             'I can identify trends and anomalies in your data'
#         ]
#         suggestion = random.choice(suggestions)
#         ss.messages.append({'role': 'assistant', 'content': suggestion})
#         st.rerun()

# with col4:
#     if st.button('🔄 Refresh Interface'):
#         st.rerun()

# # Footer
# st.markdown('---')
# st.markdown('**AWU Stock Shares AI Agent** - Powered by Streamlit | Build intelligent data-driven conversations')