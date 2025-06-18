# #!/usr/bin/python3

from langchain_huggingface.llms import HuggingFacePipeline
import os


class ArticleSummarizer():
    def __init__(
        self,
        model='thivh/t5-base-indonesian-summarization-cased-finetuned-indosum',
        task='text2text-generation',
        device=0
    ):
        self.device = device

        self.model = self.load_summarizer_model(
            model=model, task=task
        )

    def load_summarizer_model(self, model: str, task: str):
        print(f'Loading {model} model...')
        
        return HuggingFacePipeline.from_model_id(
            model_id=model,
            task=task,
            device=self.device,
            model_kwargs={}
        )
    
    def summarize_from_vectorstores(
            self, documents: list[dict[str, any]]
        ) -> list[dict[str, any]]:
        """Summarize vectorstores search results"""

        for doc in documents:
            print(doc['content'])
            doc['content'] = self.model.invoke(doc['content'])
            doc['metadata']['content_length'] = len(doc['content'])
            print()
            
        return documents


def main():
    # Import previous class
    from embedding import ChromaEmbeddings

    # Initialize Pathing
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    ws_path = os.path.dirname(folder_path)
    db_path = os.path.join(ws_path, 'chroma_db')

    # Model building
    Embeddings = ChromaEmbeddings(
        persist_directory=db_path,
        collection_name='stock_news'
    )
    Embeddings.load_existing_vectorstore()

    Summarizer = ArticleSummarizer()
    
    # Summarizer Examples
    print('\n=== Summarizer Examples ===')
    query = 'Volalitas saham perusahaan di bidang telekomunikasi'
    search_result = Embeddings.similarity_search_metadata_filter(query, k=3)
    summ_result = Summarizer.summarize_from_vectorstores(search_result)

    for i, result in enumerate(summ_result, 1):
        print(f'{i}. {result['title']}')
        print(f'   Similarity Score: {result['similarity_score']:.4f}')
        print(f'   Date: {result['date']}')
        print(f'   Url: {result['url']}')
        print(f'   Content:\n{result['content']}')
        print()


if __name__ == '__main__':
    main()