# #!/usr/bin/python3

# from datetime import datetime, timedelta
# from bs4 import BeautifulSoup
# from tqdm import tqdm
# import pandas as pd
# import requests
# import os


# class WebScraper():
#     def __init__(
#         self,
#         url='https://www.kontan.co.id/search/indeks?kanal=investasi&',
#         data_dir='./scraping_result/',
#     ):
#         self.url = url
#         self.periods = (datetime.now(), 7)
#         self.data_dir = data_dir
#         self.data = pd.DataFrame(columns=['Date', 'Title', 'Link', 'Text'])
        
#     # Change prefered news periods
#     def set_periods(self, end_date: datetime, days_periods: int):
#         start_date = end_date - timedelta(days_periods)
#         self.periods = (start_date, days_periods)

#     # Change prefered data directory
#     def set_data_dir(self, new_dir: str):
#         self.data_dir = new_dir

#     # Return data in DataFrame pandas format
#     def get_dataframe(self):
#         return self.data
    
#     # Return data in csv format
#     def get_csv(self, file_name='dataset.csv'):
        
#         # Check directory existence
#         if os.path.isdir(self.data_dir):
#             file = os.path.join(self.data_dir, file_name)
#             self.data.to_csv(file)
#         else:
#             print('='*50, '\nERROR\n', self.data_dir, 'is not found')

#     # Scraping function
#     def start_scraping(self):
#         df = pd.DataFrame(columns=['Date', 'Title', 'Link', 'Text'])
#         dates = pd.date_range(self.periods[0], periods=self.periods[1]).tolist()
#         date_tqdm = tqdm(dates)

#         # Search for every date in time periods
#         for date in date_tqdm:
#             dd = date.day
#             mm = date.month
#             yy = date.year
            
#             target = f'tanggal={dd}&bulan={mm}&tahun={yy}&pos=indeks&per_page='
#             url = (self.url + target)

#             req = requests.get(url)
#             soup = BeautifulSoup(req.text, 'html.parser')
            
#             # Check number of pages
#             nav = soup.find('ul', class_='cd-pagination')

#             if nav:
#                 nav_elems = nav.find_all('a')
#                 n_page = len([
#                     elem.text for elem in nav_elems if elem.text.isdigit()
#                 ])
#             else:
#                 n_page = 1

#             pages = ['', *(str(i) for i in range(20, n_page*20, 20))]

#             # Get all news urls in all pages
#             for index, page in enumerate(pages):
                
#                 # Make a request for every new pages
#                 if page:
#                     req = requests.get(url + page)
#                     soup = BeautifulSoup(req.text, 'html.parser')

#                 links = soup.find_all('div', class_='sp-hl linkto-black')

#                 # Get all news text
#                 for element in links:
#                     title = element.text
#                     link = element.find('a').attrs['href']

#                     desc = (
#                         f'Scrapping: {dd}/{mm}/{yy} page: {index+1}/{n_page} '
#                         + f'article: {title[:20]}... | '
#                     )
#                     date_tqdm.set_description(desc)

#                     req = requests.get(link)
#                     soup = BeautifulSoup(req.text, 'html.parser')

#                     text_elems = soup.find_all('p')[1:-1]

#                     for text_elem in text_elems:
#                         [i.extract() for i in text_elem.select('b')]
#                         [i.extract() for i in text_elem.select('a')]
#                         [i.extract() for i in text_elem.select('h2')]
#                         [i.extract() for i in text_elem.select('strong')]

#                     text = '\n'.join([elem.get_text() for elem in text_elems])

#                     df.loc[len(df)] = [date, title, link, text]
        
#         # Apply changes to class attribute
#         self.data = df.copy()


# def main():
#     file_path = os.path.abspath(__file__)
#     folder_path = os.path.dirname(file_path)
#     ws_path = os.path.dirname(folder_path)
    
#     data_dir = os.path.join(ws_path, 'dataset')
#     os.makedirs(data_dir, exist_ok=True)

#     end_date = datetime.now()
#     days_periods = 30
    
#     Scraper = WebScraper(data_dir=data_dir)
#     Scraper.set_periods(end_date, days_periods)
#     Scraper.start_scraping()

#     Scraper.get_csv()


# if __name__ == '__main__':
#     main()