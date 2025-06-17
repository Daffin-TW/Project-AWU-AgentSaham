#!/usr/bin/python3

from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import streamlit as st
from tqdm import tqdm
import pandas as pd
import requests
import os


class WebScraper():
    def __init__(
        self,
        url='https://www.kontan.co.id/search/indeks?kanal=investasi&',
        data_dir='./scraping_result/',
        start_date=datetime.now(),
        days_period=7
    ):
        self.url = url
        self.data_dir = data_dir
        self.start_date = start_date
        self.days_period = days_period
        self.data = pd.DataFrame(columns=['date', 'title', 'url', 'text'])

    def set_periods(self, start_date: datetime, days_period: int):
        """Change prefered news periods"""
        
        self.start_date = start_date
        self.days_period = days_period

        msg = (
            f'New periods: date={start_date.date}'
            + f'days={days_period} is selected'
        )
        print(msg)

    def set_data_dir(self, new_dir: str):
        """Change prefered data directory"""
        
        self.data_dir = new_dir
        print(f'New directory: {new_dir} is selected')
    
    def get_dataframe(self):
        """Return data in DataFrame pandas format"""

        return self.data

    def get_csv(self, file_name='dataset.csv') -> bool:
        """Return data in csv format

        Return True if succeeded, False otherwise
        """

        # Check directory existence
        if os.path.isdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            self.data.to_csv(file_path)
            print(f'Dataset has been saved to {file_path}')
            
            return True
        
        else:
            print('='*50, '\nERROR\n', self.data_dir, 'is not found')

            return False

    def scrap_article(self, url: str) -> str:
        """Scrap and clean all text in article"""

        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')

        text_elems = soup.find_all('p')[1:-1]

        for text_elem in text_elems:
            [i.extract() for i in text_elem.select('b')]
            [i.extract() for i in text_elem.select('a')]
            [i.extract() for i in text_elem.select('h2')]
            [i.extract() for i in text_elem.select('strong')]

        text = '\n'.join([elem.get_text() for elem in text_elems])

        return text

    def scrap_urls(self, replace_data=True, st_progress=False):
        """Get all the urls in selected time period

        Args:
            replace_data: Replace current data attribute.
            st_progress: Add progress bar UI for Streamlit.

        """

        df = pd.DataFrame(columns=['date', 'title', 'url', 'text'])
        end_date = self.start_date - timedelta(self.days_period)
        dates = pd.date_range(end_date, periods=self.days_period).tolist()
        n_dates = len(dates)
        date_tqdm = tqdm(dates)

        if st_progress:
            progress_text = 'Proses web crawling sedang berjalan...'
            st_bar = st.progress(0, text=progress_text) 

        # Search for every date in time periods
        for idx, date in enumerate(date_tqdm):
            
            # Update progress bar visual
            desc = f'Web crawling in progress... {idx+1}/{n_dates}'
            date_tqdm.set_description(desc)

            if st_progress:
                progress_percentage = (idx+1) / n_dates
                st_bar.progress(progress_percentage, progress_text)

            # Request a page on specific date
            date_str = date.strftime('%d-%m-%Y')
            dd = date.day
            mm = date.month
            yy = date.year

            target = f'tanggal={dd}&bulan={mm}&tahun={yy}&pos=indeks&per_page='
            main_url = (self.url + target)

            req = requests.get(main_url)
            soup = BeautifulSoup(req.text, 'html.parser')
            
            # Check number of pages
            nav = soup.find('ul', class_='cd-pagination')

            if nav:
                nav_elems = nav.find_all('a')
                n_page = len([
                    elem.text for elem in nav_elems if elem.text.isdigit()
                ])
            else:
                n_page = 1

            pages = ['', *(str(i) for i in range(20, n_page*20, 20))]

            # Get all news urls in all pages
            for page in pages:
                
                # Make a request for every new pages
                if page:
                    req = requests.get(main_url + page)
                    soup = BeautifulSoup(req.text, 'html.parser')

                urls_elems = soup.find_all('div', class_='sp-hl linkto-black')

                # Get all news urls
                for element in urls_elems:
                    title = element.text
                    url = element.find('a').attrs['href']

                    df.loc[len(df)] = [date_str, title, url, None]

        if st_progress:
            st_bar.empty()

        if replace_data:
            # Apply changes to class attribute
            self.data = df.copy()
        else:
            return df

    def scrap_from_urls(
            self, data: pd.DataFrame = None, 
            replace_data=True, st_progress=False
        ):
        """Scrap all text from urls

        Args:
            replace_data: Replace current data attribute.
            st_progress: Add progress bar UI for Streamlit.

        """

        if not data:
            df = self.data.copy()
        else:
            df = data.copy()

        tqdm_df = tqdm(df.iterrows())
        n = len(df)

        if st_progress:
            progress_text = 'Proses scraping teks sedang berjalan...'
            st_bar = st.progress(0, text=progress_text) 

        for idx, row in tqdm_df:
            # Update progress bar visual
            desc = f'Scrapping: {row['date']} article: {idx+1}/{n}'
            tqdm_df.set_description(desc)

            if st_progress:
                st_bar.progress(0, text=desc)

            df.loc[idx, 'text'] = self.scrap_article(row['url'])
        
        if st_progress:
            st_bar.empty()

        if replace_data:
            # Apply changes to class attribute
            self.data = df.copy()
        else:
            return df


def main():
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    ws_path = os.path.dirname(folder_path)

    data_dir = os.path.join(ws_path, 'dataset')
    os.makedirs(data_dir, exist_ok=True)

    start_date = datetime.now()
    days_period = 7

    Scraper = WebScraper(
        start_date=start_date,
        days_period=days_period,
        data_dir=data_dir
    )
    Scraper.scrap_urls()
    Scraper.scrap_from_urls()
    print(Scraper.get_dataframe())

if __name__ == '__main__':
    main()