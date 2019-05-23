# %% [markdown]
# ## Import necessary libraries

# %%
import re
import os
import requests

from bs4 import BeautifulSoup
from datetime import datetime

import numpy as np
import pandas as pd


# %% [markdown]
# ## Scrape reviews from Consumer Affairs


# %%
base_url = 'https://www.consumeraffairs.com/health_clubs/la_fitness.html'

reviews = pd.DataFrame(columns=['id', 'date', 'rating', 'review'])

if not os.path.isdir('./data'):
    os.mkdir('./data')

# Scrape all 1627 reviews from 55 pages
for i in range(1, 56):
    print('Scraping page {}...'.format(i))

    page = requests.get('{}?page={}'.format(base_url, i))
    soup = BeautifulSoup(page.content, 'html.parser')

    reviews_list = soup.find_all('div', {'class': 'rvw'})

    for outer in reviews_list:
        review = {}
        review['id'] = int(outer['data-id'])

        # print(review['id'])

        body = outer.find('div', {'class': 'rvw-bd'})
        meta = outer.find_all('meta')

        try:
            [review['rating']] = [item['content']
                                  for item in meta if item['itemprop'] == 'ratingValue']
        except(ValueError):
            review['rating'] = None

        pattern = r'Original review: (.*)'
        date_string = body.find('span', recursive=False).text
        date_match = re.match(pattern, date_string).group(1)

        # Map non-standard month abbreviations to avoid casting errors
        month_map = {'Jan.': 'January', 'Feb.': 'February', 'March': 'March',
                     'April': 'April', 'May': 'May', 'June': 'June', 'July': 'July',
                     'Aug.': 'August', 'Sept.': 'September', 'Oct.': 'October',
                     'Nov.': 'November', 'Dec.': 'December'}

        month = month_map[re.match(r'([A-Za-z]+[.]?)', date_match).group(1)]
        date_match = re.sub(r'([A-Za-z]+)[.]?', month, date_match)

        review['date'] = datetime.strptime(date_match, '%B %d, %Y')

        review_text = body.find('p').getText()

        # If review has collapsed text, append to string
        if body.find('div', {'class': 'js-collapsed'}) is not None:
            expanded_text = body.find(
                'div', {'class': 'js-collapsed'}).getText()
            review_text += ' ' + expanded_text

        # Strip newlines and excess whitespace
        review_text = review_text.replace('\n', '').replace('\r', '')
        review_text = re.sub(r' +', ' ', review_text)

        review['review'] = review_text
        reviews = reviews.append(review, ignore_index=True)

reviews.to_csv('./data/review.csv')
