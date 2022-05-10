from bs4 import BeautifulSoup as BS
import requests as req
import pandas as pd
url = 'https://www.officialdata.org/us/stocks/s-p-500/1900'


html_text = req.get(url)
soup = BS(html_text.content, 'lxml')

items = soup.find_all('th')
headings = []
for i in items:
    headings.append(i.text)
print(headings)


ix = 0
for i in soup.find_all('table', class_='regular-data table-striped'):
    ix = ix + 1
print(ix)

table = soup.find('table', class_='regular-data table-striped')
body = table.find_all('tr')

all_rows = []
for tr in body[1:]:
    body_rows = tr.find_all('td')
    row = [i.text for i in body_rows]
    all_rows.append(row)


df = pd.DataFrame(all_rows, columns=headings)
display(df)
