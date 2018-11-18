from bs4 import BeautifulSoup as bs
import requests


url = 'https://vecka.nu'

data = requests.get(url)
soup = bs(data.text, 'html.parser')


for time in soup.find_all('time'):
    print(time.text)

