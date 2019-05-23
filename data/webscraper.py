from bs4 import BeautifulSoup as bs
import requests


url = 'https://vecka.nu'
jobURL = 'https://www.linkedin.com/jobs/search/?currentJobId=1217637947&keywords=stockholm&location=Stockholm%2C%20Sverige&locationId=se%3A8064'

data = requests.get(jobURL)
soup = bs(data.text, 'html.parser')


for time in soup.find_all('job'):
    print(time.text)

