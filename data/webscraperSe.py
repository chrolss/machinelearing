from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver


driver = webdriver.Firefox()
driver.implicitly_wait(30)
driver.get(url)
