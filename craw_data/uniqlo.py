import regex as re
import pandas as pd
from tqdm import tqdm
import time
from multiprocessing import Pool
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import json

url_api = "https://www.uniqlo.com/vn/api/commerce/v3"

def get_link(url):
    service = Service(executable_path=ChromeDriverManager().install()) 
    options = Options()
    options.add_argument('--log-level=3')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    driver.maximize_window()
    wait = WebDriverWait(driver, 10)
    wait.until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))
    height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1000);")
        if driver.find_elements(By.XPATH, '//*[@class="fr-load-more"]'):
            driver.find_element(By.XPATH, '//*[@class="fr-load-more"]').click()
            time.sleep(1)
        check_height = driver.execute_script("return document.body.scrollHeight")
        if check_height == height:
            break
        height = check_height
        time.sleep(3)

    soup = bs(driver.page_source, 'html.parser')
    links = [url_api + re.sub("/vn","", i.find("a")["href"]) for i in soup.find_all('article', class_='fr-grid-item')]
    driver.quit()
    return links

def get_data(link):
    service = Service(executable_path=ChromeDriverManager().install()) 
    options = Options()
    options.add_argument('--log-level=3')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(link)
    soup = bs(driver.page_source, 'html.parser')
    data = json.loads(soup.text)
    link_images = [i["url"] for i in data["result"]["items"][0]["images"]["main"]]
    color_names = [i["name"] for i in data["result"]["items"][0]["colors"]]
    description = re.sub("<br>", "\n", data["result"]["items"][0]["longDescription"])
    title = data["result"]["items"][0]["name"]
    time.sleep(5)
    driver.quit()
    return {"title": title, "description": description, "color": color_names, "img_list": link_images}

urls = [
    "https://www.uniqlo.com/vn/vi/women/tops/tops-collections",
    "https://www.uniqlo.com/vn/vi/women/outerwear/outerwear-collections",
    "https://www.uniqlo.com/vn/vi/women/bottoms/bottoms-collections",
    "https://www.uniqlo.com/vn/vi/women/dresses-and-jumpsuits/skirts",
    "https://www.uniqlo.com/vn/vi/women/dresses-and-jumpsuits/dresses-and-jumpsuits",
    "https://www.uniqlo.com/vn/vi/women/loungewear-and-homewear/loungewear-collections",
    "https://www.uniqlo.com/vn/vi/men/tops/tops-collections",
    "https://www.uniqlo.com/vn/vi/men/outerwear/outerwear-collections",
    "https://www.uniqlo.com/vn/vi/men/bottoms/bottoms-collections",
    "https://www.uniqlo.com/vn/vi/men/loungewear-and-homewear/loungewear-collections",
    "https://www.uniqlo.com/vn/vi/kids/tops/tops-collections",
    "https://www.uniqlo.com/vn/vi/kids/outerwear/outerwear-collections",
    "https://www.uniqlo.com/vn/vi/kids/bottoms/bottoms-collections",
    "https://www.uniqlo.com/vn/vi/kids/dresses-and-jumpsuits/dresses-and-jumpsuits",
    "https://www.uniqlo.com/vn/vi/kids/loungewear-and-homewear/loungewear",
    "https://www.uniqlo.com/vn/vi/baby/toddler/all-toddlers",
    "https://www.uniqlo.com/vn/vi/baby/newborn/all-newborn"
]

if __name__ == '__main__':
    data = []
    with Pool(3) as p:
        for i in tqdm(p.imap(get_link, urls), total=len(urls)):
            with Pool(5) as p:
                for j in tqdm(p.imap(get_data, i), total=len(i)):
                    data.append(j)
    df = pd.DataFrame(data)
    df.to_csv("uniqlo.csv", index=False)