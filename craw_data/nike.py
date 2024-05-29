import json
import requests
import pandas as pd
from tqdm import tqdm
import time
from multiprocessing import Pool
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup as bs

def get_href_product(url):
    service = Service(executable_path=ChromeDriverManager().install())
    options = Options()
    driver = webdriver.Chrome(service=service, options=options)
    
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

    time.sleep(5)

    check_height = 0
    while True:
        current_height = driver.execute_script("return document.body.scrollHeight;")
        
        for i in range(check_height, current_height, 1000):
            driver.execute_script(f"window.scrollTo({i}, {i+1000});")
            time.sleep(0.5)

        if current_height == check_height:
            break
        
        time.sleep(1)
        check_height = current_height

    products = driver.find_elements(By.XPATH, '//div[@data-testid="product-card"]')
    href_product = driver.find_elements(By.XPATH, '//*[@class="product-card__link-overlay"]')
    
    href_product = [hrf.get_attribute('href') for i, hrf in enumerate(href_product) if "socks" not in bs(products[i].get_attribute('outerHTML'), 'html.parser').find("div", class_="product-card__subtitle").text.lower()]
    
    driver.quit()
    return href_product

def get_product_info(url):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        'accept-language': 'en-US,en-CA;q=0.9,en;q=0.8,hi-IN;q=0.7,hi;q=0.6'
    }

    try:
        response = requests.get(url, headers=headers)

        soup = bs(response.text, 'html.parser')
        json_data = soup.find('script', id='__NEXT_DATA__')
        json_data = json.loads(json_data.string)
        
        product_keys = json_data['props']['pageProps']["initialState"]["Threads"]["products"].keys()

        data = {
            "description": [],
            "color": [],
            "title": [],
            "img": []
        }

        for i in product_keys:
            product = json_data['props']['pageProps']["initialState"]["Threads"]["products"][i]
            description = product["descriptionPreview"]
            color = product["colorDescription"]
            title = product["title"]
            img = [node["properties"]["portraitURL"] for node in product["nodes"][0]["nodes"] if "portraitURL" in node["properties"]]
            data["description"].append(description)
            data["color"].append(color)
            data["title"].append(title)
            data["img"].append(img)
        
        return data

    except Exception as e:
        return get_product_info(url)

def get_data(url):
    hrefs = get_href_product(url)
    data = {"description": [], "color": [], "title": [], "img": []}
    
    for href in tqdm(hrefs, desc=f"Processing {url}"):
        data_product = get_product_info(href)
        if data_product is None:
            continue
        for key in data.keys():
            data[key].extend(data_product[key])
    
    return data

if __name__ == "__main__":
    urls = [
        "https://www.nike.com/vn/w/womens-clothing-5e1x6z6ymx6",
        "https://www.nike.com/vn/w/mens-clothing-6ymx6znik1"
    ]
    
    with Pool(processes=len(urls)) as pool:
        results = pool.map(get_data, urls)
    
    combined_data = {"description": [], "color": [], "title": [], "img": []}
    
    for result in results:
        for key in combined_data.keys():
            combined_data[key].extend(result[key])
    
    df = pd.DataFrame(combined_data)
    df.to_csv("nike_products.csv", index=False)