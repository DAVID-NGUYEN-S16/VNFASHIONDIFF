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
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import json
import requests
import os
from dataclasses import dataclass, asdict
from unidecode import unidecode

headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

url_base = "https://www.adidas.com.vn"

@dataclass
class Product:
    title: str
    color: str
    decripsion: str
    imgs: list

def get_href_product(url):
    global headers
    response = requests.get(url, headers=headers, timeout=10)
    soup = bs(response.content, 'html.parser')
    so_luong = int(soup.find("div", class_="count___11uU6").text.replace("[", "").replace("]", ""))
    # lập theo số lượng sản phẩm trên trang web để lấy hết tất cả các sản phẩm
    href_products = []
    for i in tqdm(range(1, so_luong, 48)):
        url = f"https://www.adidas.com.vn/vi/quan_ao?start={i}"
        response = requests.get(url, headers=headers, timeout=10)
        soup = bs(response.content, 'html.parser')
        href_products.extend([url_base + href_product.find("a", class_="glass-product-card__assets-link")["href"] 
                        for href_product in soup.find_all("div", class_="grid-item")]) # lấy link sản phẩm
    href_products = list(set(href_products)) # loại bỏ các sản phẩm trùng nhau
    return [unidecode(href).replace(' ', '-') for href in href_products]

def get_product_info(url):
    try:
        global headers
        response = requests.get(url, headers=headers, timeout=20)
        soup = bs(response.content, 'html.parser')
        data_string = soup.find("div", id="consent_blackbar").find_next_sibling().text
        data_string = data_string.replace('window.DATA_STORE = JSON.parse("', "").replace('");', "").replace('\\\\\\"','').replace('\\', '')
        data = json.loads(data_string)

        parts = url.split("/")

        product_code = parts[-1].replace(".html", "") # lấy mã sản phẩm
        # lấy thông tin sản phẩm
        color = data['productStore']["products"][product_code]["data"]["attribute_list"]["color"]
        imgs = [i['image_url'] for i in data['productStore']["products"][product_code]["data"]["view_list"]]
        title = data['productStore']["products"][product_code]["data"]["product_description"]["title"]
        try:
            decripsion = data['productStore']["products"][product_code]["data"]["product_description"]["subtitle"] + data['productStore']["products"][product_code]["data"]["product_description"]["text"]
        except:
            decripsion = None

        return {
            "title": title,
            "color": color,
            "decripsion": decripsion,
            "imgs": imgs,
        }
    except Exception as e:
        return get_product_info(url)

def get_data(url):
    try:
        global headers, url_base
        response = requests.get(url, headers=headers, timeout=20)
        soup = bs(response.content, 'html.parser')
        data = {
            "title": [],
            "color": [],
            "decripsion": [],
            "imgs": []
        }
        # kiểm tra xem nếu như sản phẩm có nhiều màu sắc thì lấy tất cả các màu sắc đó ra còn nếu chỉ có 1 màu sắc thì lấy luôn link sản phẩm đó
        if soup.find("div", class_="color-chooser-grid___1ZBx_") is not None:
            href_coler = [url_base + href["href"] for href in soup.find("div", class_="color-chooser-grid___1ZBx_").find_all("a")]
            for href in href_coler:
                data["color"].append(get_product_info(href)["color"])
                data["title"].append(get_product_info(href)["title"])
                data["decripsion"].append(get_product_info(href)["decripsion"])
                data["imgs"].append(get_product_info(href)["imgs"])
        elif soup.find("div", class_="single-color-label___29kFh") is not None:
            data["color"].append(get_product_info(url)["color"])
            data["title"].append(get_product_info(url)["title"])
            data["decripsion"].append(get_product_info(url)["decripsion"])
            data["imgs"].append(get_product_info(url)["imgs"])
        time.sleep(1)
        return data
    except Exception as e:
        return get_data(url)

if __name__ == "__main__":
    url = "https://www.adidas.com.vn/vi/quan_ao"
    hrefs = get_href_product(url=url)
    num_threads = os.cpu_count() - 3
    with Pool(processes=num_threads) as pool:
        results = list(tqdm(pool.imap(get_data, hrefs), total=len(hrefs)))

    data = []

    for result in results:
        for i in range(len(result["title"])):
            data.append({
                "title": result["title"][i],
                "color": result["color"][i],
                "decripsion": result["decripsion"][i],
                "imgs": result["imgs"][i]
            })

    df = pd.DataFrame(data)
    df.to_csv("adidas.csv", index=False)

    print("Done!")