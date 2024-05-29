import requests
from bs4 import BeautifulSoup as bs
import regex as re
import pandas as pd
from tqdm import tqdm
import time
from multiprocessing import Pool

url_base = "https://icondenim.com"

def get_one_product(product_url):
    response = requests.get(product_url)
    soup = bs(response.text, "html.parser")
    title = soup.find("h1", class_="title-product").text
    description = re.sub(r'\xa0', ' ', soup.find("div", class_="accordion-content").find_all("ul")[-1].text + soup.find("div", class_="accordion-content").find_all("p")[-1].text)
    color = [i.text for i in soup.find("div", class_="swatch-element-list").find_all("div", class_="tooltip")]
    img_list = [i["data-img"] for i in soup.find("div", class_="section slickthumb_relative_product_1").find_all("div", class_="item")]
    return {"title": title, "description": description, "color": color, "img_list": img_list}

def get_data(url):
    response = requests.get(url)
    soup = bs(response.text, "html.parser")

    data = {
        "title": [],
        "description": [],
        "color": [],
        "img_list": []
    }

    try:
        href_list = [url + i.find("a")["href"] for i in soup.find_all("li", class_="page-item")[2:-1]]
        href_list.insert(0, url)
    except:
        href_list = [url]

    for i in tqdm(href_list):
        response = requests.get(i)
        soup = bs(response.text, "html.parser")
        href_product = [url_base + i.find("a")["href"] for i in soup.find_all("div", class_="product-thumbnail")]
        for i in tqdm(href_product):
            try:
                data["title"].append(get_one_product(i)["title"])
                data["description"].append(get_one_product(i)["description"])
                data["color"].append(get_one_product(i)["color"])
                data["img_list"].append(get_one_product(i)["img_list"])
            except:
                pass

    return data


if __name__ == "__main__":
    url_list = ["https://icondenim.com/collections/nhom-ao",
        "https://icondenim.com/collections/nhom-quan"]

    data = {"title": [], 
            "description": [], 
            "color": [], 
            "img_list": []
            }

    with Pool(2) as p:
        data_list = tqdm(p.imap(get_data, url_list), total=len(url_list))
        for i in data_list:
            data["title"] += i["title"]
            data["description"] += i["description"]
            data["color"] += i["color"]
            data["img_list"] += i["img_list"]

    df = pd.DataFrame(data)
    df.to_csv("icondenim.csv", index=False)