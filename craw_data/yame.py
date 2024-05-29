import requests
from bs4 import BeautifulSoup as bs
import regex as re
import pandas as pd
from tqdm import tqdm
import time
from multiprocessing import Pool


url_base = "http://yame.vn"

def get_one_product(url):
    response = requests.get(url)
    soup = bs(response.text, "html.parser")
    img_list = [i["src"] for i in soup.find("div", class_="product-info").find_all("img", class_="img-fluid")]
    description = soup.find("div", id="moTaSanPham").text
    color = re.match(r'^.*?(?=,)', soup.find("table", class_="table-productvariants").find("td").text).group()
    title = soup.find("span", class_="productName").text
    
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
        href_page = [url_base + x["href"].replace(" ", "%20") for x in soup.find("div", class_="site-block-27 mb-2").find_all("a")[1:-1]]
        href_page.insert(0, url)
    except:
        href_page = [url]

    href_list = []

    for i in tqdm(href_page):
        response = requests.get(i)
        soup = bs(response.text, "html.parser")
        href_list.extend(url_base + x.replace(" ", "%20") for x in [j.find("div", class_="owl-carousel").find("a")["href"] for i in soup.find_all("div", class_="row") for j in i.find_all("div", class_="col-md-3 col-6") if j.find("div", class_="owl-carousel") is not None])

    for i in tqdm(href_list):
        try:
            data_one_product = get_one_product(i)
            data["title"].append(data_one_product["title"])
            data["description"].append(data_one_product["description"])
            data["color"].append(data_one_product["color"])
            data["img_list"].append(data_one_product["img_list"])
        except:
            pass
        
    
    return data

if __name__ == "__main__":
    url_list = ["https://yame.vn/shop/ao-thun?page=1&sort=11&opts=ChungLoai_%C3%81o%20Thun",
                "https://yame.vn/shop/Ao-polo-don-gian-thiet-ke-yame?page=1&sort=2&opts=",
                "https://yame.vn/shop/ao-so-mi?page=1&sort=2&opts=",
                "https://yame.vn/shop/ao-khoac?page=1&sort=2&opts=",
                "https://yame.vn/shop/quan-tay?sort=2",
                "https://yame.vn/shop/quan-jean?page=1&sort=2&opts=",
                "https://yame.vn/shop/quan-dai?page=2&sort=2&opts=",
                "https://yame.vn/shop/quan-short?page=2&sort=2&opts="]

    data = {"title": [], 
            "description": [], 
            "color": [], 
            "img_list": []
            }

    with Pool(8) as p:
        data_list = tqdm(p.imap(get_data, url_list), total=len(url_list))
        for i in data_list:
            data["title"].extend(i["title"])
            data["description"].extend(i["description"])
            data["color"].extend(i["color"])
            data["img_list"].extend(i["img_list"])

        df = pd.DataFrame(data)
        df.to_csv("yame.csv", index=False)