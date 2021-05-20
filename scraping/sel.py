from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located


from selenium.webdriver.chrome.options import Options
opts = Options()
opts.add_argument("/home/jurkis/.config/google-chrome/Default")
opts.add_argument("--disable-extensions")
#opts.add_argument("--headless")
#This example requires Selenium WebDriver 3.13 or newer


path = "https://zlatyfond.sme.sk"
links = []
with open("/home/jurkis/leto2021/slovak/linky_zl") as f:
    data = f.readlines()
    for x in data:
        links.append(path+x.strip())

#print(links[0])
import sys
#$sys.exit(0)

from tqdm import tqdm 

def every_downloads_chrome(driver):
    if not driver.current_url.startswith("chrome://downloads"):
        driver.get("chrome://downloads/")
    return driver.execute_script("""
        var items = document.querySelector('downloads-manager')
            .shadowRoot.getElementById('downloadsList').items;
        if (items.every(e => e.state === "COMPLETE"))
            return items.map(e => e.fileUrl || e.file_url);
        """)

pbar = tqdm(links)
import time
for i,dielo in enumerate(pbar):
    pbar.set_description(f"{dielo.split('/')[-1]}")
    with webdriver.Chrome(executable_path="/usr/bin/chromedriver",options=opts) as driver:
        time.sleep(0.9)
        driver.get(dielo)

        elems = driver.find_elements(By.TAG_NAME,"a")
        time.sleep(0.5)
        for e in elems:
            if e.text == "html":
                e.click()
                time.sleep(1.5)
                
