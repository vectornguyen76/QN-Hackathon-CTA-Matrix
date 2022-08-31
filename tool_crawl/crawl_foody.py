from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import csv
import re 

# Preprocess data
def preprocess(text):
    html_pattern = re.compile('<.*?>')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = html_pattern.sub(r' ', text)
    text = url_pattern.sub(r' ', text)
    text = text.replace("\n", ". ")
    text = text.replace(";", "")
    
    return text

browser = webdriver.Edge(executable_path='driver\msedgedriver.exe')

# Maximize window
browser.maximize_window()

# Open foody
path_search = "https://www.foody.vn/binh-dinh/travel/khu-du-lich"
browser.get(path_search)
sleep(2)

# Scroll page foody to get more location
for _ in range(10):
    browser.execute_script("window.scrollBy(0, 600)")
    sleep(2)
    
# Get number comments and links 
cmts = browser.find_elements(By.XPATH, "//span[@data-bind='text: TotalReview.formatK(1)']")
elements = browser.find_elements(By.XPATH, "//div[@class='resname']//h2//a")

location = []
for i in range(len(cmts)):
    if int(cmts[i].text) >= 2:
        location.append(elements[i].get_attribute("href"))
        
for link in location:
    browser.get(link)
    sleep(2)
    
    browser.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    sleep(2)
    
    reviews = browser.find_elements(By.XPATH, "//span[@ng-bind-html='Model.Description']")
    print("total reviews", len(reviews))
    
    results = []
    for review in reviews:
        text = review.text
        text = preprocess(text)
        if len(text) < 40:
            continue
        results.append([text])
    
    print("total results", len(results))
    
    with open(f"data_foody.csv", "a", encoding='utf-8', newline='') as file:
        # Create a CSV writer
        writer = csv.writer(file)
        
        # Write data to the file
        writer.writerows(results)
    
    sleep(2)

# Close browser
browser.close()