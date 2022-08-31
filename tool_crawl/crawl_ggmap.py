from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import csv
import re 

browser = webdriver.Edge(executable_path='driver\msedgedriver.exe')

# Maximize window
browser.maximize_window()

# Open google map
path_search = "https://www.google.com/maps/@9.779349,105.6189045,11z?hl=vi-VN"
browser.get(path_search)
sleep(2)

# Search 
input_search = browser.find_element(By.ID,"searchboxinput")
input_search.send_keys("nhà hàng vũng tàu")
input_search.send_keys(Keys.ENTER)
sleep(4)

def preprocess(text):
    html_pattern = re.compile('<.*?>')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = html_pattern.sub(r' ', text)
    text = url_pattern.sub(r' ', text)
    text = text.replace("\n", ".")
    text = text.replace(";", "")
    
    return text

scroll_items = browser.find_elements(By.XPATH, "//div[@class='m6QErb DxyBCb kA9KIf dS8AEf ecceSd']")
sleep(3)
for _ in range(50):
    sleep(2)
    scroll_items[1].send_keys(Keys.DOWN);
    
# Get result search
output_searchs = browser.find_elements(By.XPATH, "//div[@class='Nv2PK THOPZb CpccDe']")

print(len(output_searchs))
sleep(3)

for output_search in output_searchs:
    results = []
    
    try:
        sleep(1)
        scroll_items[1].send_keys(Keys.DOWN)
        
        index = output_searchs.index(output_search)
        print("Item: ", index)
        
        # Choose each output
        output_search.click()
        sleep(3)

        # Get label review and click
        label_review = browser.find_element(By.XPATH, "//button[@class='DkEaL']")
        numbers_review = label_review.text.replace(".", "")
        numbers_review = int(re.findall(r'\d+', numbers_review)[0])
        print("scroll", numbers_review)
        if numbers_review > 1000:
            numbers_review = 1000
        
        label_review.click()
        sleep(3)
    except:
        print("Error")
        continue
    
    # Scroll to get more result
    scroll = browser.find_element(By.XPATH, "//div[@class='m6QErb DxyBCb kA9KIf dS8AEf']")
    for _ in range(numbers_review):
        # Click "Xem thêm" to get full review
        try:
            see_more = browser.find_element(By.XPATH, "//button[@aria-label='Xem thêm']")
            see_more.click()
        except:
            pass
            
        # Scroll 
        sleep(0.5)
        scroll.send_keys(Keys.DOWN);
    
    # Get reviews
    reviews = browser.find_elements(By.XPATH, "//span[@class='wiI7pd']")
    print("total reviews", len(reviews))
    
    # Preproces review
    for review in reviews:
        text = review.text
        text = preprocess(text)
        if len(text) < 40:
            continue
        results.append([text])
    
    print("total results", len(results))
    
    # Write data to file csv
    with open(f"../datasets/data_crawl/data_vungtau_ggmap.csv", "a", encoding='utf-8', newline='') as file:
        # Create a CSV writer
        writer = csv.writer(file)
        
        # Write data
        writer.writerows(results)
    
    sleep(3)
# Đóng trình duyệt
browser.close()