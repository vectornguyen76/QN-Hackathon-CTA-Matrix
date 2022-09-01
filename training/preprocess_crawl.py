import pandas as pd
import re
import csv

def preprocess(text):
    html_pattern = re.compile('<.*?>')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = html_pattern.sub(r' ', text)
    text = url_pattern.sub(r' ', text)
    text = text.replace("\n", ". ")
    text = text.replace("\r.", "")
    text = text.replace("\r", "")
    text = text.replace(";", "")    
    
    return text

data = pd.read_csv('../datasets/data_crawl/data_vungtau_ggmap.csv')

data.columns =['Review']

results = []
i = 0
for text in data['Review']:
    i+=1
    if 40 < len(text) < 1200 and text.find("(Bản dịch của Google)") == -1 and i % 2 == 0:
        text = preprocess(text)
        results.append([text,0,0,0,0,0,0])
        
head = ['Review', 'giai_tri', 'luu_tru','nha_hang','an_uong','di_chuyen','mua_sam']

# Write data to file csv
with open(f"../datasets/data_preprocess/data_preprocess_vungtau.csv", "a", encoding='utf-8', newline='') as file:
    # Create a CSV writer
    writer = csv.writer(file)
    
    writer.writerow(head)
    
    # Write data
    writer.writerows(results)