# Import the required libraries
import os
import pandas as pd

dataframe = pd.read_excel('../datasets/data_relabel/Relabel-Done-datasets.xlsx')

# Iterate over all the files in the directory
dir = '../datasets/data_labeling/'
for file in os.listdir(dir):
   if file.startswith('Crawl-Done'):
      # Create the filepath of particular file
      dataframe = pd.concat([dataframe, pd.read_excel(dir + file)])

print(dataframe.shape)

# Shuffle dataframe
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

dataframe.to_csv('merged_dataset.csv', index=False)