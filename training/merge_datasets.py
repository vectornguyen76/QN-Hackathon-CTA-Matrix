# Import the required libraries
import os
import pandas as pd

def main():
   dataframe = pd.read_excel('../datasets/data_original/Original-datasets.xlsx')

   # Iterate over all the files in the directory
   dir = '../datasets/data_labeling/'

   for file in os.listdir(dir):
      if file.startswith('Crawl-Done'):
            
         # Create the filepath of particular file
         dataframe = pd.concat([dataframe, pd.read_excel(dir + file)])

   # Shuffle dataframe
   dataframe = dataframe.sample(frac=1).reset_index(drop=True)

   print(dataframe.shape)

   # Save to train
   dataframe.to_csv('../datasets/data_training/merged_dataset.csv', index=False)
   
if __name__ == '__main__':
    main()
    