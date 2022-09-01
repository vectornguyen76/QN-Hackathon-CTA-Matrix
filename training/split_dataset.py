from utils import split_train_test
import pandas as pd
import numpy as np

def main():
    # Read 
    all_datasets = pd.read_csv("../datasets/data_training/merged_dataset.csv")

    while True:
        train_set, test_set = split_train_test(all_datasets, 0.2)
        
        ratio = (all_datasets.iloc[:, 1:] != 0).sum() / len(all_datasets)
        
        train_ratio = (train_set.iloc[:, 1:] != 0).sum() / len(train_set)
        val_ratio = (test_set.iloc[:, 1:] != 0).sum() / len(test_set)
        
        if np.all(((ratio - val_ratio).abs() < 0.002).values) and np.all(((ratio - train_ratio).abs() < 0.002).values):
            break

    print("Samples train: ", len(train_set))
    print("Samples val: ", len(test_set))

    # Save to csv
    train_set.to_csv('../datasets/data_training/train_datasets.csv')
    test_set.to_csv('../datasets/data_training/test_datasets.csv')
    
if __name__ == '__main__':
    main()
    