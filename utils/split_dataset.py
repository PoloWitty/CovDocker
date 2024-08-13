"""
desc:	Split the dataset into train, valid, and test sets.
author:	Yangzhe Peng
date:	2024/01/05
"""

import os
import argparse
import pdb
import random

import pandas as pd


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_info_filename', type=str, default='./data/processed/dataset.filtered.csv')
    parser.add_argument('--save_dir', type=str, default='./data/processed/dataset/')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load the example.pocket.csv file
    data = pd.read_csv(args.dataset_info_filename)

    # Convert the 'date' column to datetime format
    # data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y')
    data['date'] = pd.to_datetime(data['date'])

    # Split the dataset based on the conditions
    train_set = data[data['date'] < '2020-01-01']
    valid_test_set = data[data['date'] >= '2020-01-01']
    
    valid_set = valid_test_set.sample(frac=0.5, random_state=1)
    test_set = valid_test_set.drop(valid_set.index)
    
    # valid_set = data[(data['date'] >= '2020-01-01') & (data['src'] == 'covbinderInPDB')]
    # test_set = data[(data['date'] >= '2020-01-01') & (data['src'] == 'covpdb')]

    # Print the sizes of the train, valid, and test sets
    print(f"Train set size: {len(train_set)}")
    print(f"Valid set size: {len(valid_set)}")
    print(f"Test set size: {len(test_set)}")

    train_set = train_set.assign(set='train')
    valid_set = valid_set.assign(set='valid')
    test_set = test_set.assign(set='test')
    dataset = pd.concat([train_set, valid_set, test_set])
    dataset.to_csv(os.path.join(args.dataset_info_filename).replace('.csv','.random_split.csv'), index=False)
    
    # # Save the train, valid, and test sets
    # datasets_name = ['train', 'valid', 'test']
    # for idx,dataset in enumerate([train_set, valid_set, test_set]):
    #     dataset.to_csv(os.path.join(args.save_dir, datasets_name[idx]+'.csv') , index=False)