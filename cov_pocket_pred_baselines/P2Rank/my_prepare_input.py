"""
desc:	Prepare input data for P2Rank
author:	Yangzhe Peng
date:	2024/05/12
"""

import os
import pandas as pd
import tqdm
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, default='../../data/processed/dataset.filtered.csv')
    parser.add_argument("--res-dir", type=str, default='./p2rank_res/', help='path to store p2rank output')
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()
    
    if args.only_unseen:
        assert 'unseen' in args.dataset_filename, "only-unseen flag is set, but the dataset_info_file does not contain 'unseen' split"
    
    dataset_filename = args.dataset_filename
    data_dir = args.res_dir
    os.makedirs(data_dir, exist_ok=True)
    
    df = pd.read_csv(dataset_filename)
    
    split = 'test' if not args.only_unseen else 'unseen'
    split_df = df[df['set']==split]
    
    chain_list = []
    for i,row in tqdm.tqdm(split_df.iterrows(),total=len(split_df),desc='preparing input data'):
        chain_filename = row['bonded chain']
        chain_list.append(chain_filename)
    
    # prepare input ds
    ds = f"{data_dir}/p2rank_input_protein_remove_extra_chains_10A_list.ds"
    with open(ds, "w") as out:
        for chain_filename in chain_list:
            out.write(f"../../../.{chain_filename}\n") # ../../data/processed will be ./p2rank_res/../../data/processed/ in p2rank ds
    
    