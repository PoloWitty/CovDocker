"""
desc:	Prepare unseen data for the Chemformer model. (too hard to specify unseen as test split for chemformer's code)
author:	Yangzhe Peng
date:	2024/08/05
"""

import pandas as pd
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_info_file", type=str, default="./data/processed/dataset.csv")
    parser.add_argument("--save_filename", type=str, default="./chemformer_unseen.csv")
    args = parser.parse_args()
    
    assert 'unseen' in args.dataset_info_file, "only-unseen flag is set, but the dataset_info_file does not contain 'unseen' split"

    df = pd.read_csv(args.dataset_info_file)
    df = df[df['set']=='unseen']
    df['set'] = 'test'
    df.to_csv(args.save_filename, index=False)
    print('Done! result saved to', args.save_filename)