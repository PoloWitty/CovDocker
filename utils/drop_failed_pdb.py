"""
desc:	drop those failed pdb entries from dataset.csv
author:	Yangzhe Peng
date:	2024/04/26
"""

import tqdm
import pandas as pd
import json
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_info_filename', type=str, default='./data/processed/dataset.filtered.random_split.csv')
    parser.add_argument('--process_log_filename', type=str, default='./data/processed/process_log.json')
    
    args = parser.parse_args()
    
    
    log_obj = json.load(open(args.process_log_filename))
    
    failed_pdb_list = []
    for task, log in log_obj.items():
        # if task != 'position_pred_preprocess':
        #     continue
        for reason, detail_dic in log.items():
            # if reason == 'exceed_max_len1022':
            #     continue
            failed_pdb_list += detail_dic['pdb_id']
    
    df = pd.read_csv(args.dataset_info_filename)
    df['pdb_id'] = df.apply(lambda x: x['bonded protein'].split('/')[-2], axis=1)
    df = df.drop(df[df['pdb_id'].isin(failed_pdb_list)].index)
    new_order = ['date', 'src', 'pdb_id', 'pre-reactive smiles', 'bonded protein', 'bonded pocket', 'bonded chain', 'bonded ligand', 'bond', 'length', 'products', 'reactants', 'set', 'products_wH', 'products_woAA', 'products_woAAbackbone'] # Specify the desired order of columns
    df = df.reindex(columns=new_order)
    df.to_csv(args.dataset_info_filename.replace('.csv','.filtered.csv'), index=False)
    
