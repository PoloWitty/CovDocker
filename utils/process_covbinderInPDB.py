"""
desc:	download pdb files for covbinderInPDB dataset
author:	Yangzhe Peng
date:	2024/01/05
"""


import os
import argparse

import subprocess

import tqdm
import pandas as pd


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_info_filename', type=str, default='./data/covbinderInPDB/CovBinderInPDB_2022Q4_AllRecords.csv')
    parser.add_argument('--save_dir', type=str, default='./data/covbinderInPDB/pdb/')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset_info = pd.read_csv('./data/covbinderInPDB/CovBinderInPDB_2022Q4_AllRecords.csv')

    for index, row in tqdm.tqdm(dataset_info.iterrows(),total=len(dataset_info),desc='downloading pdb'):
        pdb_id = row['pdb_id']
        # het_id = row['binder_id_in_adduct']
        # smiles = row['binder_smiles']
        # binder_id = row['binder_id']
        save_path = args.save_dir + pdb_id
        if os.path.exists(save_path+'.pdb'):
            continue
        subprocess.getoutput(f'pdb_fetch {pdb_id} > {save_path}.pdb')