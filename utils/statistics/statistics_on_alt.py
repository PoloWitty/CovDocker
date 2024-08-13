
import pandas as pd
import os
import tqdm

import multiprocessing
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True, timeout=60*15)

dataset_info_filename = './data/dataset.filtered.random_split.csv'
df = pd.read_csv(dataset_info_filename)

test_df = df[df['set']=='test']
df = test_df

Alt_num = 0
for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='statistic'):
    pdb_id = row['pdb_id']
    ligand_pdb_filename = row['bonded ligand']
    
    # run(f'python ../AutoDockTools_py3/AutoDockTools/Utilities24/prepare_pdb_split_alt_confs.py -r {ligand_pdb_filename} -o {ligand_pdb_filename.replace(".pdb","")}')
    if os.path.exists(ligand_pdb_filename.replace('.pdb','')+'_A.pdb'): # Nothing to do:no alt loc atoms
        Alt_num += 1



print(f'Alt_num: {Alt_num}')
# 221 on 3011 dataset of which 24 is on test set