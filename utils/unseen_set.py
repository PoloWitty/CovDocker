"""
desc:	low homology sub test set that is filtered on sequence identity to training set.
author:	Yangzhe Peng
date:	2024/08/01
"""

from openbabel import openbabel
import pandas as pd
import tqdm
import os
import re
from Bio import SeqIO


def pdb2fasta(pdb_filename):
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat('pdb')
    obmol = openbabel.OBMol()
    obConversion.ReadFile(obmol,pdb_filename)
    obConversion.SetOutFormat("fasta")
    out_string = obConversion.WriteString(obmol)
    return out_string

def convert_split_set_to_fasta(dataset_info_filename,output_dir):
    os.makedirs(output_dir,exist_ok=True)
    
    dataset_info = pd.read_csv(dataset_info_filename)
    for split_ in ['train','eval']:
        fasta_filename = os.path.join(output_dir, split_ + ".fasta")
        if os.path.exists(fasta_filename):
            os.remove(fasta_filename)
    
    for split in ['train','valid','test']: 
        split_dataset_info_ = dataset_info[dataset_info['set']==split]
        split_dataset_info = split_dataset_info_.dropna().reset_index()
        assert len(split_dataset_info_) == len(split_dataset_info)
        split_='eval' if split!='train' else split
        if 'filtered' not in dataset_info_filename:
            outputfilename = os.path.join(output_dir, split_ + ".fasta")
        else:
            outputfilename = os.path.join(output_dir, split_ + ".filtered.fasta")
        
        write_mode = 'w' if split=='train' else 'a'
        with open(outputfilename,write_mode) as fp:
            for i, row in tqdm.tqdm(split_dataset_info.iterrows(),total=len(split_dataset_info),desc=f'processing and saving to {outputfilename}'):
                pdb_filename = row['bonded chain']
                fasta_str = pdb2fasta(pdb_filename)
                fp.write(fasta_str+'\n')

def filter_sequences(result_file, test_set_file, output_file, identity_threshold=0.3):
    # Parse MMseqs2 result file
    high_identity_sequences = set()
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.split()
            seq_id, identity = parts[0], float(parts[2])
            if identity >= identity_threshold :
                high_identity_sequences.add(seq_id)

    # Filter test set sequences
    with open(output_file, 'w') as out_f:
        for record in SeqIO.parse(test_set_file, "fasta"):
            if record.id not in high_identity_sequences:
                SeqIO.write(record, out_f, "fasta")
                
def get_unseen_dataset(filtered_filename, orig_data_info_filename, unseen_dataset_info_filename):
    """
        param:
            filtered_filename: str, path to the filtered fasta filename (pdb id info stored in comment line)
            orig_data_info_filename: str, path to the original dataset info file
            unseen_dataset_info_filename: str, path to the result unseen dataset
    """
    # Step 1: Extract PDB IDs from the filtered FASTA file
    filtered_pdb_ids = set()
    with open(filtered_filename, 'r') as filtered_file:
        for line in filtered_file:
            if line.startswith('>'):  # This is a comment line with PDB ID
                matches = re.findall(r'/(\w{4})/', line)
                if len(matches)==2:
                    pdb_id = matches[1]
                    filtered_pdb_ids.add(pdb_id)
    print('Number of filtered PDB IDs:', len(filtered_pdb_ids))

    # Step 2: Read original dataset and filter out seen PDB IDs using pandas
    orig_data = pd.read_csv(orig_data_info_filename)
    unseen_dataset = orig_data[orig_data['pdb_id'].isin(filtered_pdb_ids) | (orig_data['set'] == 'train')]
    unseen_dataset.loc[unseen_dataset['set'] != 'train', 'set'] = 'unseen'
    
    # Step 3: Write unseen data to the result file using pandas
    unseen_dataset.to_csv(unseen_dataset_info_filename, index=False)

if __name__=='__main__':
    
    dataset_info_filename = 'data/processed/dataset.filtered.csv'
    output_dir = 'data/auxiliary/low_homology_test_set'
    
    
    # 0) convert dataset to fasta file format
    convert_split_set_to_fasta(dataset_info_filename,output_dir)
    
    
    # run bash command
    # 1） build mmseqs2 database
    # mmseqs createdb train.filtered.fasta train_filter_set_db
    # mmseqs createdb eval.filtered.fasta eval_filter_set_db
    # 2) run mmseqs2 search
    # mmseqs search eval_filter_set_db train_filter_set_db result tmp --min-seq-id 0.0 --cov-mode 0
    # 3) convert result to m8 format
    # mmseqs convertalis eval_filter_set_db train_filter_set_db result result.m8
    
    
    # # 4) filter on test set
    result_file = output_dir + '/' + "result.m8"
    eval_set_file = output_dir + '/' + "eval.filtered.fasta"
    output_file = output_dir + '/' + "filtered_eval_set.fasta"
    filter_sequences(result_file, eval_set_file, output_file, identity_threshold=0.8)
    
    # 5) output unseen dataset info file
    get_unseen_dataset(output_file, dataset_info_filename, output_dir + "/dataset.filtered.unseen.csv")