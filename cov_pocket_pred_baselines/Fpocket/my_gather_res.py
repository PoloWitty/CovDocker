"""
desc:	Compute DCC metrics for pocket prediction of fpocket
author:	Yangzhe Peng
date:	2024/05/12
"""


import os
import re
import pandas as pd
import tqdm
import warnings
import argparse

import numpy as np
import torch

from tabulate import tabulate
# from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning


# copy from covDocker/utils/prepare_position_data.py
biopython_parser = PDBParser()

def parse_pdb_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', path)
        rec = structure[0]
    return rec

valid_aa = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','MSE','PHE','PRO','PYL','SER','SEC','THR','TRP','TYR','VAL','ASX','GLX','XAA','XLE']

def get_clean_pocket_res(res_list):
    clean_pocket_res = {}
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        _, _, chain_id, (hetero, idx, insertion) = res.full_id
        if hetero == ' ':
            if res.resname not in valid_aa:
                continue
            res_id = f'{res.resname} {chain_id} {idx}'
            clean_pocket_res[res_id] = 1
    return clean_pocket_res

def extract_receptor_structure(rec, target_res, pocket_res):
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    residues = []
    lengths = []
    pocket_masks = []
    target_res_idx = None
    total_len = 0
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        pocket_mask = []
        count = 0
        invalid_res_ids = []
        chain_residues = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))

            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c) # C that coonect to O in peptide bond CO-NH
                chain_coords.append(np.array(residue_coords))
                res = residue.get_resname()
                chain_residues.append(res)
                _, _, chain_id, (_, idx, _) = residue.full_id
                res_id = f'{res} {chain_id} {idx}'
                if res_id == target_res:
                    target_res_idx = count + total_len
                pocket_mask.append(pocket_res.get(res_id,0))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)

        total_len += count
        lengths.append(count) # chain_res_num
        coords.append(chain_coords) # (chain_res_num, res_atom_num, 3)
        c_alpha_coords.append(np.array(chain_c_alpha_coords)) # (chain_res_num, 3)
        n_coords.append(np.array(chain_n_coords)) # (chain_res_num, 3)
        c_coords.append(np.array(chain_c_coords)) # (chain_res_num, 3)
        residues.append(chain_residues) # (chain_res_num,)
        pocket_masks.append(pocket_mask) # (chain_res_num,)

    coords = [item for sublist in coords for item in sublist]  # [n_residues, n_atoms, 3]
    residues = [item for sublist in residues for item in sublist]  # [n_residues,]
    c_alpha_coords = np.concatenate(c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(c_coords, axis=0)  # [n_residues, 3]
    pocket_masks = np.array([item for sublist in pocket_masks for item in sublist])  # [n_residues,]

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(lengths) == len(c_alpha_coords)
    assert sum(lengths) == len(pocket_masks)
    assert target_res_idx != None
    assert residues[target_res_idx] == target_res.split(' ')[0]
    return target_res_idx, residues, pocket_masks, coords, c_alpha_coords, n_coords, c_coords

# end copy



def get_centers(pdb_filename):
    with open(pdb_filename) as f:
        centers = []
        for line in f:
            if line.startswith('ATOM'):
                center=list(map(float,re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", ' '.join(line.split()[6:]))))[:3]
                centers.append(center)

        centers=np.asarray(centers)
        cg = centers.mean(axis=0)
    return cg


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, default='../../data/processed/dataset.filtered.csv')
    parser.add_argument("--res-dir", type=str, default='./fpocket_res/', help='path to store fpocket output')
    parser.add_argument("--run-id", type=str, default='default-run-id')
    parser.add_argument("--use-wandb",type=int, default=0, help='whether use wandb to store result')
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()
    data_dir = args.res_dir
    dataset_filename=args.dataset_filename
    assert os.path.exists(data_dir)

    if args.only_unseen:
        assert 'unseen' in args.dataset_filename, "only-unseen flag is set, but the dataset_info_file does not contain 'unseen' split"
    
    df = pd.read_csv(dataset_filename)
    
    split = 'test' if not args.only_unseen else 'unseen'
    split_df = df[df['set']==split]
    
    no_rank1_list = []
    pocket_center_pred_list = []
    pocket_center_gt_list = []
    for i,row in tqdm.tqdm(split_df.iterrows(),total=len(split_df),desc='processing'):
        
        # get target pocket center
        pocket_file = '../.'+row['bonded pocket']
        pdb_id = row['pdb_id']
        protein_file = '../.'+row['bonded chain']
        target_res = ' '.join(row['bond'].split('-')[0].split(' ')[1:])  # example: OD2 ASP D 229-CAF 3U3 D 501
        
        protein = parse_pdb_from_path(protein_file)
        protein._id = pdb_id
        pocket = parse_pdb_from_path(pocket_file)
        pocket._id = pdb_id
        pocket_res = get_clean_pocket_res(pocket.get_residues())
        
        target_res_idx, residues, pocket_masks, coords, abs_c_alpha_coords, abs_n_coords, abs_c_coords = extract_receptor_structure(protein, target_res, pocket_res)
        
        pocket_coord_target = abs_c_alpha_coords * np.expand_dims(pocket_masks, axis=-1) # (n, 3)
        pocket_center_target = pocket_coord_target.sum(axis=0) / pocket_masks.sum(axis=0) # (3,)
        
        protein_center = abs_c_alpha_coords.mean(axis=0)
        
        # get fpocket predict pocket center
        # pred_res_filename = f'{data_dir}/{pdb_id}_chain_within_10A_out/pockets/pocket1_vert.pqr'
        pred_res_filename = f'{data_dir}/{pdb_id}_chain_within_10A_out/pockets/pocket1_atm.pdb'
        
        try:
            rank1_center = get_centers(pred_res_filename)
        except KeyboardInterrupt:
            exit()
        except:
            print("No rank1, use protein mean center as pocket center")
            no_rank1_list.append(pdb_id)
            rank1_cneter = protein_center
        
        pocket_center_gt_list.append(pocket_center_target)
        pocket_center_pred_list.append(rank1_center)
    
    pocket_center_gt = torch.from_numpy(np.array(pocket_center_gt_list))
    pocket_center_pred = torch.from_numpy(np.array(pocket_center_pred_list))
    pocket_pairwise_dist = torch.nn.functional.pairwise_distance(pocket_center_pred, pocket_center_gt, p=2)

    DCC_3 = (pocket_pairwise_dist < 3).sum().item() / len(pocket_pairwise_dist)
    DCC_4 = (pocket_pairwise_dist < 4).sum().item() / len(pocket_pairwise_dist)
    DCC_5 = (pocket_pairwise_dist < 5).sum().item() / len(pocket_pairwise_dist)
    
    table = {
        'DCC_3:': DCC_3,
        'DCC_4:': DCC_4,
        'DCC_5:': DCC_5
    }
    print("No rank1 list:", no_rank1_list)
    
    
    if args.use_wandb:
        import wandb
        wandb.init(
            project="res_reactive_site",
            config = args
        )
        wandb.log(table)

    
    # use tabulate to show the result
    table_ = {k:[v] for k,v in table.items()}
    print(tabulate(table_, headers="keys"))