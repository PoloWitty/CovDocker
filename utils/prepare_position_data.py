"""
desc:	prepare pocket data for position model training
author:	Yangzhe Peng
date:	2024/01/14
"""

import os
import re
import json
import glob
import pdb
import argparse
import warnings

import numpy as np

import multiprocessing
import tqdm
import pickle
import pandas as pd
import lmdb

# from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from unimol_tools import UniMolRepr

biopython_parser = PDBParser()
clf = UniMolRepr(data_type='molecule', remove_hs=False)

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

# referred diffdock
def get_side_chain_vecs(c_alpha_coords, n_coords, c_coords):
    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    side_chain_vecs = np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1)
    return side_chain_vecs

def get_ligand_features(smiles_list):
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs = True)
    return unimol_repr['cls_repr'], unimol_repr['atomic_coords'], unimol_repr['atomic_reprs'], unimol_repr['atomic_symbol']

def parse(x):
    idx, row = x
    pocket_file = row['bonded pocket']
    pdb_id = pocket_file.split('/')[-2] # example: ./data/processed/bonded/4WSJ/4WSJ_pocket.pdb
    protein_file = row['bonded chain']
    target_res = ' '.join(row['bond'].split('-')[0].split(' ')[1:])  # example: OD2 ASP D 229-CAF 3U3 D 501
    
    protein = parse_pdb_from_path(protein_file)
    protein._id = pdb_id
    pocket = parse_pdb_from_path(pocket_file)
    pocket._id = pdb_id
    pocket_res = get_clean_pocket_res(pocket.get_residues())
    
    target_res_idx, residues, pocket_masks, coords, abs_c_alpha_coords, abs_n_coords, abs_c_coords = extract_receptor_structure(protein, target_res, pocket_res)
    side_chain_vecs = get_side_chain_vecs(abs_c_alpha_coords, abs_n_coords, abs_c_coords) # n_rel_pos, c_rel_pos
    
    smiles = row['pre-reactive smiles']
    ligand_feat = ligand_feats[idx]
    

    return True, pickle.dumps(
        {
            "smiles": smiles,
            "ligand_feat": ligand_feat,
            "CA_coordinates": abs_c_alpha_coords,
            "side_chain_vec": side_chain_vecs,
            "residue": residues,
            "pocket_mask": pocket_masks,
            "pdbid": pdb_id,
            "target": target_res_idx
        },
        protocol=-1,
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_info_file", type=str, default="./data/processed/dataset.filtered.random_split.filtered.csv")
    parser.add_argument("--save_dir", type=str, default="./data/processed/dataset/position/")
    parser.add_argument("--num_worker", type=int, default=15)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    dataset_info = pd.read_csv(args.dataset_info_file)
    
    # use unimol-tools to get ligand features first
    ligand_cls_feats, coords, ligand_atom_reprs, syms = get_ligand_features(dataset_info['pre-reactive smiles'].tolist())
    ligand_feats = []
    for ligand_cls_feat, ligand_atom_repr, sym in zip(ligand_cls_feats, ligand_atom_reprs, syms):
        nonH_mask = [s != 'H' for s in sym]
        ligand_atom_repr = ligand_atom_repr[nonH_mask] # drop H atoms
        ligand_feat = np.vstack([ligand_cls_feat, ligand_atom_repr])
        ligand_feats.append(ligand_feat)

    outpath = args.save_dir
    os.makedirs(outpath, exist_ok=True)
    failed_pdb = []
    for split in ['train','valid','test']:
        split_dataset_info = dataset_info[dataset_info['set']==split]
        split_dataset_info = split_dataset_info.dropna().reset_index()
        outputfilename = os.path.join(outpath, split + ".lmdb")
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(10e9),
        )
        txn_write = env_new.begin(write=True)
        dataset_idx = 0
        
        with tqdm.trange(len(split_dataset_info), desc=f'saving to {outputfilename}') as pbar:
            with multiprocessing.Pool(args.num_worker) as pool:
                for flag, res in pool.imap(parse, split_dataset_info.iterrows()):
                    if not flag :
                        failed_pdb.append(res)
                        continue
                    txn_write.put(f"{dataset_idx}".encode("ascii"), res)
                    dataset_idx += 1
                    pbar.update()
        txn_write.commit()
        env_new.close()
    print(f"failed pdb {len(failed_pdb)}\n {failed_pdb}")
