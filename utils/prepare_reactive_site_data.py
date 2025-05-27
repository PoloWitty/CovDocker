"""
desc:	prepare data for reactive site prediction
author:	Yangzhe Peng
date:	2024/05/12
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
from rdkit import Chem
from rdkit.Chem import AllChem

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
    res_ids = []
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
                res_ids.append(res_id)
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
    assert len(res_ids) == len(residues)
    return target_res_idx, residues, pocket_masks, coords, c_alpha_coords, n_coords, c_coords, res_ids

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol


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
    
    target_res_idx, residues, pocket_masks, coords, abs_c_alpha_coords, abs_n_coords, abs_c_coords, res_ids = extract_receptor_structure(protein, target_res, pocket_res)
    
    smiles = row['pre-reactive smiles']
    ligand_mol = Chem.MolFromSmiles(smiles)
    ligand_atoms = [atom.GetSymbol() for atom in ligand_mol.GetAtoms()]
    
    try:
        # init the ligand coordinates using rdkit
        ligand_mol.RemoveAllConformers()
        ligand_mol = generate_conformation(ligand_mol)
        ligand_coords = ligand_mol.GetConformer().GetPositions().tolist()
    except KeyboardInterrupt:
        exit()
    except:
        return False,pdb_id
    

    return True, pickle.dumps(
        {
            "pre_reactive_ligand_atoms": ligand_atoms,
            "pre_reactive_ligand_coords": ligand_coords,
            "protein_CA_coords": abs_c_alpha_coords,
            "residue": residues,
            "pocket_mask": pocket_masks,
            "pdbid": pdb_id,
            "smiles": smiles,
            "target": target_res_idx,
            "res_ids": res_ids
        },
        protocol=-1,
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_info_file", type=str, default="./data/processed/dataset.filtered.random_split.filtered.csv")
    parser.add_argument("--save_dir", type=str, default="./data/processed/dataset/reactive_site/")
    parser.add_argument("--num_worker", type=int, default=15)
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.only_unseen:
        assert 'unseen' in args.dataset_info_file, "only-unseen flag is set, but the dataset_info_file does not contain 'unseen' split"
    
    dataset_info = pd.read_csv(args.dataset_info_file)
    
    outpath = args.save_dir
    os.makedirs(outpath, exist_ok=True)
    failed_pdb = []
    splits = ['train','valid','test'] if not args.only_unseen else ['unseen']
    for split in splits:
        split_dataset_info = dataset_info[dataset_info['set']==split]
        # split_dataset_info = split_dataset_info.dropna().reset_index()
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
