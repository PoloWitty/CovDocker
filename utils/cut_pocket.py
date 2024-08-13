"""
desc:	use pymol to cut the pocket from the bonded protein and bonded ligand
author:	Yangzhe Peng
date:	2023/12/24
"""

# referred https://github.com/KyGao/EDP-data-preprocess/blob/a0d362db2d373f9eba37d7f98821c475ea81409a/getstarted/pdbbind_creatingdataset.py#L77

import os
import pdb
import argparse

import tqdm
import pandas as pd
import numpy as np

import multiprocessing
# import seaborn as sns
# import matplotlib.pyplot as plt
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import Select
import scipy
import scipy.spatial

from pymol import cmd
from rdkit import Chem

def gen_pocket(pro, lig, pock_path, cut_off=5):
    cmd.reinitialize()
    cmd.load(pro, 'pro')
    cmd.load(lig, 'lig')
    cmd.select("pock", f"byres (pro within {cut_off} of lig)")
    cmd.save(pock_path, "pock and not sol.")

def rename_old_file():
    old_file = '.' + bonded_pro.split('.')[1][:-len('_protein')] + '_pocket.pdb'
    if os.path.exists(old_file):
        new_file = '.' + bonded_pro.split('.')[1][:-len('_protein')] + '_5Apocket.pdb'
        os.rename(old_file, new_file)

# copy from https://github.com/luwei0917/TankBind/blob/main/tankbind/feature_utils.py
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

# referred TankBind
def select_chain_within_cutoff_to_ligand_v2(x):
    # pdbFile = f"/pdbbind2020/pdbbind_files/{pdb}/{pdb}_protein.pdb"
    # ligandFile = f"/pdbbind2020/renumber_atom_index_same_as_smiles/{pdb}.pdb"
    # toFile = f"{toFolder}/{pdb}_protein.pdb"
    # cutoff = 10
    pdbFile, ligandFile, cutoff, toFile = x
    
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", pdbFile)
    all_res = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    all_atoms = [atom for res in all_res for atom in res.get_atoms()]
    protein_coords = np.array([atom.coord for atom in all_atoms])
    chains = np.array([atom.full_id[2] for atom in all_atoms])

    mol = Chem.MolFromPDBFile(ligandFile)
    lig_coords = mol.GetConformer().GetPositions()
    
    protein_atom_to_lig_atom_dis = scipy.spatial.distance.cdist(protein_coords, lig_coords)

    is_in_contact = (protein_atom_to_lig_atom_dis < cutoff).max(axis=1)
    chains_in_contact = set(chains[is_in_contact])
    
    # save protein chains that belong to chains_in_contact
    class MySelect(Select):
        def accept_residue(self, residue, chains_in_contact=chains_in_contact):
            pdb, _, chain, (_, resid, insertion) = residue.full_id
            if chain in chains_in_contact:
                return True
            else:
                return False

    io=PDBIO()
    io.set_structure(s)
    io.save(toFile, MySelect())
    return toFile
# end copy

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='cut pocket from the bonded protein and bonded ligand')
    parser.add_argument('--dataset_info_filename', type=str, default='./data/processed/dataset.csv')
    parser.add_argument('--cut_off',type=int,default=10,help='the cut off distance between the protein and ligand')
    parser.add_argument('--num_worker', type=int,default=15)
    args = parser.parse_args()

    output_filename = args.dataset_info_filename
    
    df = pd.read_csv(args.dataset_info_filename)
    df['pdb_id'] = df['bonded protein'].apply(lambda x: x.split('/')[-2])
    bonded_pockets = []
    error_pdb = []
    for bonded_pro,bonded_ligand in tqdm.tqdm(zip(df['bonded protein'],df['bonded ligand']),total=len(df),desc='cutting pockets'):
        # example bonded_pro: './data/processed/bonded/5dpa/5dpa_protein.pdb'
        pdb_id = bonded_pro.split('/')[-2]
        if not os.path.exists(bonded_pro):
            error_pdb.append(pdb_id)
            bonded_pockets.append('')
            continue
        bonded_pocket = '.' + bonded_pro.split('.')[1][:-len('_protein')] + f'_{args.cut_off}Apocket.pdb'
        bonded_pockets.append(bonded_pocket)
        # gen_pocket(bonded_pro, bonded_ligand, bonded_pocket, cut_off=args.cut_off)
        # rename_old_file()
    df['bonded pocket'] = bonded_pockets
    print(f'error pdb: {len(error_pdb)}\n{error_pdb}') # 248, the same as reselect error pdb
    
    def wrapper(x):
        idx,row = x
        pdb_file = row['bonded protein']
        ligand_file = row['bonded ligand']
        cutoff = args.cut_off
        to_file = pdb_file.replace('_protein.pdb', f'_chain_within_{cutoff}A.pdb')
        try:
            return select_chain_within_cutoff_to_ligand_v2((pdb_file, ligand_file, cutoff, to_file))
        except:
            return row['pdb_id']

    bonded_chains = []
    error_pdb = []
    with tqdm.trange(len(df), desc='selecting chains') as pbar:
        with multiprocessing.Pool(args.num_worker) as pool:
            for bonded_chain in pool.imap(wrapper, df.iterrows()):
                pbar.update()
                if not bonded_chain.endswith('.pdb'):
                    error_pdb.append(bonded_chain)
                    bonded_chains.append('')
                    continue
                bonded_chains.append(bonded_chain)
    print(f'error pdb: {len(error_pdb)}\n{error_pdb}')
    df['bonded chain'] = bonded_chains
    
    # # save output
    new_order = ['date', 'src', 'pdb_id', 'pre-reactive smiles', 'bonded protein', 'bonded pocket', 'bonded chain', 'bonded ligand', 'bond', 'length', 'products', 'reactants', 'set', 'products_wH', 'products_woAA', 'products_woAAbackbone']  # Specify the desired order of columns
    df = df.reindex(columns=new_order)
    df.to_csv(output_filename, index=False)
