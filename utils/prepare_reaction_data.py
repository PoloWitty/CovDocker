"""
desc:	Prepare reaction data for the reaction prediction model
author:	Yangzhe Peng
date:	2024/02/29
"""


import os
import re
import pdb
import time
import argparse
import requests
import pickle
import random

import tqdm
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem
from rdkit.Chem import AllChem


OH_aa = ['SER','THR','ASP','GLU','TYR']
# def get_aa_type():
#     aa_type = {}
#     for aa in ['ARG','HIS','LYS']:# positive
#         aa_type[aa] = 'Ti'
#     for aa in ['ASP','GLU']:# negative
#         aa_type[aa] = 'Te'
#     for aa in ['SER','THR','ASN','GLN']:# uncharged
#         aa_type[aa] = 'Se'
#     for aa in ['CYS','SEC','GLY','PRO']:# special
#         aa_type[aa] = 'Bi'
#     for aa in ['ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP']:# hydrophobic
#         aa_type[aa] = 'Po'
#     return aa_type
# aa_type = get_aa_type()

def _download_file(url, destination):
    try:
        response = requests.get(url,timeout=5)
    except:
        time.sleep(0.5) # delay
        return False
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        # print(f"File {url} downloaded successfully!")
        return True
    else:
        # print(f"Failed to download file {url}.")
        return False

def download_file_in_subfolder(url, destination):
    '''
    download one file from the subfolder of the url
    '''
    try:
        response = requests.get(url,timeout=5)
    except:
        time.sleep(0.5) # delay
        return None
    if response.status_code == 200:
        subfolder_url = response.url
        
        soup = BeautifulSoup(response.content, 'html.parser')
        file_urls = [subfolder_url + link['href'] for link in soup.find_all('a') if link.text != ' Parent Directory']
        if len(file_urls) == 0:
            print(f"No file found in the subfolder: {subfolder_url}.")
            return None
        
        target_pattern = f'{pdb_id.lower()}_{ligand_het}_'+r'\d'+f'_{ligand_chain}_{ligand_res_id}'
        target_url = [url for url in file_urls if len(re.findall(target_pattern, url))==1]
        if len(target_url) == 0:
            print(f'No fully matched file found in the subfolder: {subfolder_url}, use {file_urls[0]} instead.')
            target_url = [file_urls[0]]
        elif len(target_url) > 1:
            # print(f'target_url: {target_url}, url: {url}, target_pattern: {target_pattern}')
            target_url = [target_url[0]]
        target_url = target_url[0]
        file_name = target_url.split('/')[-1]
        file_destination = os.path.join(destination, file_name)
        is_success = _download_file(target_url, file_destination)
        if is_success:
            return file_destination
        else:
            return None
    else:
        print(f"Failed to access the subfolder for {url}.")
        return None

# from Chemformer.molbart.modules.atom_assign import remove_AA_backbone_from_product_smiles
def remove_AA_backbone_from_product_smiles(product_smiles: str, aa_smiles: str, AA_backbone_smiles: str = 'NCC(=O)O') -> str:
    if product_smiles == "":
        return ""
    # the default parameter of AA_backbone_smiles is the smiles of GLY whose R group is H
    product_mol = Chem.MolFromSmiles(product_smiles); AA_mol = Chem.MolFromSmiles(aa_smiles)
    if product_mol is None or AA_mol is None:
        return ""
    
    AA_backbone_idx = list(AA_mol.GetSubstructMatch(Chem.MolFromSmarts(AA_backbone_smiles)))
    
    AA_idx = list(product_mol.GetSubstructMatch(AA_mol))
    if AA_idx == []:
        return ""
    
    edit_mol = Chem.EditableMol(product_mol)
    atoms_to_remove = [AA_idx[idx] for idx in AA_backbone_idx]
    edit_mol.BeginBatchEdit()
    for atom in atoms_to_remove:
        edit_mol.RemoveAtom(atom)
    edit_mol.CommitBatchEdit()
    product_mol_woBackbone = edit_mol.GetMol()
    Chem.SanitizeMol(product_mol_woBackbone)
    return Chem.MolToSmiles(product_mol_woBackbone)

def _get_post_reactive_ligand(path:str):
    atoms = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            content = line.split()
            if len(content) > 5:# not TER or END line
                atoms.append(line[12:16].strip())
    return atoms

def parse_bond_record(bond_record):
    atom1 = bond_record[:3]
    atom2 = bond_record[3:6]
    bond = bond_record[6:9]
    return atom1, atom2, bond

def _delete_one_atom_in_bond_block(bond_block,deleted_idx):
    '''
        delete_idx should start from 0
    '''
    bond_block_list = bond_block.split('\n')[:-1]
    new_bond_block = ''
    for i in range(len(bond_block_list)):
        atom1,atom2,bond = parse_bond_record(bond_block_list[i])
        # deleted bond connected to the deleted_idx
        if int(atom1) == deleted_idx+1 or int(atom2) == deleted_idx+1:
            continue
        # substrace 1 for the rest of the atoms
        atom1 = int(atom1)-1 if int(atom1) > deleted_idx+1 else int(atom1)
        atom2 = int(atom2)-1 if int(atom2) > deleted_idx+1 else int(atom2)
        bond = int(bond)
        new_bond_block += f'{atom1:3d}{atom2:3d}  {bond:1d}  0\n'
    return new_bond_block

def _delete_one_atom_in_atom_block(atom_block,deleted_idx):
    '''
        delete_idx should start from 0
    '''
    atom_block_list = atom_block.split('\n')[:-1]
    return '\n'.join([atom_block_list[i] for i in range(len(atom_block_list)) if i != deleted_idx]) + '\n'
    
def _add_one_atom_in_atom_block(atom_block,sym):
    '''
        returned added_index start from 0
    '''
    atom_block += '{0:10.4f}{1:10.4f}{2:10.4f} {3:4s}0  0  0  0  0  0  0  0  0  0  0  0\n'.format(float(0),float(0),float(0),sym)
    added_index = len(atom_block.split('\n'))-2
    return atom_block, added_index

def _add_one_atom_in_bond_block(bond_block,atom1,atom2,bond):
    '''
        atom1 and atom2 should start from 0
    '''
    bond_block += f'{atom1+1:3d}{atom2+1:3d}  {bond:1d}  0\n'
    return bond_block

def _find_connected_H_atoms(bond_block,atoms,atom_idx):
    '''
        atoms is the list of atom names
        atom_idx is the atom to be searched, and should start from 0
    '''
    bond_block_list = bond_block.split('\n')[:-1]
    H_index = [idx for idx in range(len(atoms)) if atoms[idx].startswith('H') and idx != atom_idx]
    res = []
    for i in range(len(bond_block_list)):
        atom1,atom2,bond = parse_bond_record(bond_block_list[i])
        atom1 = int(atom1)-1; atom2 = int(atom2)-1; bond = int(bond)
        if bond == 1:
            if atom1 == atom_idx and atom2 in H_index:
                res.append(atom2)
            elif atom2 == atom_idx and atom1 in H_index:
                res.append(atom1)
    return res

def _get_valence_from_bond_block(bond_block,atom_idx):
    '''
        atom_idx should start from 0
    '''
    bond_block_list = bond_block.split('\n')[:-1]
    valence = 0; double_or_triple_bond_idx = None
    for i in range(len(bond_block_list)):
        atom1,atom2,bond = parse_bond_record(bond_block_list[i])
        atom1 = int(atom1)-1; atom2 = int(atom2)-1; bond = int(bond)
        if atom1 == atom_idx or atom2 == atom_idx:
            valence += bond
            if bond == 2 or bond == 3:
                double_or_triple_bond_idx = atom1 if atom1 != atom_idx else atom2
    return valence, double_or_triple_bond_idx
    
def _substract_one_valence_in_bond_block(bond_block, atom1_idx, atom2_idx):
    '''
        atom1_idx and atom2_idx should start from 0
    '''
    bond_block_list = bond_block.split('\n')[:-1]
    new_bond_block = ''
    for i in range(len(bond_block_list)):
        atom1_,atom2_,bond = parse_bond_record(bond_block_list[i])
        atom1_ = int(atom1_)-1; atom2_ = int(atom2_)-1; bond = int(bond)
        if (atom1_ == atom1_idx and atom2_ == atom2_idx) or (atom1_ == atom2_idx and atom2_ == atom1_idx):
            bond -= 1
        new_bond_block += f'{atom1_+1:3d}{atom2_+1:3d}  {bond:1d}  0\n'
    return new_bond_block


def _add_number_to_bond_block(bond_block, number):
    '''
        this function should only be used on aa bond block
    '''
    bond_block_list = bond_block.split('\n')[:-1]
    new_bond_block = ''
    for i in range(len(bond_block_list)):
        atom1,atom2,bond = parse_bond_record(bond_block_list[i])
        atom1 = int(atom1)+number; atom2 = int(atom2)+number; bond = int(bond)
        new_bond_block += f'{atom1:3d}{atom2:3d}  {bond:1d}  0\n'
    return new_bond_block

def _build_bond_block(info:dict):
    atoms = info['_chem_comp_atom.atom_id']
    bond_map = {
        'SING': '1',
        'DOUB': '2',
        'TRIP': '3',
        'QUAD': '4',
    }
    bond_block = ''
    for atom1,atom2,bond in zip(info['_chem_comp_bond.atom_id_1'],info['_chem_comp_bond.atom_id_2'],info['_chem_comp_bond.value_order']):
        atom1 = atoms.index(atom1)
        atom2 = atoms.index(atom2)
        bond = bond_map[bond]
        bond_block += '{0:3d}{1:3d}  {2:s}  0\n'.format(atom1+1,atom2+1,bond) # index start from 1 in sdf file
    return bond_block

def _build_atom_block_and_charges(info:dict):
    atoms = info['_chem_comp_atom.atom_id']
    atom_block = ''
    charges = {}
    for idx, (x,y,z,sym,charge) in enumerate(zip(info['_chem_comp_atom.pdbx_model_Cartn_x_ideal'],info['_chem_comp_atom.pdbx_model_Cartn_y_ideal'],info['_chem_comp_atom.pdbx_model_Cartn_z_ideal'],info['_chem_comp_atom.type_symbol'],info['_chem_comp_atom.charge'])):
        if x == '?':
            # atom_block += '{0:>10s}{1:>10s}{2:>10s} {3:s}   0  0  0  0  0  0  0  0  0  0  0  0\n'.format(x,y,z,sym)
            atom_block += '{0:10.4f}{1:10.4f}{2:10.4f} {3:4s}0  0  0  0  0  0  0  0  0  0  0  0\n'.format(float(0),float(0),float(0),sym)
        else:
            atom_block += '{0:10.4f}{1:10.4f}{2:10.4f} {3:4s}0  0  0  0  0  0  0  0  0  0  0  0\n'.format(float(x),float(y),float(z),sym)
        if charge != '0':
            charges[atoms[idx]] = charge
    return atom_block, charges

def get_aa_sdf(aa_mmcif_dir:str, aa_name:str):
    aa_mmcif_path = f'{aa_mmcif_dir}/{aa_name}.cif'
    info = MMCIF2Dict(aa_mmcif_path)
    if not (('_chem_comp.id' in info) and ('_chem_comp_atom.atom_id' in info) and ('_chem_comp_bond.atom_id_1' in info) and ('_chem_comp_bond.atom_id_2' in info) and ('_chem_comp_bond.value_order' in info)):
        return None

    # build bond and atom block
    bond_block = _build_bond_block(info)
    atom_block, charges = _build_atom_block_and_charges(info)
    atoms = info['_chem_comp_atom.atom_id']

    # build charge block
    charge_blocks = []
    for atom,charge in charges.items():
        if atom in atoms:
            charge_blocks.append("{0:4d}{1:>4s}".format(atoms.index(atom)+1,charge))

    charge_block = "M  CHG{0:3d}".format(len(charge_blocks)) + ''.join(charge_blocks) + '\n' if len(charge_blocks) > 0 else ''

    # build mol_block
    mol_block = str(info['_chem_comp.id'][0]) + '    \n'
    mol_block += '     Manually          3D\n'    
    mol_block += '\n'
    mol_block += '{0:3d}{1:3d}  0  0  0  0  0  0  0  0999 V2000\n'.format(len(atom_block.split("\n"))-1,len(bond_block.split("\n"))-1) # atom num & bond num
    mol_block += atom_block
    mol_block += bond_block
    mol_block += charge_block
    mol_block += 'M  END\n'
    mol_block += '$$$$\n'
    return mol_block

def _remove_H_in_smiles(smiles):
    if type(smiles) == float or smiles == '': # float(nan) or empty string
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)
    
def convert_to_sdf(mmcif_path:str, pdb_id, bond_atom, bond_aa, bond_aa_atom, represent_aa_as_sym = False, aa_sym = 'Ti'):
    """
    Converts an MMCIF file to an SDF file format.
    """
    info = MMCIF2Dict(mmcif_path)
    if not (('_chem_comp.id' in info) and ('_chem_comp_atom.atom_id' in info) and ('_chem_comp_bond.atom_id_1' in info) and ('_chem_comp_bond.atom_id_2' in info) and ('_chem_comp_bond.value_order' in info)):
        error_info['mmcif info incomplete'].append(pdb_id)
        return None
        
    # build bond block
    bond_block = _build_bond_block(info)
    
    # build atom block and charge_block
    atom_block, charges = _build_atom_block_and_charges(info)

    # compare to post_reactive ligand and process bond_block, atom_block and charge_block
    ## check with related post-reactive pdb file
    atoms = info['_chem_comp_atom.atom_id']
    noH_atoms = [atom for atom in atoms if not atom.startswith('H')]
    ligand_path = f'./data/processed/bonded/{pdb_id.upper()}/{pdb_id.upper()}_ligand.pdb'
    post_reactive_atoms = _get_post_reactive_ligand(ligand_path)
    post_reactive_atoms = [atom for atom in post_reactive_atoms if not atom.startswith('H')]

    # (step 1) delete extra atoms in CCD molecule
    ## delete noH atoms
    if set(post_reactive_atoms) != set(noH_atoms):
        if set(post_reactive_atoms).issubset(noH_atoms):
            diff_atoms = list(set(noH_atoms) - set(post_reactive_atoms))
            if len(diff_atoms) > 0 and len(diff_atoms) < len(noH_atoms):
                for diff_atom in sorted(diff_atoms,reverse=True):
                    diff_atom_idx = atoms.index(diff_atom)
                    H_idxes = _find_connected_H_atoms(bond_block,atoms,diff_atom_idx)
                    if len(H_idxes)!=0:
                        for H_idx in sorted(H_idxes,reverse=True):
                            atom_block = _delete_one_atom_in_atom_block(atom_block,H_idx)
                            bond_block = _delete_one_atom_in_bond_block(bond_block,H_idx)
                            atoms.pop(H_idx) # should pop from the last one
                        diff_atom_idx = atoms.index(diff_atom)
                    atom_block = _delete_one_atom_in_atom_block(atom_block,diff_atom_idx)
                    bond_block = _delete_one_atom_in_bond_block(bond_block,diff_atom_idx)
                    atoms.pop(diff_atom_idx)
        else:
            print(f'<<<<warning>>>> post-reactive atoms is not the subset of CCD atoms ({pdb_id}, {bond_aa}) (atoms: {noH_atoms}, post_reactive_atoms: {post_reactive_atoms})\n')
            error_info['differ from post-reactive atoms'].append(pdb_id)
            return None

    noH_atoms = [atom for atom in atoms if not atom.startswith('H')]
    assert set(post_reactive_atoms) == set(noH_atoms)

    ## delete one H atom connected to bond atom
    H_idxes = _find_connected_H_atoms(bond_block,atoms,atoms.index(bond_atom))
    if len(H_idxes) > 0:
        H_idx = H_idxes[0] # only delete one connected H atom
        atom_block = _delete_one_atom_in_atom_block(atom_block,H_idx)
        bond_block = _delete_one_atom_in_bond_block(bond_block,H_idx)
        atoms.pop(H_idx)
        
    ## (patch) open the double or triple bond and become single or double bond and add one hydrogen
    valence, double_or_triple_bond_idx = _get_valence_from_bond_block(bond_block,atoms.index(bond_atom))
    if bond_atom.startswith('C') and valence == 4:
        # substract one valence
        bond_block = _substract_one_valence_in_bond_block(bond_block,atoms.index(bond_atom),double_or_triple_bond_idx)

        # add one hydrogen for double_or_triple_bond atom
        atom_block,added_idx = _add_one_atom_in_atom_block(atom_block,'H')
        atoms.append('H')
        bond_block = _add_one_atom_in_bond_block(bond_block,added_idx,double_or_triple_bond_idx,1)
    elif bond_atom.startswith('O') and valence == 2:
        # substract one valence
        bond_block = _substract_one_valence_in_bond_block(bond_block,atoms.index(bond_atom),double_or_triple_bond_idx)
        

    # (step 2) add aa 
    if represent_aa_as_sym:
        atom_block,aa_idx = _add_one_atom_in_atom_block(atom_block,aa_sym)
        atoms.append(aa_sym)
        bond_block = _add_one_atom_in_bond_block(bond_block,aa_idx,atoms.index(bond_atom),1)
    else:
        ## prepare aa sdf blocks
        aa_mmcif_path = f'./data/amino_acids/{bond_aa}.cif'
        ### convert mmCIF to sdf
        aa_info = MMCIF2Dict(aa_mmcif_path)
        aa_bond_block = _build_bond_block(aa_info)
        aa_atom_block,aa_charges = _build_atom_block_and_charges(aa_info)
        aa_atoms = aa_info['_chem_comp_atom.atom_id']
        ### delete one H atom connected to bond_aa_atom
        aa_H_idxes = _find_connected_H_atoms(aa_bond_block,aa_atoms,aa_atoms.index(bond_aa_atom))
        if not (len(aa_H_idxes) >= 1 and aa_charges == {}):
            if len(aa_H_idxes)==0 and bond_aa == 'HIS':
                charges['ND1'] = '1'
            elif bond_aa == 'ARG' and bond_aa_atom == 'NH1':
                charges['NH2'] = '1' # NH1 connect to aa, NH2 have a negative charge
                aa_atom_block = _delete_one_atom_in_atom_block(aa_atom_block,aa_H_idxes[0])
                aa_bond_block = _delete_one_atom_in_bond_block(aa_bond_block,aa_H_idxes[0])
                aa_atoms.pop(aa_H_idxes[0])
            elif bond_aa == 'MET' and bond_aa_atom == 'SD':
                charges['SD'] = '1'
        else:
            aa_atom_block = _delete_one_atom_in_atom_block(aa_atom_block,aa_H_idxes[0])
            aa_bond_block = _delete_one_atom_in_bond_block(aa_bond_block,aa_H_idxes[0])
            aa_atoms.pop(aa_H_idxes[0])
        ### add ligand atom number to aa bond block
        aa_bond_block = _add_number_to_bond_block(aa_bond_block,len(atoms))
        ## add aa to atom_block and bond_block
        atom_block += aa_atom_block
        bond_block += aa_bond_block
        bond_block =  _add_one_atom_in_bond_block(bond_block,aa_atoms.index(bond_aa_atom)+len(atoms),atoms.index(bond_atom),1)
        atoms += aa_atoms

    # build charge block
    charge_blocks = []
    for atom,charge in charges.items():
        if atom in atoms:
            charge_blocks.append("{0:4d}{1:>4s}".format(atoms.index(atom)+1,charge))

    ## (Patch) if connected to 'B', add one more entry to charge block
    if bond_atom.startswith('B') and (len(bond_atom)==1 or bond_atom[1].isdigit() or bond_atom == 'BOR'):
        charge_atom = "{0:4d}{1:>4s}".format(atoms.index(bond_atom)+1,'-1')
        if charge_atom not in charge_blocks:
            charge_blocks.append(charge_atom)

    charge_block = "M  CHG{0:3d}".format(len(charge_blocks)) + ''.join(charge_blocks) + '\n' if len(charge_blocks) > 0 else ''

    # build mol_block
    mol_block = str(info['_chem_comp.id'][0]) + '    \n'
    mol_block += '     Manually          3D\n'    
    mol_block += '\n'
    mol_block += '{0:3d}{1:3d}  0  0  0  0  0  0  0  0999 V2000\n'.format(len(atom_block.split("\n"))-1,len(bond_block.split("\n"))-1) # atom num & bond num
    mol_block += atom_block
    mol_block += bond_block
    mol_block += charge_block
    mol_block += 'M  END\n'
    mol_block += '$$$$\n'

    return mol_block

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_info_filename', type=str, default='./data/processed/examples.pocket_with_smiles.random_split.csv')
    parser.add_argument('--addtional_aa_rxn_filename', type=str, default='')
    args = parser.parse_args()
    # Load the example.pocket.csv file
    df = pd.read_csv(args.dataset_info_filename)
    smiles_row = [];products_woAA = []
    error_info = defaultdict(list)
    debug_info = defaultdict(list)
    destination = "./ccif/"
    os.makedirs(destination,exist_ok=True)
    df.fillna('',inplace=True)

    for idx,row in tqdm.tqdm(df.iterrows(),total=len(df)):
        # parse bond
        bond = row['bond'] # {name1} {res_name1} {chain_id1} {res_seq1}-{name2} {res_name2} {chain_id2} {res_seq2}
        aa_atom, aa_het, aa_chain, aa_res_id = bond.split('-')[0].split(' ')
        ligand_atom, ligand_het, ligand_chain, ligand_res_id = bond.split('-')[1].split(' ')
        pdb_id = row['bonded protein'].split('/')[-2]
        # (patch) unknow error in original pdb file
        if (aa_het == 'LEU' and ligand_het == 'PJE') or (aa_atom == 'CD' and aa_het ==  'GLU'):
            aa_atom = 'SG'; aa_het = 'CYS'
        if aa_atom == 'OE1' and aa_het == 'GLU':
            aa_atom = 'OE2'
        if aa_atom == 'OD1' and aa_het == 'ASP':
            aa_atom = 'OD2'
        if aa_het not in  ['VAL','ALA'] and aa_atom in ['N','CA','C','O','OXT']: # bond_aa_atom should not in backbone atoms
            if pdb_id == '4EHM' or aa_het == 'PRO':
                pass
            else:
                error_info['reaction aa error'].append(pdb_id)
                smiles_row.append('');products_woAA.append('')
                continue
        
        # already have smiles
        if row['products_wH']!='':
            smiles_row.append(row['products_wH'])
            continue
        if row['products_woAA']!='':
            products_woAA.append(row['products_woAA']) 
            continue
        
        # download mmcif file
        url = f"http://ligand-expo.rcsb.org/files/{ligand_het[0]}/{ligand_het}/ccif"
        mmcif_path = download_file_in_subfolder(url, destination)
        if mmcif_path == None:
            error_info['download error'].append(pdb_id)
            smiles_row.append('');products_woAA.append('')
            continue
        
        # convert mmcif to sdf blocks
        sdf_blocks = convert_to_sdf(mmcif_path,pdb_id=pdb_id,bond_atom=ligand_atom,bond_aa=aa_het,bond_aa_atom=aa_atom)
        if sdf_blocks == None:
            smiles_row.append('')
            continue
        
        sdf_blocks_woAA = convert_to_sdf(mmcif_path,pdb_id=pdb_id,bond_atom=ligand_atom,bond_aa=aa_het,bond_aa_atom=aa_atom,represent_aa_as_sym=True,aa_sym='Ti')
        if sdf_blocks_woAA == None:
            products_woAA.append('')
            continue
        
        # Convert sdf_blocks to RDKit molecule object
        mol = Chem.MolFromMolBlock(sdf_blocks, sanitize=True, removeHs=False,strictParsing=False)
        if mol == None:
            error_info['sdf to mol error'].append(pdb_id)
            smiles_row.append('')
            continue
        
        mol_woAA = Chem.MolFromMolBlock(sdf_blocks_woAA, sanitize=True, removeHs=False,strictParsing=False)
        if mol_woAA == None:
            error_info['sdf to mol error'].append(pdb_id)
            products_woAA.append("")
            continue
        
        # Convert RDKit molecule object to SMILES expression
        smiles = Chem.MolToSmiles(mol)
        smiles_row.append(smiles)
        
        smiles_woAA = Chem.MolToSmiles(mol_woAA)
        products_woAA.append(smiles_woAA)
        
        
        time.sleep(0.3)  # Delay for 300 milliseconds
        
    df['products_wH'] = smiles_row
    df['products_woAA'] = products_woAA
    
    for k,v in error_info.items():
        print(f'{k}: {len(v)}')
        print(v)
        
    for k,v in debug_info.items():
        print(f'{k}: {len(v)}')
        print(v)
    
    aa_smiles = {}
    for aa in ['ALA','ARG','ASN','ASP','CYS','GLU','GLY','HIS','ILE','LEU','LYS','MET','PRO','SER','THR','TYR','VAL']:
        aa_mmcif_dir = './data/amino_acids'
        aa_sdf = get_aa_sdf(aa_mmcif_dir,aa)
        mol = Chem.MolFromMolBlock(aa_sdf, sanitize=True, removeHs=False,strictParsing=False)
        smiles = Chem.MolToSmiles(mol)
        aa_smiles[aa] = smiles
    print(aa_smiles)
    
    # # save reactants and products
    reactants = []
    for i,row in tqdm.tqdm(df.iterrows(),total=len(df),desc='save reactants and products'):
        if row['products_wH'] != '' and (not ('[Ru' in row['pre-reactive smiles'])):
            bond = row['bond']
            aa_atom, aa_het, aa_chain, aa_res_id = bond.split('-')[0].split(' ')
            aa_smile = aa_smiles[aa_het]
            ligand_smile = row['pre-reactive smiles']
            assert len(ligand_smile.split('.')) == 1,f'Expect pre reactive ligand only contain one ligand, but got {ligand_smile}'
            reactants.append(f'{_remove_H_in_smiles(aa_smile)}.{_remove_H_in_smiles(ligand_smile)}')
        else:
            reactants.append('')
    df['reactants'] = reactants
    
    # remove aa backbone
    
    # old
    # def wrapper_remove_AA_backbone_from_product_smiles(smiles:str, aa_smiles:str):
    #     if smiles == '' or '->' in smiles:
    #         return ''
    #     else:
    #         try:
    #             return remove_AA_backbone_from_product_smiles(smiles, aa_smiles=aa_smiles)
    #         except:
    #             return ''
    # tqdm.tqdm.pandas(total=len(df), desc='remove AA backbone from products column')
    # df['products_woAAbackbone'] = df['products'].progress_apply(wrapper_remove_AA_backbone_from_product_smiles)
    
    # new
    col_products_woAAbackbone = []
    for i,row in tqdm.tqdm(df.iterrows(), total=len(df), desc='remove AA backbone from products column'):
        if row['products'] == '' or '->' in row['products']:
            col_products_woAAbackbone.append('')
        else:
            try:
                product_woAAbackbone = remove_AA_backbone_from_product_smiles(row['products'], aa_smiles=row["reactants"].split(".")[0] )
                col_products_woAAbackbone.append(product_woAAbackbone)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                col_products_woAAbackbone.append('')
    df['products_woAAbackbone'] = col_products_woAAbackbone
    
    # drop H 
    df['products'] = df['products_wH'].apply(_remove_H_in_smiles)
    df['products_woAA'] = df['products_woAA'].apply(_remove_H_in_smiles)
    df['products_woAAbackbone'] = df['products_woAAbackbone'].apply(_remove_H_in_smiles)
    
    # Add more aa  to reaction related columns
    if args.addtional_aa_rxn_filename != '':
        addtional_aa_reaction = pickle.load(open(args.addtional_aa_rxn_filename,'rb'))
        sample_size = 50000
        sampled_addtional_aa_reaction = random.sample(addtional_aa_reaction, sample_size)
        new_rows = {
                'src': ['uspto_sep']*sample_size, 
                'reactants':[ sample['reactants'] for sample in sampled_addtional_aa_reaction],
                'products': [sample['products'] for sample in sampled_addtional_aa_reaction],
                'set': ['train']*sample_size
                }
        new_rows = pd.DataFrame(new_rows)
        auged_df = pd.concat([df, new_rows], ignore_index=True)
        auged_df = df.append(new_rows, ignore_index=True)
        auged_df_filename = args.dataset_info_filename.replace('.csv','.aug_train.csv')
        auged_df.to_csv(auged_df_filename, index=False)
    

    df.to_csv(args.dataset_info_filename,index=False)