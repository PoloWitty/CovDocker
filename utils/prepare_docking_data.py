from openbabel import openbabel
from tempfile import NamedTemporaryFile
import pypdb
from rdkit.Chem import AllChem
from rdkit import Chem

import os
import argparse
from typing import List

import numpy as np

from tqdm import tqdm
import pickle
import pandas as pd
import lmdb

from Bio.PDB import PDBParser

AA_sym = 'Ti'

# modified from https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/main/rf2aa/data/parsers.py#L744
def parse_mol(filename, filetype="pdb", string=False, remove_H=True, generate_conformer: bool = False):
    """Parse small molecule file content.

    Parameters
    ----------
    filename : str
    filetype : str
    string : bool
        If True, `filename` is a string containing the molecule data.
    remove_H : bool
        Whether to remove hydrogen atoms.

    Returns
    -------
    obmol: OBMol
        openbabel molecule object representing the pocket
    sdf_string: str
        converted sdf file string
    atom_types: List[int] (N_atoms, )
    atom_coords: List[List[int]] (N_atoms, 3) float
        Atom coordinates
    bond_table: List[List[int]] (N_bonds, 2)
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(filetype)
    obmol = openbabel.OBMol()
    if string:
        obConversion.ReadString(obmol,filename)
    # elif filetype=='sdf':
    #     molstring = clean_sdffile(filename)
    #     obConversion.ReadString(obmol,molstring)
    else:
        obConversion.ReadFile(obmol,filename)
    if generate_conformer:
        builder = openbabel.OBBuilder()
        builder.Build(obmol)
        ff = openbabel.OBForceField.FindForceField("mmff94")
        did_setup = ff.Setup(obmol)
        if did_setup:
            ff.FastRotorSearch()
            ff.GetCoordinates(obmol)
        else:
            raise ValueError(f"Failed to generate 3D coordinates for molecule {filename}.")
    if remove_H:
        obmol.DeleteHydrogens()
        # the above sometimes fails to get all the hydrogens
        i = 1
        while i < obmol.NumAtoms()+1:
            if obmol.GetAtom(i).GetAtomicNum()==1:
                obmol.DeleteAtom(obmol.GetAtom(i))
            else:
                i += 1
    atom_types = [obmol.GetAtom(i).GetAtomicNum()
                 for i in range(1, obmol.NumAtoms()+1)]
    atom_coords = [[obmol.GetAtom(i).x(),obmol.GetAtom(i).y(), obmol.GetAtom(i).z()] 
                                for i in range(1, obmol.NumAtoms()+1)]# (natoms, 3)
    bond_table = [[b.GetBeginAtomIdx()-1, b.GetEndAtomIdx()-1, b.GetBondOrder()] for b in openbabel.OBMolBondIter(obmol)]
    obConversion.SetOutFormat("sdf")
    sdf_string = obConversion.WriteString(obmol)
    return obmol, sdf_string, atom_types, atom_coords, bond_table

# def delete_extra_atoms(gt_mol, target_mol):
#     edit_mol = Chem.EditableMol(target_mol)
    
#     atoms_to_remove = [AA_idx[idx] for idx in AA_backbone_idx]
#     edit_mol.BeginBatchEdit()
#     for atom in atoms_to_remove:
#         edit_mol.RemoveAtom(atom)
#     edit_mol.CommitBatchEdit()
#     edited_mol = edit_mol.GetMol()
#     Chem.SanitizeMol(edited_mol)
#     return edited_mol

# modified from https://gist.github.com/PatWalters/c046fee2760e6894ed13e19b8c99193b
def process_ligand(ligand_pdb_string, bond_info, template_smiles = None):
    """
    Add bond orders to a pdb ligand
    1. Get the corresponding SMILES from pypdb or input param
    2. Create a template molecule from the SMILES in step 1
    3. Read the pdb_string into an RDKit molecule
    4. Assign the bond orders from the template from step 2
    :param ligand_pdb_string: pdb string of ligand
    :param res_name: residue name of ligand to extract
    :return: molecule with bond orders assigned
    :return: sdf_string of the molecule
    """
    
    if template_smiles == None:
        res_name = pdb_mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName()
        chem_desc = pypdb.describe_chemical(f"{res_name}")
        template_smiles = None
        for item in chem_desc.get('pdbx_chem_comp_descriptor', []):
            if item.get('type') == 'SMILES':
                template_smiles = item.get('descriptor')
                break
    assert template_smiles != None
    
    # bond_ligand_atom, bond_ligand_het, bond_ligand_chain, bond_ligand_chain_idx = bond_info
    pdb_complex = AllChem.MolFromPDBBlock(ligand_pdb_string)
    new_mol, sdf_string = None, None
    for pdb_mol in Chem.GetMolFrags(pdb_complex, asMols=True):
        template = AllChem.MolFromSmiles(template_smiles)
        pdb_info = pdb_mol.GetAtomWithIdx(0).GetPDBResidueInfo()
        pdb_ligand_het, pdb_ligand_chain, pdb_ligand_chain_idx = pdb_info.GetResidueName(), pdb_info.GetChainId(), str(pdb_info.GetResidueNumber())
        if bond_info[1:] != [pdb_ligand_het, pdb_ligand_chain, pdb_ligand_chain_idx]:
            continue
        try:
            new_mol = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
        except:
            continue
        sdf_string = Chem.MolToMolBlock(new_mol)
        if new_mol.GetNumAtoms() == template.GetNumAtoms(): # already found the exact match
            break
    return new_mol, sdf_string

# copy from https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/main/rf2aa/data/covale.py#L299C1-L308C26
def create_and_populate_temp_file(data):
    # Create a temporary file
    with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        # Write the string to the temporary file
        temp_file.write(data)

        # Get the filename
        temp_file_name = temp_file.name

    return temp_file_name

def _find_pocket_bond_idx(pdb_filename, pocket_bond_list, ignore_H=True):
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
    ter_cnt = 0
    H_cnt = 0
    D_cnt = 0
    for i, line in enumerate(lines):
        if line.startswith('TER'):
            ter_cnt += 1
            continue
        name = line[12:16].strip(); resName = line[17:20].strip()
        chainID = line[21]; resSeq = line[22:26].strip()
        if ignore_H and name.startswith('H'):
            H_cnt += 1
            continue
        if name.startswith('D'):
            D_cnt += 1
            continue
        if [name,resName,chainID,resSeq] == pocket_bond_list:
            return i - ter_cnt - H_cnt - D_cnt
    return None

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

def move_center_to(coords:np.array, target_center:List[int]):
    centroid = np.mean(coords, axis=0)

    # Translate the centroid of the point net to the origin
    coords -= centroid

    coords += np.array(target_center)
    return coords

def remove_AA_from_product_smiles(product_smiles: str, aa_smiles: str) -> str:
    if product_smiles == "":
        return ""
    product_mol = Chem.MolFromSmiles(product_smiles); AA_mol = Chem.MolFromSmiles(aa_smiles)
    if product_mol is None or AA_mol is None:
        return ""
    
    AA_idxes = list(product_mol.GetSubstructMatches(AA_mol, maxMatches=10))
    if AA_idxes == []:
        return ""
    
    final_smi = ''
    for atoms_to_remove in AA_idxes:
        edit_mol = Chem.EditableMol(product_mol)
        edit_mol.BeginBatchEdit()
        found_H_idx = False
        for atom in atoms_to_remove:
            atom_obj = product_mol.GetAtomWithIdx(atom)
            # check whether the atom is the connect atom between aa and ligand
            bonds = atom_obj.GetBonds()
            H_idx = -1 # the idx to be connected to H
            for bond in bonds:
                for i in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                    if i not in atoms_to_remove:
                        H_idx = i
                        found_H_idx = True
                        break
            if H_idx !=-1:
                edit_mol.ReplaceAtom(atom, Chem.Atom(1))
            else:
                edit_mol.RemoveAtom(atom)
        assert found_H_idx, f"Cannot find the connect atom between aa and ligand"
        edit_mol.CommitBatchEdit()
        product_mol_woAA = edit_mol.GetMol()
        try:
            Chem.SanitizeMol(product_mol_woAA) # invalid mol
            smi = Chem.MolToSmiles(product_mol_woAA)
        except:
            smi = '.'
        final_smi = smi if '.' not in smi else final_smi
    
    return final_smi


def parse(row, max_len):
    pocket_file = row['bonded pocket']
    ligand_file = row['bonded ligand']
    pdb_id = pocket_file.split('/')[-2] # example: ./data/processed/bonded/4WSJ/4WSJ_pocket.pdb
    # bond example: OD2 ASP D 229-CAF 3U3 D 501
    bond_pocket_atom, bond_pocket_res,_,_ = row['bond'].split('-')[0].split(' ')
    bond_ligand_atom, bond_ligand_het,bond_ligand_chain,bond_ligand_chain_idx = row['bond'].split('-')[1].split(' ')
    
    # post_react_ligand_smi = row['products_woAA'].replace(AA_sym, 'H')
    aa_smiles = row['reactants'].split('.')[0]
    post_react_ligand_smi = remove_AA_from_product_smiles(row['products'], aa_smiles)
    if '[Ru' in post_react_ligand_smi: # can not processed by openbabel
        return None    
    if '[Pd]' in post_react_ligand_smi: # drop ligand contain Pd
        return None
    post_react_ligand_mol = Chem.MolFromSmiles(post_react_ligand_smi)
    Chem.RemoveStereochemistry(post_react_ligand_mol)
    post_react_ligand_mol = Chem.RemoveHs(post_react_ligand_mol)
    post_react_ligand_smi = Chem.MolToSmiles(post_react_ligand_mol)
    
    # parse pocket and ligand files
    pocket_obmol, _, pocket_atoms, pocket_coords, pocket_bonds = parse_mol(pocket_file, remove_H=True)
    ligand_mol, sdf_string = process_ligand(open(ligand_file).read(), [bond_ligand_atom, bond_ligand_het, bond_ligand_chain, bond_ligand_chain_idx], template_smiles=post_react_ligand_smi, )
    if ligand_mol == None:
        return None
    smi = Chem.MolToSmiles(ligand_mol)
    _, _, atoms, holo_coords, bonds = parse_mol(sdf_string, filetype='sdf', string=True)
    assert atoms == [atom.GetAtomicNum() for atom in ligand_mol.GetAtoms()], 'atoms from mol should be the same order as openbabel result'
    if len(atoms) == 1: # will cause no bond
        return None
    max_bonds = np.array(bonds).max() 
    max_pocket_bonds = np.array(pocket_bonds).max()
    if max_bonds + 1 != len(atoms) or max_pocket_bonds + 1 != len(pocket_atoms): # openbabel processor return mismatched result for atoms and bonds
        return None
    ligand_sdf_filename = '.'.join(row['bonded ligand'].split('.')[:-1]) + '.sdf'
    with open(ligand_sdf_filename,'w') as fp:
        fp.write(sdf_string)
    if len(atoms) + len(pocket_atoms) > max_len: # skip too long sequences
        return None

    # get bond info
    bond_ligand_idx = None
    # get bond ligand idx
    for i in range(ligand_mol.GetNumAtoms()):
        atom_name = ligand_mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()
        if atom_name == bond_ligand_atom:
            bond_ligand_idx = i
            break
    assert bond_ligand_idx != None
    # get bond pocket idx
    name, resName, chainID, resSeq = row['bond'].split('-')[0].split(' ')
    bond_pocket_idx = _find_pocket_bond_idx(pocket_file, [name,resName,chainID,resSeq])
    assert bond_pocket_idx != None
    assert pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue().GetName() == bond_pocket_res,f'Expect residue {bond_pocket_res} but got {pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue().GetName()}'
    assert pocket_obmol.GetAtom(bond_pocket_idx+1).GetType()[0] == bond_pocket_atom[0],f'Expect atom {bond_pocket_atom} but got {pocket_obmol.GetAtom(bond_pocket_idx+1).GetType()}'

    # init the ligand coordinates using rdkit
    ligand_mol.RemoveAllConformers()
    ligand_mol = generate_conformation(ligand_mol)
    coords = ligand_mol.GetConformer().GetPositions().tolist()
    
    # abnormal inter-bond distance
    if np.linalg.norm(np.array(holo_coords[bond_ligand_idx]) - np.array(pocket_coords[bond_pocket_idx]) ) > 5:
        return None

    # NEVER DO THIS HERE!!! bug here
    # # move coordinates center to target point
    # coords = move_center_to(np.array(coords), [0,0,0]).tolist()
    # pocket_coords = move_center_to(np.array(pocket_coords), [0,0,0]).tolist()
    # holo_coords = move_center_to(np.array(holo_coords), [0,0,0]).tolist()

    return pickle.dumps(
        {
            "atoms": atoms,
            "coordinates": coords,
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coords,
            "bonds": bonds,
            "pocket_bonds": pocket_bonds,
            "holo_coordinates": holo_coords,
            "holo_pocket_coordinates": pocket_coords,
            "inter_bond": [bond_pocket_idx, bond_ligand_idx],
            "mol": ligand_mol,
            "smi": smi,
            "pocket": pdb_id,
        },
        protocol=-1,
    )

def write_lmdb(dataset_info, outpath="./results", max_len=1022, only_unseen=False):
    os.makedirs(outpath, exist_ok=True)
    failed_pdb = []
    splits = ['train','valid','test'] if not only_unseen else ['unseen']
    for split in splits: 
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
        for i, row in tqdm(split_dataset_info.iterrows(),total=len(split_dataset_info),desc=f'processing and saving to {outputfilename}'):
            # try:
            inner_output = parse(row, max_len)
            if inner_output == None:
                pdb_id = row['bonded protein'].split('/')[-2]
                failed_pdb.append(pdb_id)
                continue
            # except:
            #     print(f"error processing {row['bonded pocket']}, skipping for now")
            #     continue
            txn_write.put(f"{dataset_idx}".encode("ascii"), inner_output)
            dataset_idx += 1
        txn_write.commit()
        env_new.close()
    print(f"failed pdb: {len(failed_pdb)}\n{failed_pdb}")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_info_file", type=str, default="./data/processed/dataset.filtered.random_split.csv")
    parser.add_argument("--save_dir", type=str, default="./data/processed/dataset/docking/")
    parser.add_argument("--max_len", type=int, default=1024-4) # 1024 - bos - eos - bos - eos
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.only_unseen:
        assert 'unseen' in args.dataset_info_file, "only-unseen flag is set, but the dataset_info_file does not contain 'unseen' split"
    
    dataset_info = pd.read_csv(args.dataset_info_file)
    write_lmdb(dataset_info, args.save_dir, args.max_len, args.only_unseen)
