


from openbabel import openbabel
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd
from multiprocessing import Pool
import tqdm
from rdkit.Geometry import Point3D

import argparse
import json
import copy




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

def get_res_pdb(res_obmol):
    ob_conversion = openbabel.OBConversion()
    out_mol = openbabel.OBMol()
    out_mol.AddResidue(res_obmol)
    base_idx = 1e8
    for obatom in openbabel.OBResidueAtomIter(res_obmol):
        out_mol.AddAtom(obatom)
        base_idx = min(base_idx, obatom.GetIdx())
    for obatom in openbabel.OBResidueAtomIter(res_obmol):
        for b in openbabel.OBAtomBondIter(obatom):
            out_mol.AddBond(b.GetBeginAtomIdx()-base_idx+1, b.GetEndAtomIdx()-base_idx+1, b.GetBondOrder())

    ob_conversion.SetOutFormat("pdb")
    pdb_string = ob_conversion.WriteString(out_mol)
    return pdb_string

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

def assign_substruct_coords(mol, match_indices, new_coords):
    # put this coordinates into the substructure of mol
    mol.RemoveAllConformers()
    AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()
    
    for i,mol_idx in enumerate(match_indices):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(mol_idx,Point3D(x,y,z))


def parse(x):
    i,row = x
    name, resName, chainID, resSeq = row['bond'].split('-')[0].split(' ') # SG CYS A 145-C8 T9P A 405
    
    # if resName in ['ASP','GLU','HIS']:
    #     return
    
    pocket_file = row['bonded pocket'].replace('./data/processed/', args.work_dir)
    ligand_file = row['bonded ligand'].replace('./data/processed/', args.work_dir)
    ligcovalent_smi = row['products'].replace('./data/processed/', args.work_dir)
    
    aa_smiles = row['reactants'].split('.')[0]
    
    ligcovalent_mol = Chem.MolFromSmiles(ligcovalent_smi)
    # ligand part
    lig_part_mol = Chem.MolFromMolFile(ligand_file.replace('.pdb','.sdf'))


    # aa part
    pocket_obmol, _, pocket_atoms, pocket_coords, pocket_bonds = parse_mol(pocket_file, remove_H=True)
    bond_pocket_idx = _find_pocket_bond_idx(pocket_file, [name,resName,chainID,resSeq])
    assert bond_pocket_idx != None
    assert pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue().GetName() == resName,f'Expect residue {resName} but got {pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue().GetName()}'
    
    second_bond_pocket_idx = None
    for b in pocket_bonds:
        if bond_pocket_idx in b[:2]:
            second_bond_pocket_idx = b[0] if b[1]==bond_pocket_idx else b[1]
            break
    try:
        assert second_bond_pocket_idx != None
    except:
        return row['pdb_id']
    # aa_pdb_string = get_res_pdb(pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue())
    # aa_pdb_mol = Chem.MolFromPDBBlock(aa_pdb_string)
    # aa_template = Chem.MolFromSmiles(aa_smiles)
    
    
    lig_idx_list = ligcovalent_mol.GetSubstructMatch(lig_part_mol)
    lig_idx = set(lig_idx_list)
    bond_idx = set() # the idx to be connected to aa
    aa_bond_idx = None
    for atom in lig_idx:
        atom_obj = ligcovalent_mol.GetAtomWithIdx(atom)
        #check whether the atom is the connect atom between aa and ligand
        bonds = atom_obj.GetBonds()
        for bond in bonds:
            b = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            connect_idx = b - (b & lig_idx)
            if connect_idx != set():
                bond_idx = b - connect_idx
                aa_bond_idx = connect_idx
                break
    
    # "Cannot find the connect atom between aa and ligand"
    if bond_idx==set():
        return row['pdb_id']
    
    
    second_bond_idx = set()
    for bond in ligcovalent_mol.GetAtomWithIdx(next(iter(bond_idx))).GetBonds():
        second_bond_idx = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]) & lig_idx - bond_idx
        if second_bond_idx != set():
            break
    # "Cannot find the second connect atom between aa and ligand"
    if second_bond_idx==set():
        return row['pdb_id']

    second_aa_bond_idx = set()
    for bond in ligcovalent_mol.GetAtomWithIdx(next(iter(aa_bond_idx))).GetBonds():
        second_aa_bond_idx = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]) - bond_idx - aa_bond_idx
        if second_aa_bond_idx != set():
            break
    # "Cannot find the second connect atom for aa"
    if second_aa_bond_idx==set():
        return row['pdb_id']

    
    # assign coords
    bond_aa_coord = pocket_coords[bond_pocket_idx]
    second_bond_aa_coord = pocket_coords[second_bond_pocket_idx]
    assign_substruct_coords(ligcovalent_mol, list(lig_idx) + list(aa_bond_idx) + list(second_aa_bond_idx), lig_part_mol.GetConformer().GetPositions().tolist() + [bond_aa_coord] + [second_bond_aa_coord])

    # get the ligand part that used in autodock (ligand part plus one bonded aa atom)
    edit_mol = Chem.EditableMol(copy.deepcopy(ligcovalent_mol))
    atoms_to_remove = set(range(ligcovalent_mol.GetNumAtoms())) - lig_idx - set(aa_bond_idx) - set(second_aa_bond_idx)
    edit_mol.BeginBatchEdit()
    for atom in atoms_to_remove:
        edit_mol.RemoveAtom(atom)
    edit_mol.CommitBatchEdit()
    autodock_lig_mol = edit_mol.GetMol()
    
    bond_idx = next(iter(bond_idx))
    aa_bond_idx = next(iter(aa_bond_idx))
    second_aa_bond_idx = next(iter(second_aa_bond_idx))
    
    autodock_lig_idx = ligcovalent_mol.GetSubstructMatch(autodock_lig_mol)
    autodock_bond_idx = autodock_lig_idx.index(bond_idx) + 1 # start from 1 in preprareCovalent.py
    autodock_aa_idx = autodock_lig_idx.index(aa_bond_idx) + 1
    autodock_second_aa_idx = autodock_lig_idx.index(second_aa_bond_idx) + 1
    

    with open(ligand_file.replace('_ligand.pdb','_ligindices.txt'),'w') as fp:
        fp.write(f'{autodock_second_aa_idx},{autodock_aa_idx}') # the order is important for prepareCovalent.py, must be SG-Cb order
    try:    
        # autodock_lig_mol = Chem.AddHs(autodock_lig_mol)
        Chem.SanitizeMol(autodock_lig_mol)
        Chem.MolToMolFile(autodock_lig_mol, ligand_file.replace('.pdb','.autodock_gt.sdf'))
        
        autodock_lig_mol.RemoveAllConformers()
        autodock_lig_mol = generate_conformation(autodock_lig_mol)
        # autodock_lig_mol = Chem.AddHs(autodock_lig_mol)
        Chem.MolToMolFile(autodock_lig_mol, ligand_file.replace('.pdb','.autodock_randomConform.sdf'))
    except:
        return row['pdb_id']

    return True

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default='./data', help='Path to data, note that the output will be in this dir too')
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()

    if args.only_unseen:
        df = pd.read_csv(args.work_dir+'/dataset.unseen.csv')
        df = df[df['set']=='unseen']
    else:
        df = pd.read_csv(args.work_dir+'/dataset.csv')
        df = df[df['set']=='test']
        
    # for x in df.iterrows():
    #     parse(x)
    #     breakpoint()
    
    error_pdb = []
    with tqdm.trange(len(df), desc='prepare random conform and ligindices') as pbar:
        with Pool(10) as pool:
            for inner_output in pool.imap(parse, df.iterrows()):
                if inner_output != True:
                    error_pdb.append(inner_output)
                pbar.update()
    
    print(f"failed pdb {len(error_pdb)}")
    print(error_pdb)
    # ['7AWE', '7KRF', '7FD5', '6YPD', '6Y1N', '6Y49', '7FD4', '7KRC', '6YEN']